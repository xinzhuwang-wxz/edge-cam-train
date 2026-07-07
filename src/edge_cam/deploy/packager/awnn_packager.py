"""AWNN 打包器（V861 端侧部署，engineering §6 ★）——取代 ACUITY/pegasus 假设，见 [[ADR-0009]]。

链路：**FP32 logits ONNX → AWNN config(编码 VS861 策略) → `awnntools build` → `_ipu.param/.bin`**。
板端由 AWNN Runtime 加载 `_ipu`；上游铁律不变（只产 FP32 ONNX，INT8 交 AWNN PTQ）。

**VS861 量化策略**（本轮 /loop 实测定案，见 `docs/detect/04-V861-AWNN部署转换.md` §4.1）：
- **混合精度——检测头 `gfl_cls` 卷积保 fp32**：AWNN 默认全 INT8+percentile 会裁掉 cls 头稀有高峰值
  → 丢强检出（松鼠 0.91→0.33 漏检）；头保 fp32 修复 **框召回 4/4 / IoU 0.945**，`_ipu` 仍 1.3MB。
  合 [[ADR-0007]]「保护头」。这是本模块存在的核心理由：**部署=量化，量化策略决定板上精度**。
- 骨干 `symmetric_i8` / `per-channel` / `percentile` 校准。
- `use_npu_preprocess`：把归一化(BGR mean/norm)折进 NPU，板端直接喂 0-255 BGR。

真跑需 AWNN docker 镜像（`awnn:1.0.2`）；arm64 mac 以 `--platform linux/amd64` 模拟即可。
`gen_config` 为纯函数（可单测）；`pack` 负责 docker 编译（副作用）。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml

# feeder 检测器归一化常数（BGR 序，实测自 round2 导出图内 Sub/Div；= (x-mean)/std, norm=1/std）
FEEDER_MEAN_BGR: list[float] = [103.53, 116.28, 123.675]
FEEDER_NORM_BGR: list[float] = [0.017429, 0.017507, 0.017125]
# NanoDet-Plus 检测头 cls/reg 输出卷积（4 个 stride）→ 混合精度保 fp32 的白名单
NANODET_HEAD_FP32: list[str] = [
    "/head/gfl_cls.0/Conv",
    "/head/gfl_cls.1/Conv",
    "/head/gfl_cls.2/Conv",
    "/head/gfl_cls.3/Conv",
]


def gen_config(
    onnx_name: str,
    calib_txt: str,
    *,
    model_dir: str = "/data/onnx",
    output_dir: str = "/data/build_out",
    input_name: str = "data",
    input_shape: tuple[int, int, int, int] = (1, 3, 416, 416),
    mean: list[float] | None = None,
    norm: list[float] | None = None,
    color_space: str = "BGR",
    head_fp32_layers: list[str] | None = None,
    use_npu_preprocess: bool = True,
    calibration_algorithm: str = "percentile",
) -> dict[str, Any]:
    """生成 AWNN 编译配置（纯函数）。默认即 VS861 检测部署策略（头 fp32 混合精度 + npu 预处理）。

    容器内路径约定：工作区挂载到 `/data`。`head_fp32_layers=[]` 可关混合精度（退回全 INT8）。
    """
    mean = FEEDER_MEAN_BGR if mean is None else mean
    norm = FEEDER_NORM_BGR if norm is None else norm
    head_fp32 = NANODET_HEAD_FP32 if head_fp32_layers is None else head_fp32_layers

    preprocess = [
        {
            "name": input_name,
            "color_space": color_space,
            "mean": list(mean),
            "norm": list(norm),
            "tensor_layout": "NCHW",
            "shape": list(input_shape),
        }
    ]
    dataset = [
        {
            "type": "DATASET_TYPE_TXT",
            "path": calib_txt,
            "image_types": [],
            "preprocess_conf": preprocess,
        }
    ]
    build_conf: dict[str, Any] = {
        "build_mode": "auto",
        "export_type": "standard",
        "debug_enable": True,
        "enable_onnxsim": True,
        "use_npu_preprocess": use_npu_preprocess,
        "quantize_conf": {
            "quantized_dtype": "symmetric_i8",
            "calibration_algorithm": calibration_algorithm,
            "quantized_method": "per-channel",
        },
        "calibration_engine": "default",
        "dataset_conf": dataset,
        "opt_level": 1,
    }
    if head_fp32:  # 混合精度：检测头保 fp32（VS861 策略核心）
        build_conf["hybrid_quantization_conf"] = {"white_list": list(head_fp32)}

    stem = Path(onnx_name).stem
    return {
        "general_conf": {
            "model_type": "onnx",
            "model_path": model_dir,
            "model_names": [onnx_name],
            "output": f"{output_dir}/{stem}",
        },
        "build_conf": build_conf,
    }


def write_config(cfg: dict[str, Any], path: str | Path) -> Path:
    """落盘 config.yml。AWNN 硬要求：**禁用 Tab**（yaml.safe_dump 用空格缩进，天然满足）。"""
    path = Path(path)
    path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, default_flow_style=False)
    )
    return path


class AwnnPackager:
    """实现 `PackagerBackend`：FP32 ONNX → V861 `_ipu.param/.bin`（AWNN docker PTQ）。

    workspace 需含 `onnx/<model>.onnx` 与校准清单；产物落 `build_out/<stem>/<stem>_ipu.param|bin`。
    """

    def __init__(
        self, image: str = "awnn:1.0.2", platform: str = "linux/amd64", cpus: int | None = 3
    ) -> None:
        self.image = image
        self.platform = platform  # arm64 mac 上以 amd64 模拟
        self.cpus = cpus  # 限核降温（None=不限）

    def pack(self, onnx_path: str | Path, out_path: str | Path, calib_dataset: str | Path) -> Path:
        """workspace = onnx 祖父目录（含 onnx/ 与 calib/）。calib_dataset 为容器内 /data 路径。"""
        onnx_path = Path(onnx_path)
        workspace = onnx_path.parent.parent  # <ws>/onnx/model.onnx → <ws>
        cfg = gen_config(onnx_path.name, str(calib_dataset), output_dir="/data/build_out")
        cfg_dir = workspace / "configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_host = write_config(cfg, cfg_dir / f"{onnx_path.stem}_v861.yml")
        cfg_cont = f"/data/configs/{cfg_host.name}"
        cmd = ["docker", "run", "--rm", "--platform", self.platform]
        if self.cpus:
            cmd += ["--cpus", str(self.cpus)]
        cmd += [
            "-v",
            f"{workspace}:/data",
            self.image,
            "bash",
            "-lc",
            f"cd /data && awnntools build {cfg_cont}",
        ]
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            raise RuntimeError(
                f"awnntools build 失败 (rc={rc})；确认 docker 镜像 {self.image} 已加载。cmd={cmd}"
            )
        stem = onnx_path.stem
        ipu = workspace / "build_out" / stem / f"{stem}_ipu.param"
        if not ipu.exists():
            raise RuntimeError(f"未产出 _ipu：{ipu}")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        return ipu
