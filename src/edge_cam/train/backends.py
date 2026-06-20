"""模型族训练 seam（[[ADR-0003]] C1）：把「训练 + 导 FP32 ONNX」收成一个 Protocol。

每族一个 adapter（分类 in-process / NanoDet subprocess / 未来 YOLO），工厂按名派发；
消融/发布/级联 caller 只认 `TrainerBackend`，加模型族不改 caller。

- `train(cfg) -> TrainedRef`：不透明句柄（分类=内存模型；subprocess=带路径的 ref）。
- `export_fp32_onnx(ref, out, input_size) -> Path`：经 `edge_cam.onnx_artifact` 过 FP32 契约门。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from omegaconf import DictConfig

TrainedRef = Any  # 族自定义的不透明句柄；caller 只把 train 的产物原样喂给 export


@runtime_checkable
class TrainerBackend(Protocol):
    """一个模型族的训练+导出 seam。"""

    def train(self, cfg: DictConfig) -> TrainedRef:
        """执行训练，返回供 export 用的句柄（不导出，导出归发布路）。"""
        ...

    def export_fp32_onnx(self, ref: TrainedRef, out_path: str | Path, input_size: int) -> Path:
        """把训练产物导成 FP32 ONNX（过 onnx_artifact 契约门），返回产物路径。"""
        ...


class ClassifyBackend:
    """timm + Lightning 分类族（in-process）。包 train/classify 现有逻辑，行为不变。"""

    def train(self, cfg: DictConfig) -> TrainedRef:
        from edge_cam.train.classify.train import run

        return run(cfg)

    def export_fp32_onnx(self, ref: TrainedRef, out_path: str | Path, input_size: int) -> Path:
        from edge_cam.onnx_artifact import export_fp32_onnx

        return export_fp32_onnx(ref, out_path, input_size=input_size)


@dataclass
class NanodetRef:
    """NanoDet 训练产物句柄（subprocess 无内存模型 → 带齐 export 所需路径）。"""

    checkpoint: str | Path
    config_path: str | Path
    nanodet_repo: str | Path
    nanodet_python: str


class NanodetBackend:
    """NanoDet 检测族（subprocess，独立 env）。包 run_nanodet，行为不变。

    cfg 需含 detect.{config_path, checkpoint, nanodet_repo, nanodet_python}。"""

    def train(self, cfg: DictConfig) -> NanodetRef:
        from edge_cam.train.detect.run_nanodet import train_nanodet

        d = cfg.detect
        code = train_nanodet(d.config_path, d.nanodet_repo, d.nanodet_python)
        if code != 0:
            raise RuntimeError(f"NanoDet 训练失败（exit {code}）")
        return NanodetRef(d.checkpoint, d.config_path, d.nanodet_repo, d.nanodet_python)

    def export_fp32_onnx(self, ref: NanodetRef, out_path: str | Path, input_size: int) -> Path:
        from edge_cam.train.detect.run_nanodet import export_nanodet_onnx

        code = export_nanodet_onnx(
            ref.config_path,
            ref.checkpoint,
            out_path,
            ref.nanodet_repo,
            ref.nanodet_python,
            input_shape=(input_size, input_size),
        )
        if code != 0:
            raise RuntimeError(f"NanoDet 导出 ONNX 失败（exit {code}）")
        return Path(out_path)


_BACKENDS: dict[str, type] = {
    "classify": ClassifyBackend,
    "nanodet": NanodetBackend,
    # "yolo": YoloBackend,   # 未来新检测器：实现 TrainerBackend 后在此注册（caller 不改）
}


def get_backend(name: str) -> TrainerBackend:
    """按名取训练后端。未知名抛 ValueError（含可选清单，便于排错）。"""
    try:
        return _BACKENDS[name]()
    except KeyError:
        raise ValueError(
            f"未知训练后端 {name!r}；可选：{sorted(_BACKENDS)}（新族实现 TrainerBackend 后注册）"
        ) from None


def register_backend(name: str, backend_cls: type) -> None:
    """注册新模型族后端（如 YoloBackend）。让加新检测器无需改本模块。"""
    _BACKENDS[name] = backend_cls
