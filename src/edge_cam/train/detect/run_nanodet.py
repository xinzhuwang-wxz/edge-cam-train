"""NanoDet 训练/导出的子进程包装（engineering §2 的 ONNX 边界）。

NanoDet 在**自己 pin 的 checkout + 自己的环境**里跑（旧 pytorch-lightning，与本仓 2.6 冲突），
我们只生成 config（指向本仓数据）+ 调它的 tools 脚本 + 消费它导出的 FP32 ONNX。

环境准备（一次）：
    git -C third_party/nanodet checkout be9b4a9   # 锁版（NANODET_PINNED_COMMIT）
    conda create -n nanodet python=3.10 && conda activate nanodet
    pip install -r third_party/nanodet/requirements.txt && pip install -e third_party/nanodet
    # NanoDet-Plus-m 预训练权重（Google Drive，需手动下，见 third_party/nanodet/README）
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def train_nanodet(config_path: str | Path, nanodet_repo: str | Path, nanodet_python: str) -> int:
    """在 NanoDet 环境里跑训练：<nanodet_python> tools/train.py <config>。返回退出码。"""
    repo = Path(nanodet_repo)
    cmd = [nanodet_python, str(repo / "tools" / "train.py"), str(config_path)]
    print(f"[nanodet] $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(repo), check=False).returncode


def export_nanodet_onnx(
    config_path: str | Path,
    checkpoint: str | Path,
    out_onnx: str | Path,
    nanodet_repo: str | Path,
    nanodet_python: str,
    input_shape: tuple[int, int] = (320, 320),
) -> int:
    """调 NanoDet 自带 export_onnx.py 导 FP32 ONNX（后处理 decode/NMS 留 CPU，本仓负责）。"""
    repo = Path(nanodet_repo)
    cmd = [
        nanodet_python,
        str(repo / "tools" / "export_onnx.py"),
        "--cfg_path",
        str(config_path),
        "--model_path",
        str(checkpoint),
        "--out_path",
        str(out_onnx),
        "--input_shape",
        f"{input_shape[0]},{input_shape[1]}",
    ]
    print(f"[nanodet] $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(repo), check=False).returncode
