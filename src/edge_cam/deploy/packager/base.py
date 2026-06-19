"""打包后端协议（engineering §3：Protocol + 工厂派发，便于加 AcuityPackager）。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class PackagerBackend(Protocol):
    """FP32 ONNX → 部署产物（如 .nb）。实现见 acuity_packager（板子相关）。"""

    def pack(self, onnx_path: str | Path, out_path: str | Path, calib_dataset: str | Path) -> Path:
        """量化 + 编译为端侧产物，返回产物路径。"""
        ...
