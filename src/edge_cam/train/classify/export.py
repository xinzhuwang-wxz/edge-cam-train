"""分类导出 ONNX 的兼容入口（实现已收编进 `edge_cam.onnx_artifact`，[[ADR-0003]] C4）。

历史名保留向后兼容：
- `export_onnx` / `_try_simplify` → onnx_artifact 同名
- `verify_onnx` → `check_onnx_matches_torch`（数值对齐）
- `verify_onnx_loadable` → `check_onnx_contract`（结构契约）
新代码请直接用 `edge_cam.onnx_artifact.export_fp32_onnx`（导出+契约+对齐一步到位）。
"""

from __future__ import annotations

from edge_cam.onnx_artifact import (
    _try_simplify,
    check_onnx_contract,
    check_onnx_matches_torch,
    export_fp32_onnx,
    export_onnx,
)

# 向后兼容别名（旧调用方/测试仍用这些名字）
verify_onnx = check_onnx_matches_torch
verify_onnx_loadable = check_onnx_contract

__all__ = [
    "export_onnx",
    "export_fp32_onnx",
    "verify_onnx",
    "verify_onnx_loadable",
    "check_onnx_contract",
    "check_onnx_matches_torch",
    "_try_simplify",
]
