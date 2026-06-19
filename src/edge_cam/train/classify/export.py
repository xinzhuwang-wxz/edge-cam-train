"""导出 FP32 ONNX（engineering 铁律：上游只产 FP32 ONNX，INT8 交 ACUITY）。

链路：PyTorch → ONNX → (onnxsim 静态化, 可选) → onnxruntime 自检。
后续 slice 4 用 ORT-QDQ 在此 ONNX 上模拟 INT8 掉点（不进部署）。"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def export_onnx(
    model: nn.Module,
    out_path: str | Path,
    input_size: int = 224,
    opset: int = 13,
    simplify: bool = True,
) -> Path:
    """导出静态 shape 的 FP32 ONNX，返回产物路径。

    model 须已 eval()；输入约定 NCHW=1×3×size×size（端侧串行单帧）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    dummy = torch.randn(1, 3, input_size, input_size)

    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes=None,  # 静态 shape：pegasus 友好
        dynamo=False,  # 用经典 TorchScript 导出器（NPU 工具链更可预测；免 onnxscript 依赖）
    )

    if simplify:
        _try_simplify(out_path)
    return out_path


def _try_simplify(path: Path) -> None:
    """onnxsim 静态化简化；未安装则跳过（不致命）。"""
    try:
        import onnx
        import onnxsim
    except ImportError:
        return
    model = onnx.load(str(path))
    simplified, ok = onnxsim.simplify(model)
    if ok:
        onnx.save(simplified, str(path))


def verify_onnx(
    path: str | Path, model: nn.Module, input_size: int = 224, atol: float = 1e-3
) -> bool:
    """onnxruntime 跑一遍与 PyTorch 输出对齐校验（CPU）。classify 进程内导出用（有 torch 对照）。"""
    import numpy as np
    import onnxruntime as ort

    model = model.eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        ref = model(dummy).numpy()

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = np.asarray(sess.run(None, {"input": dummy.numpy()})[0])
    return bool(np.allclose(ref, out, atol=atol))


def verify_onnx_loadable(path: str | Path) -> bool:
    """结构校验：onnx.checker 通过 + 输入为静态 shape（无动态维）。

    「FP32 + 静态 + 可校验」是上 ACUITY 的产物契约（架构审查 C）。无 torch 模型对照时用
    （如 detect 子进程导出后）；classify 侧另有 verify_onnx 做输出对齐。
    """
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    for inp in model.graph.input:
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_param:  # 动态维（命名符号）→ 违反静态 shape 契约
                raise ValueError(f"ONNX 含动态维 {dim.dim_param!r}（pegasus 需静态 shape）: {path}")
    return True
