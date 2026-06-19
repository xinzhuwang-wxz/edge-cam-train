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
    """用 onnxruntime 跑一遍，与 PyTorch 输出对齐校验（CPU）。"""
    import numpy as np
    import onnxruntime as ort

    model = model.eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        ref = model(dummy).numpy()

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = np.asarray(sess.run(None, {"input": dummy.numpy()})[0])
    return bool(np.allclose(ref, out, atol=atol))
