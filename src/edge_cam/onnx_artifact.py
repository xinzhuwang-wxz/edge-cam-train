"""FP32 ONNX 产物契约的唯一归宿（铁律：上游只产 FP32 ONNX，INT8/.nb 交 ACUITY）。

统一出口 `export_fp32_onnx`：导出 → **总跑** `check_onnx_contract`（FP32+静态 shape，上 ACUITY
硬契约）→ 有 torch 模型时**加跑** `check_onnx_matches_torch`（数值对齐）。两族共用同一道门：
- 分类（in-process）：有模型对象 → 走全套（导出 + 契约 + 数值对齐）。
- 检测（子进程导出后无 torch 对照）：调方只用 `check_onnx_contract` 校验产物。

[[ADR-0003]] C4：替代此前散在 classify/export.py 与 run_nanodet.py 的两套混淆命名校验。
"""

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
    """torch → 静态 shape FP32 ONNX，返回产物路径（仅导出，不校验）。

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
        dynamo=False,  # 经典 TorchScript 导出器（NPU 工具链更可预测；免 onnxscript 依赖）
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


def check_onnx_contract(path: str | Path) -> bool:
    """结构契约校验：onnx.checker 通过 + 输入为静态 shape（无动态维）。

    「FP32 + 静态 + 可校验」是上 ACUITY 的产物硬契约。无 torch 模型对照时也能跑
    （如检测子进程导出后）；任何族导出后都过这道门。"""
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    for inp in model.graph.input:
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_param:  # 动态维（命名符号）→ 违反静态 shape 契约
                raise ValueError(f"ONNX 含动态维 {dim.dim_param!r}（pegasus 需静态 shape）: {path}")
    return True


def check_onnx_matches_torch(
    path: str | Path, model: nn.Module, input_size: int = 224, atol: float = 1e-3
) -> bool:
    """数值对齐校验：onnxruntime 跑一遍与 PyTorch 输出对齐（CPU）。仅 in-process 有模型时用。"""
    import numpy as np
    import onnxruntime as ort

    model = model.eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        ref = model(dummy).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = np.asarray(sess.run(None, {"input": dummy.numpy()})[0])
    return bool(np.allclose(ref, out, atol=atol))


def export_fp32_onnx(
    model: nn.Module,
    out_path: str | Path,
    *,
    input_size: int = 224,
    opset: int = 13,
    simplify: bool = True,
    check_match: bool = True,
) -> Path:
    """统一出口（in-process）：导出 + 结构契约（总跑）+ 数值对齐（check_match 时跑）。

    检测子进程导出走自己的 export 工具，导出后调方单独跑 `check_onnx_contract`。"""
    path = export_onnx(model, out_path, input_size=input_size, opset=opset, simplify=simplify)
    check_onnx_contract(path)
    if check_match:
        check_onnx_matches_torch(path, model, input_size=input_size)
    return path
