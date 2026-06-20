"""本地 INT8 掉点预估：ORT-QDQ static（engineering §2）。

ACUITY 将消费同形态 ONNX → ORT 静态量化是**方向性**信号，用来在上游淘汰掉点严重
的配置。**真实 INT8 数字必须来自 ACUITY/板子**——本产物不进部署（engineering 铁律）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader


def _collect_calib(loader: DataLoader, max_samples: int) -> list[np.ndarray]:
    """收集校准样本，**逐样本 (1,C,H,W)** 以匹配静态 batch=1 的 ONNX。"""
    samples: list[np.ndarray] = []
    for images, _ in loader:
        arr = images.numpy().astype(np.float32)
        for i in range(arr.shape[0]):
            samples.append(arr[i : i + 1])
            if len(samples) >= max_samples:
                return samples
    return samples


def _ensure_opset13(fp32_onnx: str, out_path: Path) -> str:
    """per-channel QDQ 需 opset≥13;低于则用 version_converter 升级到 13(检测导出常 opset11)。"""
    import onnx
    from onnx import version_converter

    m = onnx.load(fp32_onnx)
    if m.opset_import[0].version >= 13:
        return fp32_onnx
    up = out_path.with_name(out_path.stem + "_op13.onnx")
    onnx.save(version_converter.convert_version(m, 13), str(up))
    return str(up)


def quantize_int8(
    fp32_onnx: str | Path,
    calib_loader: DataLoader,
    out_path: str | Path,
    max_calib_samples: int = 200,
    *,
    per_channel: bool = True,
) -> Path:
    """对 FP32 ONNX 做 ORT-QDQ static INT8 量化，返回量化后 ONNX 路径。

    per_channel：权重逐通道量化(更贴近 ACUITY 板上行为,需 opset≥13,自动升级);False=逐张量。
    校准样本逐张 (batch=1) 喂入，匹配静态 ONNX；代表性集应含夜视/噪声/压缩样本（plan §C.7）。
    不同网络可经 Quantizer seam(eval.quantizers)选不同量化法,config 切换([[ADR-0003]])。"""
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    fp32_onnx, out_path = str(fp32_onnx), Path(out_path)
    if per_channel:
        fp32_onnx = _ensure_opset13(fp32_onnx, out_path)
    sess = ort.InferenceSession(fp32_onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    batches = _collect_calib(calib_loader, max_calib_samples)

    class _Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._it = iter({input_name: b} for b in batches)

        def get_next(self) -> dict | None:  # type: ignore[override]  # ORT 用 None 表结束
            return next(self._it, None)

    quantize_static(
        fp32_onnx,
        str(out_path),
        _Reader(),
        quant_format=QuantFormat.QDQ,
        per_channel=per_channel,
        weight_type=QuantType.QInt8,
    )
    return out_path
