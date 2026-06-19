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


def quantize_int8(
    fp32_onnx: str | Path,
    calib_loader: DataLoader,
    out_path: str | Path,
    max_calib_samples: int = 200,
) -> Path:
    """对 FP32 ONNX 做 ORT-QDQ static INT8 量化，返回量化后 ONNX 路径。

    校准样本逐张 (batch=1) 喂入，匹配静态 ONNX；代表性集应含夜视/噪声/压缩样本（plan §C.7）。"""
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    fp32_onnx, out_path = str(fp32_onnx), Path(out_path)
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
        per_channel=True,
        weight_type=QuantType.QInt8,
    )
    return out_path
