"""量化 seam（[[ADR-0003]]）：不同网络量化法不同 → 注册一次,config 切换。

与 TrainerBackend / Detector / PackagerBackend 同款"注册-工厂-config 派发"模式:
    q = get_quantizer(cfg.quant.method)   # "ort_qdq_per_channel" / "ort_qdq_per_tensor" / ...
    int8_onnx = q.quantize(fp32_onnx, calib_loader, out)
新量化法(QAT、不同校准、厂商工具)实现 Quantizer 后 register_quantizer 即可,caller 不改。

注意(铁律):这些是**上游 ORT-QDQ 模拟**(方向性);板上真实 INT8 = ACUITY/pegasus(走
deploy.PackagerBackend 的 .nb 路,另一条 seam)。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from torch.utils.data import DataLoader


@runtime_checkable
class Quantizer(Protocol):
    """FP32 ONNX + 校准集 → INT8 ONNX(掉点预估用)。"""

    def quantize(
        self, fp32_onnx: str | Path, calib_loader: DataLoader, out_path: str | Path
    ) -> Path: ...


class OrtQdqQuantizer:
    """ONNXRuntime QDQ static 量化。per_channel 切逐通道(贴近 ACUITY)/逐张量。"""

    def __init__(self, per_channel: bool = True, max_calib_samples: int = 200) -> None:
        self.per_channel = per_channel
        self.max_calib_samples = max_calib_samples

    def quantize(
        self, fp32_onnx: str | Path, calib_loader: DataLoader, out_path: str | Path
    ) -> Path:
        from edge_cam.eval.quant_estimate import quantize_int8

        return quantize_int8(
            fp32_onnx,
            calib_loader,
            out_path,
            max_calib_samples=self.max_calib_samples,
            per_channel=self.per_channel,
        )


_QUANTIZERS: dict[str, Callable[..., Quantizer]] = {
    "ort_qdq_per_channel": OrtQdqQuantizer,  # 默认:逐通道(贴近板上 ACUITY)
    "ort_qdq_per_tensor": lambda **kw: OrtQdqQuantizer(per_channel=False, **kw),
    # "qat": QatQuantizer,  # 未来:量化感知训练等;实现 Quantizer 后注册,config 切换
}


def get_quantizer(name: str = "ort_qdq_per_channel", **kwargs) -> Quantizer:
    """按名取量化器（默认逐通道 ORT-QDQ）。未知名抛 ValueError(含可选清单)。"""
    try:
        return _QUANTIZERS[name](**kwargs)
    except KeyError:
        raise ValueError(
            f"未知量化器 {name!r}；可选：{sorted(_QUANTIZERS)}（新法实现 Quantizer 后注册）"
        ) from None


def register_quantizer(name: str, factory: Callable[..., Quantizer]) -> None:
    """注册新量化法(如 QAT / 厂商工具),让加新法无需改本模块。"""
    _QUANTIZERS[name] = factory
