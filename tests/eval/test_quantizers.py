"""量化 seam（[[ADR-0003]]）：注册-工厂-config 切换(纯,不跑真量化)。"""

from __future__ import annotations

import pytest

from edge_cam.eval.quantizers import (
    OrtQdqQuantizer,
    Quantizer,
    get_quantizer,
    register_quantizer,
)


def test_default_is_per_channel() -> None:
    q = get_quantizer()  # 默认
    assert isinstance(q, OrtQdqQuantizer) and q.per_channel is True


def test_config_switch_per_tensor() -> None:
    q = get_quantizer("ort_qdq_per_tensor")
    assert isinstance(q, OrtQdqQuantizer) and q.per_channel is False


def test_unknown_quantizer_raises() -> None:
    with pytest.raises(ValueError, match="未知量化器"):
        get_quantizer("nope")


def test_protocol_and_register() -> None:
    assert isinstance(OrtQdqQuantizer(), Quantizer)

    class FakeQ:
        def quantize(self, fp32_onnx, calib_loader, out_path):  # noqa: ANN001
            return out_path

    register_quantizer("fake", FakeQ)
    q = get_quantizer("fake")
    assert isinstance(q, FakeQ) and isinstance(q, Quantizer)  # 新法即插即用
