"""TrainerBackend seam（[[ADR-0003]] C1）：工厂派发 + Protocol 合规 + 注册扩展点（纯，快）。"""

from __future__ import annotations

import pytest

from edge_cam.train.backends import (
    ClassifyBackend,
    NanodetBackend,
    TrainerBackend,
    get_backend,
    register_backend,
)


def test_factory_returns_known_backends() -> None:
    assert isinstance(get_backend("classify"), ClassifyBackend)
    assert isinstance(get_backend("nanodet"), NanodetBackend)


def test_unknown_backend_raises_with_listing() -> None:
    with pytest.raises(ValueError, match="未知训练后端"):
        get_backend("does-not-exist")


def test_adapters_satisfy_protocol() -> None:
    # runtime_checkable：两个 adapter 都满足 TrainerBackend → 真 seam（≥2 adapter）
    assert isinstance(ClassifyBackend(), TrainerBackend)
    assert isinstance(NanodetBackend(), TrainerBackend)


def test_register_backend_extends_factory() -> None:
    class DummyBackend:
        def train(self, cfg):  # noqa: ANN001
            return "ref"

        def export_fp32_onnx(self, ref, out_path, input_size):  # noqa: ANN001
            return out_path

    register_backend("dummy", DummyBackend)
    b = get_backend("dummy")
    assert isinstance(b, DummyBackend)
    assert isinstance(b, TrainerBackend)  # 新族即插即用,无需改 caller
