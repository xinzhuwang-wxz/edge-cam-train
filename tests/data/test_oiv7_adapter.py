"""OIV7 FiftyOne adapter：纯逻辑（相对→绝对框、样本转换、spec/映射、注册）——不调 fiftyone。"""

from __future__ import annotations

from edge_cam.data.adapters.detect import FEEDER5_CATEGORIES, available_adapters, build_adapter
from edge_cam.data.adapters.detect.fiftyone_oiv7 import (
    OIV7_LABEL_MAP,
    FiftyOneOiv7Adapter,
    _rel_to_abs,
)


def test_rel_to_abs_bbox() -> None:
    assert _rel_to_abs([0.1, 0.2, 0.5, 0.25], 200, 100) == [20.0, 20.0, 100.0, 25.0]


def test_sample_to_raw_converts_boxes_to_absolute() -> None:
    raw = FiftyOneOiv7Adapter._sample_to_raw(
        "/fo/cache/x.jpg", 100, 100, [("Bird", [0.0, 0.0, 0.5, 0.5]), ("Dog", [0.5, 0.5, 0.5, 0.5])]
    )
    assert raw.path == "/fo/cache/x.jpg"
    assert raw.boxes == [("Bird", [0.0, 0.0, 50.0, 50.0]), ("Dog", [50.0, 50.0, 50.0, 50.0])]


def test_label_map_uses_animal_mouse_not_computer_mouse() -> None:
    assert OIV7_LABEL_MAP["Mouse"] == "other_animal"
    assert "Computer mouse" not in OIV7_LABEL_MAP  # 不能误收电脑鼠标
    assert set(OIV7_LABEL_MAP.values()) <= set(FEEDER5_CATEGORIES)


def test_direct_adapter_is_default_commercial_nonexhaustive() -> None:
    # open_images_v7 默认 = 直下版（绕开 fiftyone）；语义与 fiftyone 版一致
    ad = build_adapter("open_images_v7", "/root/x")
    assert ad.spec.commercial_safe and ad.spec.role == "train"
    assert ad.spec.exhaustive is False  # 按类拉 → 非穷尽
    assert ad.spec.attribution is True  # CC-BY 逐图署名
    assert ad.spec.license == "CC-BY-4.0"


def test_fiftyone_adapter_constructs_with_max_samples() -> None:
    ad = FiftyOneOiv7Adapter("/tmp/fo_cache", max_samples=10)
    assert ad.spec.name == "open_images_v7" and ad.max_samples == 10


def test_registered() -> None:
    reg = available_adapters()
    assert "open_images_v7" in reg  # 直下版（默认）
    assert "open_images_v7_fiftyone" in reg  # fiftyone 版（py3.10+ 备用）
