"""feeder-cam 大类清单 + COCO/OIV7 映射（纯函数）。"""

from __future__ import annotations

from edge_cam.data.detection_classes import (
    CORE_CLASSES,
    FEEDER_CAM_CLASSES,
    DetectionDataConfig,
    class_index,
    coco_to_unified,
    oiv7_to_unified,
    source_labels,
)


def test_bird_is_index_zero() -> None:
    idx = class_index()
    assert idx["bird"] == 0
    assert len(idx) == len(FEEDER_CAM_CLASSES)


def test_core_classes_present() -> None:
    names = {c.name for c in FEEDER_CAM_CLASSES}
    assert set(CORE_CLASSES) <= names


def test_no_livestock() -> None:
    # 观鸟器见不到 livestock
    names = {c.name for c in FEEDER_CAM_CLASSES}
    assert names.isdisjoint({"horse", "cow", "sheep", "zebra", "giraffe", "elephant"})


def test_coco_mapping() -> None:
    m = coco_to_unified()
    assert m["bird"] == "bird"
    assert m["bear"] == "bear"
    assert "horse" not in m  # 未纳入


def test_oiv7_mapping() -> None:
    m = oiv7_to_unified()
    assert m["Squirrel"] == "squirrel"
    assert m["Raccoon"] == "raccoon"
    assert m["Hedgehog"] == "hedgehog"


def test_source_labels_coco_only_supported() -> None:
    coco = source_labels("coco")
    assert set(coco) == {"bird", "cat", "dog", "bear"}  # COCO 只有这几类 feeder 相关


def test_config_selected_filters() -> None:
    config = DetectionDataConfig(classes=["bird", "squirrel"])
    selected = config.selected()
    assert {c.name for c in selected} == {"bird", "squirrel"}
    assert source_labels("oiv7", selected) == ["Bird", "Squirrel"]
