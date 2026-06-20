"""ingest 注册表（#15 / ADR-0003 #6）：源→manifest 注册-工厂-config 切换。"""

from __future__ import annotations

import pytest

from edge_cam.data.ingest_registry import (
    available_ingests,
    get_ingest,
    register_ingest,
)


def test_builtin_two_real_adapters() -> None:
    # ≥2 真 adapter = 真 seam(分类 imagefolder / 检测 coco_detection)
    assert {"imagefolder", "coco_detection"} <= set(available_ingests())
    from edge_cam.data.detection_ingest import build_detection_manifest
    from edge_cam.data.prep import prepare

    assert get_ingest("imagefolder") is prepare
    assert get_ingest("coco_detection") is build_detection_manifest


def test_unknown_source_raises() -> None:
    with pytest.raises(ValueError, match="未知 ingest 源"):
        get_ingest("nope")


def test_register_new_source() -> None:
    register_ingest("fake_src", lambda: lambda cfg: "manifest")
    assert get_ingest("fake_src")(None) == "manifest"  # 新源即插,caller 不改
