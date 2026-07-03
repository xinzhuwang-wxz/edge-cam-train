"""ingest 注册表（#15 / ADR-0003 #6）：源→manifest 注册-工厂-config 切换。"""

from __future__ import annotations

import pytest

from edge_cam.data.ingest_registry import (
    available_ingests,
    get_ingest,
    register_ingest,
)


def test_builtin_imagefolder_adapter() -> None:
    # 分类走 imagefolder；检测不走本注册表（有自己的 adapter/build，ADR-0006 D0）
    assert "imagefolder" in set(available_ingests())
    assert "coco_detection" not in set(available_ingests())  # 旧 11 类路径已移除
    from edge_cam.data.prep import prepare

    assert get_ingest("imagefolder") is prepare


def test_unknown_source_raises() -> None:
    with pytest.raises(ValueError, match="未知 ingest 源"):
        get_ingest("nope")


def test_register_new_source() -> None:
    register_ingest("fake_src", lambda: lambda cfg: "manifest")
    assert get_ingest("fake_src")(None) == "manifest"  # 新源即插,caller 不改
