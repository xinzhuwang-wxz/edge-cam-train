"""acquire CLI + 4 源 acquire 声明（ADR-0006 D1/D3）：数据来源单一事实源、可插拔。"""

from __future__ import annotations

from edge_cam.data.adapters.detect.acquire import list_sources
from edge_cam.data.adapters.detect.base import build_adapter

# 每个内置源期望的 acquire method（open_images_v7 有自动下载器 → s3_direct；LILA/COCO → manual）
_EXPECTED = {
    "ena24": "manual",
    "caltech_ct": "manual",
    "coco2017": "manual",
    "open_images_v7": "s3_direct",
}


def test_builtin_adapters_declare_acquire() -> None:
    """4 个内置源都声明了 acquire（method + 非空 urls）——数据从哪来在代码里，非 prose。"""
    for name, method in _EXPECTED.items():
        spec = build_adapter(name, "raw").spec
        assert spec.acquire is not None, f"{name} 未声明 acquire"
        assert spec.acquire.method == method
        assert spec.acquire.urls, f"{name} acquire.urls 为空"


def test_list_sources_covers_builtins() -> None:
    """--list 从各 adapter 的 spec 汇出全源清单（无需另维护 sources.yaml）。"""
    by_name = {s["name"]: s for s in list_sources("raw")}
    for name, method in _EXPECTED.items():
        assert name in by_name, f"{name} 不在 list_sources"
        acq = by_name[name]["acquire"]
        assert acq is not None and acq["method"] == method
    # 溯源关键字段随清单披露
    assert by_name["coco2017"]["role"] == "eval_only"
    assert by_name["open_images_v7"]["commercial_safe"] is True


def test_oiv7_fetch_missing_bbox_csv_raises_actionable(tmp_path) -> None:
    """OIV7 获取逻辑已折进 adapter（旧脚本已删）：缺 bbox CSV → 可执行错误（含 URL）。"""
    ad = build_adapter("open_images_v7", str(tmp_path))
    dest = ad.raw_dir(tmp_path)
    dest.mkdir(parents=True)
    import pytest

    with pytest.raises(FileNotFoundError, match="bbox CSV"):
        ad._fetch(dest)  # bbox CSV 不在 dest → 抛（faithfully，取代旧脚本的同款前置）
