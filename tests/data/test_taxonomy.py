"""taxonomy seam（ADR-0002）：Identity 占位 + EbirdTaxonomy 真 adapter（≥2 → 真 seam）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from edge_cam.data.taxonomy import EbirdTaxonomy, IdentityTaxonomy


def test_identity_lowercases() -> None:
    assert IdentityTaxonomy().to_taxon_key("ABBOTTS BABBLER") == "abbotts babbler"
    assert IdentityTaxonomy().to_taxon_key("   ") is None


def test_ebird_resolves_to_canonical_code() -> None:
    """源俗名（大小写/空白无关）→ eBird code。"""
    tax = EbirdTaxonomy(
        {"House Sparrow": "houspa", "American Robin": "amerob"}, version="ebird-v2024"
    )
    assert tax.to_taxon_key("house sparrow") == "houspa"  # 归一匹配
    assert tax.to_taxon_key("  AMERICAN   ROBIN ") == "amerob"
    assert tax.version == "ebird-v2024"
    assert tax.coverage_keys == {"houspa", "amerob"}


def test_ebird_unmapped_returns_none() -> None:
    """未映射 → None（调用方决定层级回退/丢弃，不编造键）。"""
    assert EbirdTaxonomy({"House Sparrow": "houspa"}).to_taxon_key("Dodo") is None


def test_ebird_empty_mapping_rejected() -> None:
    with pytest.raises(ValueError, match="映射表为空"):
        EbirdTaxonomy({})


def test_ebird_from_csv(tmp_path: Path) -> None:
    csv = tmp_path / "birds525_ebird.csv"
    csv.write_text(
        "label,ebird_code\nHouse Sparrow,houspa\nAmerican Robin,amerob\n", encoding="utf-8"
    )
    tax = EbirdTaxonomy.from_csv(csv, version="ebird-v2024")
    assert tax.to_taxon_key("House Sparrow") == "houspa"
    assert tax.coverage_keys == {"houspa", "amerob"}


def test_seam_end_to_end_prep_to_regional(flat_imagefolder: Path, tmp_path: Path) -> None:
    """修复验证：eBird 表 → prep manifest 的 taxon_key 是 eBird code → RegionalMask 匹配成功。

    这正是占位假 seam 做不到的：IdentityTaxonomy 下区域清单对不上、会抛交集为 0。"""
    from edge_cam.data.prep import DataPrepConfig, build_manifest
    from edge_cam.eval.regional import RegionalMask

    csv = tmp_path / "map.csv"
    csv.write_text(
        "label,ebird_code\nclass_a,clsaaa\nclass_b,clsbbb\ntiny,tinyyy\n", encoding="utf-8"
    )
    cfg = DataPrepConfig(name="t", root=str(flat_imagefolder), seed=0, taxonomy_csv=str(csv))
    m = build_manifest(cfg)

    taxon_of = {r.label: r.taxon_key for r in m.records if r.taxon_key}
    assert taxon_of["class_a"] == "clsaaa"  # 规范 eBird key，非小写俗名

    # 区域清单用 eBird code → 匹配成功（占位 seam 在此会 ValueError 交集为 0）
    mask = RegionalMask.from_taxon_keys({"clsaaa"}, m.class_to_idx, taxon_of)
    assert mask.allowed_idx == {m.class_to_idx["class_a"]}
