"""eBird 映射脚本：俗名归一（撇号删除、去空格回退）+ build 端到端。"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_ebird_mapping import build, norm


def test_norm_deletes_apostrophe() -> None:
    # 关键：撇号删除而非变空格，让 "Abbott's" 对齐 BIRDS-525 的 "ABBOTTS"
    assert norm("Abbott's Babbler") == "ABBOTTS BABBLER"
    assert norm("ABBOTTS BABBLER") == "ABBOTTS BABBLER"
    assert norm("Anna’s Hummingbird") == "ANNAS HUMMINGBIRD"  # 花式撇号
    assert norm("Black & White Warbler") == "BLACK AND WHITE WARBLER"


def test_build_matches_with_apostrophe_and_despace(tmp_path: Path) -> None:
    tax = tmp_path / "tax.csv"
    tax.write_text(
        "SCIENTIFIC_NAME,COMMON_NAME,SPECIES_CODE,CATEGORY\n"
        "Malacocincla abbotti,Abbott's Babbler,abbbab1,species\n"
        "Haematopus moquini,African Oystercatcher,afroys1,species\n"
        "Dodo dodo,Dodo Bird,dodo1,species\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "t",
                "version": "v0",
                "seed": 0,
                # 注意拼写差异：无撇号 + "OYSTER CATCHER" 分写
                "class_to_idx": {"ABBOTTS BABBLER": 0, "AFRICAN OYSTER CATCHER": 1, "UNKNOWN X": 2},
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "map.csv"
    build(str(tax), str(manifest), str(out))
    rows = out.read_text(encoding="utf-8").splitlines()
    body = {r.split(",")[0]: r.split(",")[1] for r in rows[1:]}
    assert body["ABBOTTS BABBLER"] == "abbbab1"  # 撇号匹配
    assert body["AFRICAN OYSTER CATCHER"] == "afroys1"  # 去空格回退匹配
    assert "UNKNOWN X" not in body  # 无对应 → 不编造


def test_alias_fallback(tmp_path: Path) -> None:
    """人工别名表把笔误类救回（自动匹配不中 → 别名回退）。"""
    tax = tmp_path / "tax.csv"
    tax.write_text(
        "SCIENTIFIC_NAME,COMMON_NAME,SPECIES_CODE,CATEGORY\n"
        "Setophaga fusca,Blackburnian Warbler,bkbwar,species\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "t",
                "version": "v0",
                "seed": 0,
                "class_to_idx": {"BLACKBURNIAM WARBLER": 0},
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    aliases = tmp_path / "alias.csv"
    aliases.write_text("label,ebird_code\nBLACKBURNIAM WARBLER,bkbwar\n", encoding="utf-8")
    out = tmp_path / "map.csv"
    build(str(tax), str(manifest), str(out), aliases=str(aliases))
    rows = out.read_text(encoding="utf-8").splitlines()
    body = {r.split(",")[0]: r.split(",")[1] for r in rows[1:]}
    assert body["BLACKBURNIAM WARBLER"] == "bkbwar"  # 笔误经别名救回 + 补回学名
    assert "Setophaga fusca" in out.read_text(encoding="utf-8")
