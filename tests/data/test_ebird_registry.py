"""registry→Hierarchy 桥（R1.1）：命门尺子接真实 eBird 层级树。先红后绿。

registry 是命门层级 roll-up 的数据地基（ADR-0002）——树错了层级可用率就错，
故按 §5 安全面纪律 TDD。含一条打真实 vendored registry 的 smoke。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from edge_cam.data.ebird_registry import DEFAULT_ROOT, EbirdRegistry
from edge_cam.eval.hierarchical import Hierarchy, hierarchical_usability


def _fake_registry(tmp_path: Path) -> EbirdRegistry:
    """5 类小 registry：genus houspa/eutspa=Passer；amerob/eurbla=Turdus；greti=Parus。"""
    recs = [
        {
            "ebird_code": "houspa",
            "genus": "Passer",
            "family_code": "passer1",
            "sci_name": "Passer domesticus",
        },
        {
            "ebird_code": "eutspa",
            "genus": "Passer",
            "family_code": "passer1",
            "sci_name": "Passer montanus",
        },
        {
            "ebird_code": "amerob",
            "genus": "Turdus",
            "family_code": "turdid1",
            "sci_name": "Turdus migratorius",
        },
        {
            "ebird_code": "eurbla",
            "genus": "Turdus",
            "family_code": "turdid1",
            "sci_name": "Turdus merula",
        },
        {
            "ebird_code": "greti",
            "genus": "Parus",
            "family_code": "paridae1",
            "sci_name": "Parus major",
        },
    ]
    p = tmp_path / "species.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    return EbirdRegistry.load(tmp_path)


def test_lookup_genus_family(tmp_path: Path) -> None:
    reg = _fake_registry(tmp_path)
    assert reg.genus("houspa") == "Passer"
    assert reg.family("amerob") == "turdid1"  # family_code（稳定键，非学名）
    assert reg.genus("dodo") is None  # 未知码 → None，不编造


def test_coverage_splits_present_missing(tmp_path: Path) -> None:
    reg = _fake_registry(tmp_path)
    present, missing = reg.coverage(["houspa", "dodo", "greti"])
    assert present == ["houspa", "greti"]
    assert missing == ["dodo"]


def test_hierarchy_arrays_aligned_to_class_order(tmp_path: Path) -> None:
    """genus/family 数组须与传入类顺序逐位对齐（= 模型 logits 类顺序）。"""
    reg = _fake_registry(tmp_path)
    codes = ["greti", "houspa", "amerob"]  # 故意打乱
    genus, family = reg.hierarchy_arrays(codes)
    assert genus == ["Parus", "Passer", "Turdus"]
    assert family == ["paridae1", "passer1", "turdid1"]


def test_hierarchy_arrays_missing_code_raises(tmp_path: Path) -> None:
    """缺失码不静默编造键 → 报错（调用方先 coverage() 剔除）。"""
    reg = _fake_registry(tmp_path)
    with pytest.raises(KeyError):
        reg.hierarchy_arrays(["houspa", "dodo"])


def test_hierarchy_from_registry_builds_tree(tmp_path: Path) -> None:
    reg = _fake_registry(tmp_path)
    hier = Hierarchy.from_registry(["houspa", "eutspa", "amerob"], reg)
    assert hier.num_classes == 3
    assert hier.genus == ["Passer", "Passer", "Turdus"]


def test_bridge_wires_into_hierarchical_usability(tmp_path: Path) -> None:
    """端到端：registry 建的树喂命门 metric，属级 roll-up 生效（别先造完再用的证据）。"""
    reg = _fake_registry(tmp_path)
    hier = Hierarchy.from_registry(["houspa", "eutspa", "amerob", "eurbla", "greti"], reg)
    # 两个 Passer 种各 0.35/0.30（种级均 <0.5 不过门），属 Passer=0.65 过门；真值=houspa(属 Passer)
    probs = torch.log(torch.tensor([[0.35, 0.30, 0.15, 0.12, 0.08]]))
    m = hierarchical_usability(probs, torch.tensor([0]), hier)
    assert m.report_genus == 1 and m.usable_rate == 1.0  # 回退到正确的属 → 可用


def test_real_vendored_registry_smoke() -> None:
    """打真实 vendored registry（data/taxonomy/ebird_clements_2025）。"""
    if not (DEFAULT_ROOT / "species.jsonl").exists():
        pytest.skip("vendored registry 未就位")
    reg = EbirdRegistry.load()
    assert len(reg.species) > 11000  # 全 eBird ~11,167
    assert reg.genus("ostric2") == "Struthio"  # 已知种
    assert reg.family("ostric2") == "struth1"
    assert "eBird/Clements" in reg.version and "2025" in reg.version
