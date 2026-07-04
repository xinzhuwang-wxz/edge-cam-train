"""检测数据门（round2 训练前 gate）：量/均衡/框合理/许可/署名/信任 各项 pass+fail。"""

from __future__ import annotations

from edge_cam.contracts.schemas.detection_manifest import (
    FEEDER5_CATEGORIES,
    DetBox,
    DetectionManifest,
    DetImageRecord,
)
from edge_cam.data.gate import gate

_SMALL = {"bird": 2, "squirrel": 1}  # 测试用小目标，避免造上万框


def _rec(cls, box, *, license="CC0", author="a", url="u", prov="gt", w=100, h=100):
    return DetImageRecord(
        path=f"{cls}.jpg",
        split="train",
        width=w,
        height=h,
        boxes=[DetBox(bbox=box, category_id=FEEDER5_CATEGORIES[cls], label_provenance=prov)],
        source="s",
        license=license,
        author=author,
        original_url=url,
    )


def _mani(records):
    return DetectionManifest(
        name="t", version="v0", categories=dict(FEEDER5_CATEGORIES), records=records
    )


def test_gate_pass_all_good():
    """达标 + 均衡 + 框合理 + 商用许可 + 署名 → PASS。"""
    recs = [_rec("bird", [10, 10, 20, 20]) for _ in range(3)]  # bird 3≥2
    recs += [_rec("squirrel", [10, 10, 20, 20])]  # squirrel 1≥1
    r = gate(_mani(recs), min_boxes=_SMALL, max_imbalance=6.0)
    assert r.passed, r.summary()


def test_gate_fails_volume_shortfall():
    """bird 框不够 → 量检查 FAIL。"""
    recs = [_rec("bird", [10, 10, 20, 20])] + [_rec("squirrel", [10, 10, 20, 20])]  # bird 1<2
    r = gate(_mani(recs), min_boxes=_SMALL)
    assert not r.passed
    assert any("数据量" in n and not ok for n, ok, _ in r.checks)


def test_gate_fails_coordinate_bug():
    """框超出图边界（CCT 式坐标错位：框比图大 2 倍）→ 框合理 FAIL。"""
    recs = [_rec("bird", [10, 10, 20, 20]) for _ in range(2)]
    recs += [_rec("squirrel", [10, 10, 400, 400], w=100, h=100)]  # 框 410>100 越界
    r = gate(_mani(recs), min_boxes=_SMALL)
    assert not r.passed
    assert any("框坐标" in n and not ok for n, ok, _ in r.checks)


def test_gate_fails_noncommercial_license():
    """NC 许可 → §4 许可红线 FAIL。"""
    recs = [_rec("bird", [10, 10, 20, 20], license="CC-BY-NC") for _ in range(2)]
    recs += [_rec("squirrel", [10, 10, 20, 20])]
    r = gate(_mani(recs), min_boxes=_SMALL)
    assert not r.passed
    assert any("许可" in n and not ok for n, ok, _ in r.checks)


def test_gate_fails_ccby_without_attribution():
    """CC-BY 图缺 author/url → 署名 FAIL（§4 逐图署名）。"""
    recs = [
        _rec("bird", [10, 10, 20, 20], license="CC-BY-4.0", author=None, url=None) for _ in range(2)
    ]
    recs += [_rec("squirrel", [10, 10, 20, 20])]
    r = gate(_mani(recs), min_boxes=_SMALL)
    assert not r.passed
    assert any("署名" in n and not ok for n, ok, _ in r.checks)


def test_gate_reports_provenance_mix():
    """信任分层：md_pseudo 未审框计入 provenance_mix。"""
    recs = [_rec("bird", [10, 10, 20, 20], prov="md_pseudo") for _ in range(2)]
    recs += [_rec("squirrel", [10, 10, 20, 20], prov="md_human_verified")]
    r = gate(_mani(recs), min_boxes=_SMALL)
    assert r.provenance_mix.get("md_pseudo") == 2
    assert r.provenance_mix.get("md_human_verified") == 1
