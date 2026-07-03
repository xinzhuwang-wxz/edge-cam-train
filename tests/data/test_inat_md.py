"""iNat + MD 伪标注 adapter（ADR-0006 D7 补 bird 覆盖）：选图过滤 + MD-COCO 消费 + 信任分层。"""

from __future__ import annotations

import json

from edge_cam.data.adapters.detect import FEEDER5_CATEGORIES, build_adapter
from edge_cam.data.adapters.detect.inat_md import InatMdAdapter, InatObs, select_inat


def _obs(pid, taxon, lic="CC-BY-4.0", lat=1.0, lon=2.0, grade="research"):
    return InatObs(photo_id=pid, taxon_id=taxon, license=lic, lat=lat, lon=lon, quality_grade=grade)


def test_select_inat_license_tightened_to_cc0_ccby() -> None:
    """商用收紧：NC 变体被拒（§4，去传染）。"""
    obs = [_obs("1", "t1", lic="CC-BY-NC"), _obs("2", "t1", lic="CC0-1.0")]
    kept = select_inat(obs, per_taxon_cap=10)
    assert [o.photo_id for o in kept] == ["2"]  # 只留 CC0


def test_select_inat_research_and_geo_gates() -> None:
    obs = [
        _obs("1", "t1", grade="needs_id"),  # 非 research → 丢
        _obs("2", "t1", lat=None),  # 无 geo → 丢
        _obs("3", "t1"),  # 过
    ]
    assert [o.photo_id for o in select_inat(obs, per_taxon_cap=10)] == ["3"]


def test_select_inat_per_taxon_cap_treats_longtail() -> None:
    """per-taxon 配额：常见种封顶、稀有种全留（确定性保序）。"""
    obs = [_obs(str(i), "common") for i in range(5)] + [_obs("r", "rare")]
    kept = select_inat(obs, per_taxon_cap=2)
    ids = [o.photo_id for o in kept]
    assert ids == ["0", "1", "r"]  # common 封顶 2、rare 全留


def test_declares_inat_acquire() -> None:
    spec = build_adapter("inat_md", "raw").spec
    assert spec.acquire is not None and spec.acquire.method == "inat_open_data"
    assert spec.commercial_safe is True and spec.exhaustive is False  # iNat 非穷尽标注


def test_md_coco_boxes_tagged_md_pseudo(tmp_path) -> None:
    """读 MD 伪标注 COCO：animal→bird 映射 + 框 label_provenance=md_pseudo（信任分层）。"""
    base = tmp_path / "commercial" / "inat_md"
    base.mkdir(parents=True)
    (base / "inat_md_coco.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "x.jpg", "width": 100, "height": 100}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 9, 9]}],
                "categories": [{"id": 1, "name": "animal"}],  # MD 出 animal
            }
        ),
        encoding="utf-8",
    )
    recs = InatMdAdapter(str(tmp_path)).build_records()
    assert len(recs) == 1
    assert recs[0].boxes[0].category_id == FEEDER5_CATEGORIES["bird"]  # animal→bird
    assert recs[0].boxes[0].label_provenance == "md_pseudo"  # 框来源=MD 伪标注


def test_human_verified_provenance_override(tmp_path) -> None:
    """人审通过 → label_provenance=md_human_verified（Label Studio 后）。"""
    base = tmp_path / "commercial" / "inat_md"
    base.mkdir(parents=True)
    (base / "inat_md_coco.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "x.jpg", "width": 50, "height": 50}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 5, 5]}],
                "categories": [{"id": 1, "name": "bird"}],
            }
        ),
        encoding="utf-8",
    )
    recs = InatMdAdapter(str(tmp_path), label_provenance="md_human_verified").build_records()
    assert recs[0].boxes[0].label_provenance == "md_human_verified"
