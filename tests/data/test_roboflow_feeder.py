"""Roboflow 喂鸟器域 adapter（ADR-0006 D7 补 feeder 短板）：多 split 合并 + 5 类映射 + acquire。"""

from __future__ import annotations

import json

from edge_cam.data.adapters.detect import FEEDER5_CATEGORIES, build_adapter
from edge_cam.data.adapters.detect.roboflow_feeder import RoboflowFeederAdapter


def _write_split(base, split, images, anns):
    d = base / split
    d.mkdir(parents=True)
    (d / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "images": images,
                "annotations": anns,
                "categories": [{"id": 1, "name": "bird"}, {"id": 2, "name": "squirrel"}],
            }
        ),
        encoding="utf-8",
    )


def _export(tmp_path):
    base = tmp_path / "commercial" / "roboflow_feeder"
    # 两个 split 的 image id 都从 0/1 开始 → 测全局重编号防冲突
    _write_split(
        base,
        "train",
        images=[{"id": 0, "file_name": "a.jpg", "width": 100, "height": 100}],
        anns=[{"id": 0, "image_id": 0, "category_id": 1, "bbox": [1, 1, 5, 5]}],
    )
    _write_split(
        base,
        "valid",
        images=[{"id": 0, "file_name": "b.jpg", "width": 100, "height": 100}],
        anns=[{"id": 0, "image_id": 0, "category_id": 2, "bbox": [2, 2, 6, 6]}],
    )
    return base


def test_declares_roboflow_acquire() -> None:
    spec = build_adapter("roboflow_feeder", "raw").spec
    assert spec.acquire is not None and spec.acquire.method == "roboflow"
    assert spec.name == "roboflow_feeder" and spec.commercial_safe is True
    assert spec.attribution is True  # CC-BY 逐图署名


def test_merge_multi_split_remaps_ids(tmp_path) -> None:
    _export(tmp_path)
    ad = RoboflowFeederAdapter(str(tmp_path))
    coco = ad._load_coco()
    assert len(coco["images"]) == 2  # train+valid 合并
    ids = [im["id"] for im in coco["images"]]
    assert len(set(ids)) == 2  # 全局重编号，无冲突（原来都是 0）
    # file_name 前缀 split 目录名（与磁盘布局一致）
    names = {im["file_name"] for im in coco["images"]}
    assert names == {"train/a.jpg", "valid/b.jpg"}
    # annotation 的 image_id 正确重指
    for a in coco["annotations"]:
        assert a["image_id"] in set(ids)


def test_build_records_maps_to_5class(tmp_path) -> None:
    _export(tmp_path)
    recs = RoboflowFeederAdapter(str(tmp_path)).build_records()
    assert len(recs) == 2
    cids = {b.category_id for r in recs for b in r.boxes}
    assert cids == {FEEDER5_CATEGORIES["bird"], FEEDER5_CATEGORIES["squirrel"]}
    # 框来源默认 gt（人工标注）、路径含 image_root 前缀
    assert all(b.label_provenance == "gt" for r in recs for b in r.boxes)
    assert all(r.path.startswith("commercial/roboflow_feeder/") for r in recs)


def test_audit_unmapped_flags_unknown(tmp_path) -> None:
    base = tmp_path / "commercial" / "roboflow_feeder"
    _write_split(
        base,
        "train",
        images=[{"id": 0, "file_name": "a.jpg", "width": 10, "height": 10}],
        anns=[{"id": 0, "image_id": 0, "category_id": 2, "bbox": [1, 1, 2, 2]}],
    )
    # squirrel 在默认 label_map 里 → 不该被 flag；构造一个不在 map 的类目
    ad = RoboflowFeederAdapter(str(tmp_path), label_map={"bird": "bird"})  # 只认 bird
    unmapped = ad.audit_unmapped()
    assert "squirrel" in unmapped  # 未映射 → 上线前据此校正
