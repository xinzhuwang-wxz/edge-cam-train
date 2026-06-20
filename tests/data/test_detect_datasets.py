"""具体检测数据集 adapter（ENA24/Caltech-CT/COCO2017）：映射/划分/负样本/路由（合成 json）。"""

from __future__ import annotations

import json

from edge_cam.data.adapters.detect import (
    FEEDER5_CATEGORIES,
    assemble,
    available_adapters,
    build_adapter,
)


def _write(path, coco):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco), encoding="utf-8")


def _ena24(tmp_path):
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 80}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 2, 3, 4]},  # Bird
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [5, 6, 7, 8]},  # E. Gray Squirrel
            {"id": 3, "image_id": 1, "category_id": 16, "bbox": [0, 0, 9, 9]},  # Dog→other
            {"id": 4, "image_id": 1, "category_id": 9, "bbox": [0, 0, 1, 1]},  # Vehicle→drop
        ],
        "categories": [
            {"id": 0, "name": "Bird"},
            {"id": 1, "name": "Eastern Gray Squirrel"},
            {"id": 16, "name": "Dog"},
            {"id": 9, "name": "Vehicle"},
        ],
    }
    _write(tmp_path / "commercial/ena24/ena24_public.json", coco)


def test_ena24_maps_drops_livestock_and_splits_by_image(tmp_path) -> None:
    _ena24(tmp_path)
    ad = build_adapter("ena24", str(tmp_path))
    recs = ad.build_records()
    assert len(recs) == 1
    ids = sorted(b.category_id for b in recs[0].boxes)
    expect = [FEEDER5_CATEGORIES[c] for c in ("bird", "squirrel", "other_animal")]
    assert ids == sorted(expect)  # Vehicle 丢；Dog→other_animal
    assert recs[0].path == "commercial/ena24/a.jpg"
    assert recs[0].source == "ena24" and recs[0].license == "CDLA-Permissive"
    assert ad.spec.split_unit == "image" and ad.spec.commercial_safe and ad.spec.exhaustive


def _eccv(tmp_path, anns, images, cats, fname="train_annotations.json"):
    """写 ECCV18 注解文件到 caltech_ct/eccv18/eccv_18_annotation_files/。"""
    p = tmp_path / "commercial/caltech_ct/eccv18/eccv_18_annotation_files" / fname
    _write(p, {"images": images, "annotations": anns, "categories": cats})


def test_caltech_ct_eccv_location_split_and_empty_negatives(tmp_path) -> None:
    imgs = [
        {"id": "i1", "file_name": "x.jpg", "width": 100, "height": 100, "location": "loc7"},
        {"id": "e1", "file_name": "e.jpg", "width": 100, "height": 100, "location": "loc7"},
    ]
    anns = [
        {"id": "a1", "image_id": "i1", "category_id": 11, "bbox": [1, 1, 2, 2]},  # bird
        {"id": "m2", "image_id": "e1", "category_id": 30},  # empty: 无 bbox → 负样本
    ]
    cats = [{"id": 11, "name": "bird"}, {"id": 30, "name": "empty"}]
    _eccv(tmp_path, anns, imgs, cats)

    ad = build_adapter("caltech_ct", str(tmp_path), negative_quota=None)
    recs = {r.path: r for r in ad.build_records()}
    pos = recs["commercial/caltech_ct/eccv_18_all_images_sm/x.jpg"]
    neg = recs["commercial/caltech_ct/eccv_18_all_images_sm/e.jpg"]  # empty → 负样本
    assert [b.category_id for b in pos.boxes] == [FEEDER5_CATEGORIES["bird"]]
    assert neg.boxes == []
    assert len({r.split for r in recs.values()}) == 1  # 同 location=loc7 → 同 split
    assert ad.spec.split_unit == "location"


def test_caltech_ct_negative_quota_zero_drops_empties(tmp_path) -> None:
    imgs = [
        {"id": "i1", "file_name": "x.jpg", "width": 9, "height": 9, "location": "l"},
        {"id": "e1", "file_name": "e.jpg", "width": 9, "height": 9, "location": "l"},
    ]
    anns = [
        {"id": "a1", "image_id": "i1", "category_id": 11, "bbox": [1, 1, 2, 2]},
        {"id": "m2", "image_id": "e1", "category_id": 30},  # empty, 无 bbox
    ]
    _eccv(tmp_path, anns, imgs, [{"id": 11, "name": "bird"}, {"id": 30, "name": "empty"}])
    recs = build_adapter("caltech_ct", str(tmp_path), negative_quota=0).build_records()
    assert all(r.boxes for r in recs)  # quota=0 → 无负样本


def test_coco2017_is_feasibility_eval_only(tmp_path) -> None:
    coco = {
        "images": [{"id": 1, "file_name": "c.jpg", "width": 50, "height": 50}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 16, "bbox": [1, 1, 2, 2]},  # bird
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]},  # person
            {"id": 3, "image_id": 1, "category_id": 19, "bbox": [1, 1, 2, 2]},  # horse→drop
        ],
        "categories": [
            {"id": 16, "name": "bird"},
            {"id": 1, "name": "person"},
            {"id": 19, "name": "horse"},
        ],
    }
    _write(tmp_path / "feasibility/coco/annotations/instances_val2017.json", coco)
    ad = build_adapter("coco2017", str(tmp_path))
    assert ad.spec.role == "eval_only" and not ad.spec.commercial_safe
    recs = ad.build_records()
    ids = sorted(b.category_id for b in recs[0].boxes)
    assert ids == sorted([FEEDER5_CATEGORIES["bird"], FEEDER5_CATEGORIES["person"]])  # horse 丢
    assert recs[0].path == "feasibility/coco/val2017/c.jpg"


def test_registered_and_assemble_routes_commercial_vs_feasibility(tmp_path) -> None:
    assert {"ena24", "caltech_ct", "coco2017"} <= set(available_adapters())
    _ena24(tmp_path)
    coco = {
        "images": [{"id": 1, "file_name": "c.jpg", "width": 9, "height": 9}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 16, "bbox": [1, 1, 2, 2]}],
        "categories": [{"id": 16, "name": "bird"}],
    }
    _write(tmp_path / "feasibility/coco/annotations/instances_val2017.json", coco)
    adapters = [build_adapter("ena24", str(tmp_path)), build_adapter("coco2017", str(tmp_path))]
    out = assemble(adapters)
    # ena24 可商用 → train/test；coco2017 feasibility → eval_feasibility
    assert all(r.source == "ena24" for r in out["train"].records + out["test"].records)
    assert all(r.source == "coco2017" for r in out["eval_feasibility"].records)
    assert out["eval_feasibility"].records  # 非空
