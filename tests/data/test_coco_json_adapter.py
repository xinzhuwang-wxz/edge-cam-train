"""CocoJsonAdapter：合成 COCO json 测解析/负样本/分组键/未映射审计（不依赖真实数据）。"""

from __future__ import annotations

import json

from edge_cam.data.adapters.detect import FEEDER5_CATEGORIES, CocoJsonAdapter, DatasetSpec


def _write_coco(tmp_path, images, annotations, categories):
    p = tmp_path / "instances.json"
    p.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories}),
        encoding="utf-8",
    )
    return p


def _spec(**kw):
    base = dict(
        name="ena24",
        raw_format="coco_json",
        label_map={"Bird": "bird", "Eastern Gray Squirrel": "squirrel", "Raccoon": "other_animal"},
        license="CDLA-Permissive",
        commercial_safe=True,
        exhaustive=True,
        negative_quota=None,  # 全留负样本
    )
    base.update(kw)
    return DatasetSpec(**base)


def test_parse_maps_categories_and_marks_empty_negative(tmp_path) -> None:
    cats = [
        {"id": 1, "name": "Bird"},
        {"id": 2, "name": "Raccoon"},
        {"id": 3, "name": "Horse"},  # 不在 label_map → 丢
    ]
    images = [
        {"id": 10, "file_name": "a.jpg", "width": 640, "height": 480, "location": "siteA"},
        {"id": 11, "file_name": "b.jpg", "width": 640, "height": 480, "location": "siteB"},  # 空图
    ]
    anns = [
        {"id": 1, "image_id": 10, "category_id": 1, "bbox": [1, 2, 3, 4]},
        {"id": 2, "image_id": 10, "category_id": 3, "bbox": [0, 0, 5, 5]},  # Horse
    ]
    jp = _write_coco(tmp_path, images, anns, cats)
    ad = CocoJsonAdapter(_spec(), jp, group_key_field="location")
    raws = list(ad.load_raw())
    assert len(raws) == 2
    a = next(r for r in raws if r.path == "a.jpg")
    assert {lbl for lbl, _ in a.boxes} == {"Bird", "Horse"}  # 源标签，未映射在 load_raw 不丢
    assert a.group_key == "siteA"
    b = next(r for r in raws if r.path == "b.jpg")
    assert b.is_negative and b.boxes == []

    recs = ad.build_records()
    a_rec = next(r for r in recs if r.path == "a.jpg")
    assert [bx.category_id for bx in a_rec.boxes] == [FEEDER5_CATEGORIES["bird"]]  # Horse 已丢
    b_rec = next(r for r in recs if r.path == "b.jpg")
    assert b_rec.boxes == []  # 空图穷尽源 → 负样本保留


def test_no_bbox_annotation_skipped(tmp_path) -> None:
    # 无 bbox 注解（相机陷阱 empty 帧）→ 不计框；该图 0 框 → 负样本
    cats = [{"id": 1, "name": "Bird"}, {"id": 30, "name": "empty"}]
    images = [{"id": 1, "file_name": "e.jpg", "width": 10, "height": 10}]
    anns = [{"id": 1, "image_id": 1, "category_id": 30}]  # empty, 无 bbox 键
    jp = _write_coco(tmp_path, images, anns, cats)
    raw = next(iter(CocoJsonAdapter(_spec(), jp).load_raw()))
    assert raw.boxes == [] and raw.is_negative


def test_image_root_prefixes_path(tmp_path) -> None:
    cats = [{"id": 1, "name": "Bird"}]
    images = [{"id": 1, "file_name": "x.jpg", "width": 10, "height": 10}]
    anns = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2]}]
    jp = _write_coco(tmp_path, images, anns, cats)
    ad = CocoJsonAdapter(_spec(), jp, image_root="ena24/imgs")
    assert next(iter(ad.load_raw())).path == "ena24/imgs/x.jpg"


def test_audit_unmapped_reports_uncovered_categories(tmp_path) -> None:
    cats = [
        {"id": 1, "name": "Bird"},
        {"id": 2, "name": "Horse"},
        {"id": 3, "name": "Vehicle"},
    ]
    images = [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}]
    anns = [
        {"id": 1, "image_id": 1, "category_id": 2, "bbox": [0, 0, 1, 1]},
        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [0, 0, 1, 1]},
        {"id": 3, "image_id": 1, "category_id": 3, "bbox": [0, 0, 1, 1]},
    ]
    jp = _write_coco(tmp_path, images, anns, cats)
    ad = CocoJsonAdapter(_spec(), jp)
    assert ad.audit_unmapped() == {"Horse": 2, "Vehicle": 1}  # Bird 已映射 → 不报
