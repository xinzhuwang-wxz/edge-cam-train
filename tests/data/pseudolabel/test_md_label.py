"""MD preds → 伪标注 COCO（保 score，只留 animal→bird；纯函数）。"""

from __future__ import annotations

from edge_cam.data.pseudolabel.md_label import build_pseudolabel_coco


def _images():
    return [
        {"id": 0, "file_name": "inat/1.jpg", "width": 100, "height": 100},
        {"id": 1, "file_name": "inat/2.jpg", "width": 100, "height": 100},
    ]


def test_keeps_animal_drops_person() -> None:
    """iNat 是 Aves：MD animal(1) 留、person(2) 丢（bird 图上的人多为误检）。"""
    preds = [
        {"image_id": 0, "category_id": 1, "bbox": [1, 1, 9, 9], "score": 0.88},
        {"image_id": 0, "category_id": 2, "bbox": [2, 2, 8, 8], "score": 0.4},  # person → 丢
        {"image_id": 1, "category_id": 1, "bbox": [3, 3, 7, 7], "score": 0.25},
    ]
    coco = build_pseudolabel_coco(_images(), preds)
    assert len(coco["annotations"]) == 2
    assert {a["score"] for a in coco["annotations"]} == {0.88, 0.25}
    assert coco["categories"] == [{"id": 1, "name": "animal"}]


def test_annotation_ids_contiguous_and_category_normalized() -> None:
    preds = [{"image_id": 0, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.5}]
    coco = build_pseudolabel_coco(_images(), preds)
    a = coco["annotations"][0]
    assert a["id"] == 1 and a["category_id"] == 1 and a["image_id"] == 0


def test_empty_preds_yields_no_boxes() -> None:
    coco = build_pseudolabel_coco(_images(), [])
    assert coco["annotations"] == [] and len(coco["images"]) == 2
