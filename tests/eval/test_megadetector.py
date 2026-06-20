"""MegaDetector 评估纯函数：GT 折叠 / IoU / 类无关召回（不调 pytorch-wildlife）。"""

from __future__ import annotations

from edge_cam.contracts.schemas.detection_manifest import DetBox, DetectionManifest, DetImageRecord
from edge_cam.data.adapters.detect import FEEDER5_CATEGORIES
from edge_cam.eval.megadetector import (
    build_gt_coco_collapsed,
    class_recall_by_any,
    iou_xywh,
)


def _manifest():
    recs = [
        DetImageRecord(
            path="a.jpg",
            split="test",
            width=100,
            height=100,
            boxes=[
                DetBox(bbox=[10, 10, 20, 20], category_id=FEEDER5_CATEGORIES["bird"]),
                DetBox(bbox=[50, 50, 10, 10], category_id=FEEDER5_CATEGORIES["person"]),
            ],
        ),
        DetImageRecord(
            path="b.jpg",
            split="test",
            width=100,
            height=100,
            boxes=[DetBox(bbox=[0, 0, 5, 5], category_id=FEEDER5_CATEGORIES["squirrel"])],
        ),
    ]
    return DetectionManifest(
        name="t", version="v0", categories=dict(FEEDER5_CATEGORIES), records=recs
    )


def test_build_gt_coco_collapsed_maps_to_animal_person() -> None:
    coco, records = build_gt_coco_collapsed(_manifest(), "test")
    assert len(records) == 2 and len(coco["images"]) == 2
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    assert cats == {1: "animal", 2: "person"}
    by_cat = sorted(a["category_id"] for a in coco["annotations"])
    assert by_cat == [1, 1, 2]  # bird→1, squirrel→1, person→2


def test_iou_xywh() -> None:
    assert iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou_xywh([0, 0, 10, 10], [20, 20, 5, 5]) == 0.0
    assert abs(iou_xywh([0, 0, 10, 10], [5, 0, 10, 10]) - (50 / 150)) < 1e-6


def test_class_recall_by_any() -> None:
    gt = {0: [[10, 10, 20, 20]], 1: [[0, 0, 10, 10]]}  # 两张图各 1 个 bird GT
    preds = {0: [[11, 11, 19, 19]]}  # 图0 有命中框；图1 无 → 召回 1/2
    assert class_recall_by_any(gt, preds, 0.5) == 0.5
    assert class_recall_by_any(gt, {}, 0.5) == 0.0  # 无 pred → 0
