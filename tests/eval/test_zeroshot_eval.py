"""零样本评测 harness（检测实验计划 §6）：bird 召回@conf、any-animal、squirrel 盲区、按源分组。"""

from __future__ import annotations

from edge_cam.contracts.schemas.detection_manifest import (
    FEEDER5_CATEGORIES,
    DetBox,
    DetectionManifest,
    DetImageRecord,
)
from edge_cam.eval.zeroshot_eval import (
    ANIMAL_CLASSES,
    COCO_FOLD,
    MD_FOLD,
    Pred,
    bird_recall_curve,
    class_recall,
    preds_from_coco,
    recall_at_conf,
    recall_rate,
)


def _box(cls: str, xywh):
    return DetBox(bbox=list(xywh), category_id=FEEDER5_CATEGORIES[cls])


def _manifest():
    # 两源各两图：ena24 有 bird+squirrel，coco 有 bird。image_id = 记录下标。
    recs = [
        DetImageRecord(
            path="e0.jpg",
            split="test",
            width=100,
            height=100,
            boxes=[_box("bird", [10, 10, 20, 20])],
            source="ena24",
        ),
        DetImageRecord(
            path="e1.jpg",
            split="test",
            width=100,
            height=100,
            boxes=[_box("squirrel", [50, 50, 20, 20])],
            source="ena24",
        ),
        DetImageRecord(
            path="c0.jpg",
            split="test",
            width=100,
            height=100,
            boxes=[_box("bird", [30, 30, 20, 20])],
            source="coco2017",
        ),
    ]
    return DetectionManifest(
        name="t", version="v0", categories=dict(FEEDER5_CATEGORIES), records=recs
    )


def test_preds_from_coco_fold_and_drop():
    """COCO-80 预测 → 折叠对比标签；未在 fold 的类(car)丢弃;squirrel 无 COCO 类=盲区之源。"""
    cats = {16: "bird", 3: "car", 18: "dog", 1: "person"}
    raw = [
        {"image_id": 0, "category_id": 16, "bbox": [1, 1, 5, 5], "score": 0.9},  # bird→bird
        {"image_id": 0, "category_id": 18, "bbox": [2, 2, 5, 5], "score": 0.8},  # dog→other_animal
        {"image_id": 1, "category_id": 3, "bbox": [0, 0, 9, 9], "score": 0.7},  # car→丢
    ]
    preds = preds_from_coco(raw, cats, COCO_FOLD)
    assert {p.label for p in preds} == {"bird", "other_animal"}  # car 丢弃
    assert len(preds) == 2


def test_md_fold_animal_person():
    cats = {0: "animal", 1: "person", 2: "vehicle"}
    raw = [
        {"image_id": 0, "category_id": 0, "bbox": [1, 1, 5, 5], "score": 0.9},  # animal
        {"image_id": 0, "category_id": 2, "bbox": [2, 2, 5, 5], "score": 0.8},  # vehicle→丢
    ]
    preds = preds_from_coco(raw, cats, MD_FOLD)
    assert [p.label for p in preds] == ["animal"]  # vehicle 不在 MD_FOLD → 丢


def test_recall_at_conf_iou_and_threshold():
    gt = [[10, 10, 20, 20]]
    # 高 IoU 高 conf → 命中
    hit = [Pred(0, "bird", [11, 11, 20, 20], 0.9)]
    assert recall_at_conf(gt, hit, conf=0.3) == (1, 1)
    # conf 不够 → 不命中
    assert recall_at_conf(gt, [Pred(0, "bird", [11, 11, 20, 20], 0.1)], conf=0.3) == (0, 1)
    # IoU 不够（框偏太远）→ 不命中
    assert recall_at_conf(gt, [Pred(0, "bird", [80, 80, 10, 10], 0.9)], conf=0.3) == (0, 1)


def test_bird_recall_by_source_coco_labels():
    """COCO 检测器（label=bird）：bird GT 被 bird pred 命中，按源分。"""
    m = _manifest()
    preds = [
        Pred(0, "bird", [11, 11, 20, 20], 0.9),  # ena24 e0 命中
        Pred(2, "bird", [31, 31, 20, 20], 0.9),  # coco c0 命中
        Pred(1, "bird", [51, 51, 20, 20], 0.9),  # e1 是 squirrel GT，不算 bird
    ]
    r = class_recall(m, preds, gt_classes={"bird"}, match_labels={"bird"}, conf=0.3)
    assert recall_rate(r["ena24"]) == 1.0  # e0 的 1 只 bird 命中
    assert recall_rate(r["coco2017"]) == 1.0
    assert r["__all__"] == (2, 2)


def test_squirrel_blind_spot_zeroshot_coco():
    """零样本 COCO 盲区：预测里没有 squirrel 标签 → squirrel 召回=0（COCO 无松鼠类）。"""
    m = _manifest()
    preds = [Pred(0, "bird", [11, 11, 20, 20], 0.9)]  # 只有 bird，无 squirrel
    r = class_recall(m, preds, gt_classes={"squirrel"}, match_labels={"squirrel"}, conf=0.3)
    assert recall_rate(r["__all__"]) == 0.0  # squirrel 全漏（盲区实证）


def test_md_animal_hits_bird_gt():
    """MD 不分种：animal 框命中 bird GT 也算 bird 召回（match_labels={animal}）。"""
    m = _manifest()
    preds = [Pred(0, "animal", [11, 11, 20, 20], 0.9), Pred(1, "animal", [51, 51, 20, 20], 0.9)]
    bird = class_recall(m, preds, gt_classes={"bird"}, match_labels={"animal"}, conf=0.3)
    assert recall_rate(bird["ena24"]) == 1.0  # animal 命中 bird GT
    # any-animal：bird+squirrel 都被 animal 命中
    anyani = class_recall(m, preds, gt_classes=ANIMAL_CLASSES, match_labels={"animal"}, conf=0.3)
    assert anyani["ena24"] == (2, 2)  # e0 bird + e1 squirrel 都命中


def test_bird_recall_curve_over_confs():
    m = _manifest()
    preds = [Pred(0, "bird", [11, 11, 20, 20], 0.25), Pred(2, "bird", [31, 31, 20, 20], 0.9)]
    curve = bird_recall_curve(m, preds, match_labels={"bird"}, confs=(0.1, 0.3))
    # bird_recall_curve 已返回召回率(float)。conf 0.1：两只都过→all 1.0；conf 0.3：只 c0(0.9)过
    assert curve[0.1]["__all__"] == 1.0
    assert curve[0.3]["ena24"] == 0.0  # e0 的 pred 0.25<0.3 → 漏
    assert curve[0.3]["coco2017"] == 1.0
