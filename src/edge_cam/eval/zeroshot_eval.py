"""零样本检测评测 harness（检测实验计划 §5-6）：任意检测器的折叠预测 + 带框 GT → 召回/盲区指标。

把 `megadetector.py` 的"折叠 + IoU 匹配"泛化成**任意检测器**（MD/NanoDet/RTMDet 同一把尺子）。
- 预测已折叠到**对比标签**（bird/squirrel/cat/person/other_animal，或 MD 的 animal）；
  `image_id` = manifest 记录下标（与 GT 对齐）。
- 本模块只做**召回**（纯 numpy，可测）：bird 召回@conf、any-animal、squirrel/other 盲区，按源分组。
- AP50 另走 `detect_metrics.evaluate_coco`（pycocotools，box 上跑）—— 不在此。

用法（实验计划 §6，`m`=manifest）：
    bird 召回（COCO 检测器）: gt_classes={"bird"}, match_labels={"bird"}
    bird 召回（MD 不分种）  : gt_classes={"bird"}, match_labels={"animal"}
    any-animal（跨模型）    : gt_classes=ANIMAL_CLASSES, match_labels=ANIMAL_CLASSES|{"animal"}
    squirrel 盲区（COCO）   : gt_classes={"squirrel"}, match_labels={"squirrel"}
"""

from __future__ import annotations

from dataclasses import dataclass

from edge_cam.contracts.schemas.detection_manifest import FEEDER5_CATEGORIES, DetectionManifest
from edge_cam.eval.megadetector import iou_xywh

# 折叠口径：非人动物 4 类（any-animal 用）；person 单列。
ANIMAL_CLASSES: frozenset[str] = frozenset({"bird", "squirrel", "cat", "other_animal"})

# COCO-80 → 对比标签（实验计划 §5）。COCO **无 squirrel** → 零样本 squirrel 盲区。
COCO_FOLD: dict[str, str] = {
    "bird": "bird",
    "cat": "cat",
    "person": "person",
    "dog": "other_animal",
}
# MegaDetector → 对比标签（不分种，animal 通吃）。
MD_FOLD: dict[str, str] = {"animal": "animal", "person": "person"}


@dataclass
class Pred:
    """一个折叠后的检测预测：image_id（=manifest 记录下标）+ 对比标签 + COCO bbox[xywh] + 置信。"""

    image_id: int
    label: str
    bbox: list[float]
    score: float


def preds_from_coco(
    pred_records: list[dict], cat_names: dict[int, str], fold: dict[str, str]
) -> list[Pred]:
    """模型 COCO 预测 [{image_id,category_id,bbox,score}] + {cat_id:name} + fold(name→对比标签)
    → [Pred]。**未在 fold 的类丢弃**（COCO 的 car / MD 的 vehicle）。bbox=COCO xywh。"""
    out: list[Pred] = []
    for p in pred_records:
        name = cat_names.get(p["category_id"])
        label = fold.get(name) if name is not None else None
        if label is None:
            continue
        out.append(
            Pred(
                image_id=int(p["image_id"]),
                label=label,
                bbox=[float(v) for v in p["bbox"]],
                score=float(p["score"]),
            )
        )
    return out


def recall_at_conf(
    gt_boxes: list[list[float]], preds: list[Pred], *, conf: float, iou_thr: float = 0.5
) -> tuple[int, int]:
    """(matched, total)：GT 框被 score≥conf 的 pred 命中（IoU≥thr）。类无关，caller 先按类过滤。"""
    strong = [p for p in preds if p.score >= conf]
    matched = sum(1 for g in gt_boxes if any(iou_xywh(g, p.bbox) >= iou_thr for p in strong))
    return matched, len(gt_boxes)


def class_recall(
    manifest: DetectionManifest,
    preds: list[Pred],
    *,
    gt_classes: set[str] | frozenset[str],
    match_labels: set[str] | frozenset[str],
    conf: float = 0.3,
    iou_thr: float = 0.5,
) -> dict[str, tuple[int, int]]:
    """按源+总体：gt_classes 的 GT 框被（label∈match_labels 且 score≥conf）pred 命中@IoU 的召回。

    返回 `{source: (matched, total), "__all__": (matched, total)}`。同一 GT 框只要被任一合规 pred
    命中即计（类内匹配，不要求 pred 类=GT 类——支持 MD 的 animal 命中 bird GT、any-animal 折叠）。
    """
    gt_ids = {FEEDER5_CATEGORIES[c] for c in gt_classes}
    preds_by_img: dict[int, list[Pred]] = {}
    for p in preds:
        if p.label in match_labels:
            preds_by_img.setdefault(p.image_id, []).append(p)
    agg: dict[str, list[int]] = {}
    for img_id, r in enumerate(manifest.records):
        gt = [list(b.bbox) for b in r.boxes if b.category_id in gt_ids]
        if not gt:
            continue
        m, t = recall_at_conf(gt, preds_by_img.get(img_id, []), conf=conf, iou_thr=iou_thr)
        for key in (r.source or "unknown", "__all__"):
            slot = agg.setdefault(key, [0, 0])
            slot[0] += m
            slot[1] += t
    return {k: (v[0], v[1]) for k, v in agg.items()}


def recall_rate(counts: tuple[int, int]) -> float:
    """(matched, total) → 比率（total=0 → 0.0）。召回=命中/GT；查准=TP/预测框（同式）。"""
    matched, total = counts
    return matched / total if total else 0.0


def _greedy_match(
    gt_boxes: list[list[float]], preds: list[Pred], *, conf: float, iou_thr: float
) -> tuple[int, int]:
    """(tp, n_pred)：score≥conf 的 pred 按分降序**贪心 1-to-1** 配 gt_boxes（每 GT 至多配一次）。

    TP=配上的 pred 数；n_pred=参与 pred 总数（含空图无 GT 可配的误框，全记 FP）。"""
    strong = sorted((p for p in preds if p.score >= conf), key=lambda p: p.score, reverse=True)
    used = [False] * len(gt_boxes)
    tp = 0
    for p in strong:
        best, best_iou = -1, iou_thr
        for gi, g in enumerate(gt_boxes):
            if used[gi]:
                continue
            i = iou_xywh(g, p.bbox)
            if i >= best_iou:
                best, best_iou = gi, i
        if best >= 0:
            used[best] = True
            tp += 1
    return tp, len(strong)


def class_precision(
    manifest: DetectionManifest,
    preds: list[Pred],
    *,
    gt_classes: set[str] | frozenset[str],
    match_labels: set[str] | frozenset[str],
    conf: float = 0.3,
    iou_thr: float = 0.5,
) -> dict[str, tuple[int, int]]:
    """按源+总体查准：（label∈match_labels 且 score≥conf）框里，贪心配到 gt_classes GT 的占比。

    返回 `{source: (tp, n_pred), ...}`——`recall_rate` 一除即查准率。
    **空背景图（该源无此类 GT）上的框全算 FP** → 自动惩罚"不该框的误框"，无需另算负样本误报。
    """
    gt_ids = {FEEDER5_CATEGORIES[c] for c in gt_classes}
    preds_by_img: dict[int, list[Pred]] = {}
    for p in preds:
        if p.label in match_labels:
            preds_by_img.setdefault(p.image_id, []).append(p)
    agg: dict[str, list[int]] = {}
    for img_id, r in enumerate(manifest.records):
        img_preds = preds_by_img.get(img_id, [])
        if not img_preds:
            continue  # 该图没画框 → 不影响查准（查准只问"画出的框对不对"）
        gt = [list(b.bbox) for b in r.boxes if b.category_id in gt_ids]
        tp, npred = _greedy_match(gt, img_preds, conf=conf, iou_thr=iou_thr)
        for key in (r.source or "unknown", "__all__"):
            slot = agg.setdefault(key, [0, 0])
            slot[0] += tp
            slot[1] += npred
    return {k: (v[0], v[1]) for k, v in agg.items()}


def bird_recall_curve(
    manifest: DetectionManifest,
    preds: list[Pred],
    *,
    match_labels: set[str] | frozenset[str],
    confs: tuple[float, ...] = (0.10, 0.20, 0.30),
    iou_thr: float = 0.5,
) -> dict[float, dict[str, float]]:
    """bird 召回随 conf 变化（实验计划 §6.1，bird 为先）。返回 {conf: {source: recall}}。

    match_labels：COCO 检测器传 {"bird"}；MD 传 {"animal"}（不分种，任意 animal 框命中 bird GT）。
    """
    out: dict[float, dict[str, float]] = {}
    for c in confs:
        counts = class_recall(
            manifest, preds, gt_classes={"bird"}, match_labels=match_labels, conf=c, iou_thr=iou_thr
        )
        out[c] = {src: recall_rate(v) for src, v in counts.items()}
    return out
