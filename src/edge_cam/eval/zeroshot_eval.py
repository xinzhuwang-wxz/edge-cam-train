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


@dataclass
class Pred:
    """一个折叠后的检测预测：image_id（=manifest 记录下标）+ 对比标签 + COCO bbox[xywh] + 置信。"""

    image_id: int
    label: str
    bbox: list[float]
    score: float


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
    """(matched, total) → 召回率（total=0 → 0.0）。"""
    matched, total = counts
    return matched / total if total else 0.0


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
