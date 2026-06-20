"""检测评测指标结构化 + 汇入实验总表（plan §B.1/§B.2；架构审查 #2）。

检测训练/评测跑在 NanoDet 独立 env（pycocotools COCOeval）。本模块把它的结果**结构化**
（mAP@.5、mAP@.5:.95、bird-recall@IoU0.5、per-class AP）并写成与分类消融同形态的总表行，
让 B.2 检测选型和 B.3 分类一样可对比（plan §B.6 一页纸）。

两种入口：
- from_coco_stats：已有 pycocotools COCOeval.stats（12 浮点）+ per-class → 结构化（env 无关，可测）
- evaluate_coco：直接给 GT + 预测 COCO-json，惰性 import pycocotools 跑 COCOeval（在 nanodet env）
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

# pycocotools COCOeval.stats 顺序（固定）
_AP_5095, _AP_50 = 0, 1  # stats[0]=AP@.5:.95, stats[1]=AP@.5


@dataclass
class DetectionMetrics:
    """一个检测器配置的评测结果（B.2 一行）。"""

    map_50: float
    map_5095: float
    bird_recall_50: float | None = None  # bird 类 recall@IoU0.5（最关键，plan §B.1）
    per_class_ap: dict[str, float] = field(default_factory=dict)


def from_coco_stats(
    stats: list[float],
    *,
    bird_recall_50: float | None = None,
    per_class_ap: dict[str, float] | None = None,
) -> DetectionMetrics:
    """pycocotools COCOeval.stats（12 浮点）→ DetectionMetrics（env 无关）。"""
    if len(stats) < 2:
        raise ValueError(f"COCOeval.stats 至少 2 个值，得到 {len(stats)}")
    return DetectionMetrics(
        map_50=round(stats[_AP_50], 4),
        map_5095=round(stats[_AP_5095], 4),
        bird_recall_50=bird_recall_50,
        per_class_ap=per_class_ap or {},
    )


def evaluate_coco(gt_json: str | Path, pred_json: str | Path) -> DetectionMetrics:
    """GT + 预测 COCO-json → COCOeval（惰性 import pycocotools，在 nanodet env 跑）。

    bird-recall@.5 取 bird 类(category_id 由 GT 的 categories 定)的 recall。
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(str(gt_json))
    dt = gt.loadRes(str(pred_json))
    ev = COCOeval(gt, dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    name_of = {c["id"]: c["name"] for c in gt.dataset["categories"]}
    # per-class AP@.5:.95（precision 维度求均值）
    per_class: dict[str, float] = {}
    for i, cid in enumerate(gt.getCatIds()):
        prec = ev.eval["precision"][:, :, i, 0, 2]
        valid = prec[prec > -1]
        per_class[name_of[cid]] = round(float(valid.mean()) if valid.size else 0.0, 4)
    return from_coco_stats(list(ev.stats), per_class_ap=per_class)


_DET_FIELDS = ["label", "map_50", "map_5095", "bird_recall_50"]


def append_detection_row(metrics: DetectionMetrics, label: str, out_dir: str | Path) -> Path:
    """把一行检测结果写入 detect_ablation.csv（B.2 表，与分类 ablation 分开）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "detect_ablation.csv"
    rows: list[dict] = []
    if csv_path.exists():
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    rows = [r for r in rows if r.get("label") != label]  # 同 label 覆盖
    rows.append(
        {
            "label": label,
            "map_50": metrics.map_50,
            "map_5095": metrics.map_5095,
            "bird_recall_50": metrics.bird_recall_50 if metrics.bird_recall_50 is not None else "",
        }
    )
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_DET_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return csv_path
