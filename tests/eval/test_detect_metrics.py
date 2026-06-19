"""检测指标结构化 + 汇表（env 无关；不依赖 pycocotools）。"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from edge_cam.eval.detect_metrics import (
    append_detection_row,
    from_coco_stats,
)


def test_from_coco_stats_extracts_map() -> None:
    # COCOeval.stats: [AP@.5:.95, AP@.5, AP@.75, ...]
    stats = [0.42, 0.68, 0.45, 0.1, 0.4, 0.6, 0.3, 0.5, 0.55, 0.2, 0.5, 0.6]
    m = from_coco_stats(stats, bird_recall_50=0.81, per_class_ap={"bird": 0.7, "cat": 0.5})
    assert m.map_50 == 0.68
    assert m.map_5095 == 0.42
    assert m.bird_recall_50 == 0.81
    assert m.per_class_ap["bird"] == 0.7


def test_from_coco_stats_rejects_short() -> None:
    with pytest.raises(ValueError, match="至少 2"):
        from_coco_stats([0.4])


def test_append_detection_row_dedups(tmp_path: Path) -> None:
    m1 = from_coco_stats([0.4, 0.6] + [0] * 10, bird_recall_50=0.8)
    append_detection_row(m1, "nanodet-plus-m@416", tmp_path)
    # 同 label 再写 → 覆盖不重复
    m2 = from_coco_stats([0.45, 0.65] + [0] * 10, bird_recall_50=0.82)
    csv_path = append_detection_row(m2, "nanodet-plus-m@416", tmp_path)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["map_50"] == "0.65"  # 覆盖为新值
    # 不同 label → 追加
    append_detection_row(m1, "picodet-s@320", tmp_path)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert {r["label"] for r in rows} == {"nanodet-plus-m@416", "picodet-s@320"}
