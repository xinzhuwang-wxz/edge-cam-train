"""MD 当 bird 教师的难度量化：匹配 + 汇总（纯函数）。"""

from __future__ import annotations

from edge_cam.eval.md_difficulty import difficulty_summary, match_bird_to_md


def test_match_picks_best_iou_and_conf() -> None:
    gt = {0: [[0, 0, 10, 10]]}
    md = {0: [([0, 0, 10, 10], 0.9), ([50, 50, 5, 5], 0.3)]}  # 第一框完美重叠
    m = match_bird_to_md(gt, md, iou_thr=0.5)
    assert len(m) == 1
    assert m[0]["matched"] is True and m[0]["conf"] == 0.9 and m[0]["iou"] == 1.0


def test_match_miss_when_no_overlap() -> None:
    gt = {0: [[0, 0, 10, 10]]}
    md = {0: [([90, 90, 5, 5], 0.8)]}  # 不重叠
    m = match_bird_to_md(gt, md, iou_thr=0.5)
    assert m[0]["matched"] is False and m[0]["iou"] == 0.0


def test_match_missing_image_counts_as_miss() -> None:
    """GT 有鸟但 MD 该图零框 → 漏框。"""
    m = match_bird_to_md({0: [[0, 0, 10, 10]]}, {}, iou_thr=0.5)
    assert m[0]["matched"] is False and m[0]["conf"] == 0.0


def test_difficulty_summary_recall_and_buckets() -> None:
    matches = [
        {"iou": 1.0, "conf": 0.9, "matched": True},  # auto
        {"iou": 0.6, "conf": 0.45, "matched": True},  # review
        {"iou": 0.55, "conf": 0.1, "matched": True},  # 命中但低置信 → 会丢
        {"iou": 0.0, "conf": 0.0, "matched": False},  # 漏框 → 丢
    ]
    s = difficulty_summary(matches, conf_hi=0.7, conf_lo=0.2)
    assert s["n_gt_bird"] == 4
    assert s["recall@iou"] == 0.75  # 3/4 命中
    assert s["miss_rate"] == 0.25
    assert s["hit_conf_buckets"] == {"<lo(会丢)": 1, "[lo,hi)(人审)": 1, ">=hi(自动收)": 1}
    # triage 预期：auto 1/4=25%，review 1/4=25%，drop（漏1+命中但低1）=50%
    assert s["triage_expectation"] == {"auto_%": 25.0, "review_%": 25.0, "drop_%": 50.0}


def test_difficulty_summary_empty() -> None:
    assert difficulty_summary([]) == {"n_gt_bird": 0}
