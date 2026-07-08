"""检测 margin 纯函数测试（round3 §6，无卡）。"""

from __future__ import annotations

from edge_cam.eval.detect_margin import margin_stats, top1_top2

NAMES = ["bird", "squirrel", "cat", "person", "other_animal"]


def test_top1_top2_separated() -> None:
    # 清晰: bird 0.9 远超次高 0.1 → margin 0.8
    idx, m = top1_top2([0.9, 0.1, 0.05, 0.0, 0.0])
    assert idx == 0
    assert abs(m - 0.8) < 1e-9


def test_top1_top2_ambiguous() -> None:
    # "都差不多": 0.55 vs 0.45 → margin 0.1（squirrel 险胜 bird）
    idx, m = top1_top2([0.45, 0.55, 0.0, 0.0, 0.0])
    assert idx == 1
    assert abs(m - 0.1) < 1e-9


def test_top1_top2_single() -> None:
    assert top1_top2([0.7]) == (0, 0.7)


def test_margin_stats_median_and_frac() -> None:
    dets = [
        {"class_scores": [0.9, 0.1, 0, 0, 0]},  # margin 0.8
        {"class_scores": [0.45, 0.55, 0, 0, 0]},  # margin 0.1（top1 squirrel）
        {"class_scores": [0.8, 0.4, 0, 0, 0]},  # margin 0.4
    ]
    st = margin_stats(dets, NAMES, min_top1=0.4)
    ms = sorted(st.margins)
    assert len(ms) == 3
    assert all(abs(a - b) < 1e-9 for a, b in zip(ms, [0.1, 0.4, 0.8], strict=True))
    assert abs(st.median() - 0.4) < 1e-9
    assert abs(st.frac_below(0.2) - 1 / 3) < 1e-9  # 只有 0.1 < 0.2


def test_margin_stats_min_top1_filter() -> None:
    # top1 分 0.3 < min_top1 0.4 → 不计入（没触发）
    dets = [{"class_scores": [0.3, 0.05, 0, 0, 0]}, {"class_scores": [0.9, 0.1, 0, 0, 0]}]
    st = margin_stats(dets, NAMES, min_top1=0.4)
    assert st.margins == [0.8]


def test_per_class_median() -> None:
    dets = [
        {"class_scores": [0.9, 0.1, 0, 0, 0]},  # bird, margin 0.8
        {"class_scores": [0.5, 0.9, 0, 0, 0]},  # squirrel, margin 0.4
        {"class_scores": [0.6, 0.95, 0, 0, 0]},  # squirrel, margin 0.35
    ]
    pcm = margin_stats(dets, NAMES).per_class_median()
    assert abs(pcm["bird"] - 0.8) < 1e-9
    assert abs(pcm["squirrel"] - 0.375) < 1e-9  # (0.35+0.4)/2
