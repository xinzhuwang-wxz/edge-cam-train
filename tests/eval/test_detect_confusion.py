"""检测混淆矩阵纯函数测试（合成数据，无卡）。round3 §6 回归护栏地基。"""

from __future__ import annotations

from edge_cam.eval.detect_confusion import ConfusionMatrix, _iou_xywh, build_confusion

CATS = [{"id": 1, "name": "bird"}, {"id": 2, "name": "squirrel"}, {"id": 3, "name": "cat"}]
BOX = [10.0, 10.0, 20.0, 20.0]  # x, y, w, h
FAR = [200.0, 200.0, 20.0, 20.0]


def _gt(anns: list[tuple[int, int, list[float]]], n_img: int = 1) -> dict:
    return {
        "images": [{"id": i} for i in range(n_img)],
        "annotations": [{"image_id": im, "category_id": c, "bbox": b} for im, c, b in anns],
        "categories": CATS,
    }


def _pred(img: int, cid: int, box: list[float], score: float) -> dict:
    return {"image_id": img, "category_id": cid, "bbox": box, "score": score}


def test_iou_basic() -> None:
    assert _iou_xywh(BOX, BOX) == 1.0
    assert _iou_xywh(BOX, FAR) == 0.0


def test_perfect_match() -> None:
    cm = build_confusion(_gt([(0, 1, BOX)]), [_pred(0, 1, BOX, 0.9)])
    assert cm.matrix[0][0] == 1
    assert cm.diagonal_rate() == 1.0
    assert cm.class_confusion_rate() == 0.0
    assert cm.per_class_recall()["bird"] == 1.0


def test_squirrel_predicted_as_bird() -> None:
    # 真值松鼠框，被判成 bird（同位置）→ 落非对角格 [squirrel][bird]
    cm = build_confusion(_gt([(0, 2, BOX)]), [_pred(0, 1, BOX, 0.9)])
    assert cm.matrix[1][0] == 1  # true squirrel(idx1) → pred bird(idx0)
    assert cm.diagonal_rate() == 0.0
    assert cm.class_confusion_rate() == 1.0
    assert ("squirrel", "bird", 1) in cm.confused_pairs()


def test_missed_detection() -> None:
    cm = build_confusion(_gt([(0, 1, BOX)]), [])
    assert cm.matrix[0][cm.bg] == 1
    assert cm.per_class_recall()["bird"] == 0.0


def test_false_positive() -> None:
    cm = build_confusion(_gt([], n_img=1), [_pred(0, 3, BOX, 0.9)])
    assert cm.matrix[cm.bg][2] == 1


def test_conf_threshold_filters() -> None:
    # 预测分数低于阈值 → 被滤 → GT 变漏检
    cm = build_confusion(_gt([(0, 1, BOX)]), [_pred(0, 1, BOX, 0.2)], conf_thr=0.4)
    assert cm.matrix[0][0] == 0
    assert cm.matrix[0][cm.bg] == 1


def test_iou_threshold_no_overlap() -> None:
    # 预测框不重叠 → GT 漏检 + 预测误报
    cm = build_confusion(_gt([(0, 1, BOX)]), [_pred(0, 1, FAR, 0.9)])
    assert cm.matrix[0][cm.bg] == 1
    assert cm.matrix[cm.bg][0] == 1


def test_greedy_one_to_one() -> None:
    # 2 GT + 2 预测同图：贪心一对一，勿重复占用
    g = _gt([(0, 1, BOX), (0, 2, FAR)])
    p = [_pred(0, 1, BOX, 0.9), _pred(0, 2, FAR, 0.8)]
    cm = build_confusion(g, p)
    assert cm.matrix[0][0] == 1  # bird→bird
    assert cm.matrix[1][1] == 1  # squirrel→squirrel
    assert cm.diagonal_rate() == 1.0


def test_to_markdown_shape() -> None:
    cm = build_confusion(_gt([(0, 1, BOX)]), [_pred(0, 1, BOX, 0.9)])
    md = cm.to_markdown()
    assert "true\\pred" in md
    assert "bg" in md
    assert isinstance(cm, ConfusionMatrix)
