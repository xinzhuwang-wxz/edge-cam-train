"""置信分层：高→自动收 / 中→人审 / 低→丢（纯函数）。"""

from __future__ import annotations

from edge_cam.data.pseudolabel.triage import triage_by_confidence


def _coco():
    return {
        "images": [
            {"id": 10, "file_name": "a.jpg", "width": 100, "height": 100},  # 有 0.9 → auto
            {"id": 20, "file_name": "b.jpg", "width": 100, "height": 100},  # max 0.45 → review
            {"id": 30, "file_name": "c.jpg", "width": 100, "height": 100},  # max 0.1 → drop
            {"id": 40, "file_name": "d.jpg", "width": 100, "height": 100},  # 无框 → drop
        ],
        "annotations": [
            {"id": 1, "image_id": 10, "category_id": 1, "bbox": [0, 0, 9, 9], "score": 0.9},
            {"id": 2, "image_id": 10, "category_id": 1, "bbox": [0, 0, 5, 5], "score": 0.3},
            {"id": 3, "image_id": 20, "category_id": 1, "bbox": [0, 0, 9, 9], "score": 0.45},
            {"id": 4, "image_id": 20, "category_id": 1, "bbox": [0, 0, 5, 5], "score": 0.1},
            {"id": 5, "image_id": 30, "category_id": 1, "bbox": [0, 0, 9, 9], "score": 0.1},
        ],
        "categories": [{"id": 1, "name": "animal"}],
    }


def test_three_way_routing_counts() -> None:
    r = triage_by_confidence(_coco(), conf_hi=0.7, conf_lo=0.2)
    assert r.stats["auto_images"] == 1
    assert r.stats["review_images"] == 1
    assert r.stats["dropped_images"] == 2  # c(低) + d(无框)


def test_auto_keeps_only_high_boxes() -> None:
    """auto 图只留 ≥hi 的框（0.3 的中置信框不掺进自动收）。"""
    r = triage_by_confidence(_coco(), conf_hi=0.7, conf_lo=0.2)
    assert len(r.auto["annotations"]) == 1
    assert r.auto["annotations"][0]["score"] == 0.9


def test_review_keeps_boxes_above_lo() -> None:
    """review 图预标注留 ≥lo 的框（0.45 留、0.1 丢）。"""
    r = triage_by_confidence(_coco(), conf_hi=0.7, conf_lo=0.2)
    assert len(r.review["annotations"]) == 1
    assert r.review["annotations"][0]["score"] == 0.45


def test_dropped_ids() -> None:
    r = triage_by_confidence(_coco(), conf_hi=0.7, conf_lo=0.2)
    assert set(r.dropped_image_ids) == {30, 40}


def test_rebuilt_ids_are_contiguous() -> None:
    """输出 COCO 重排：image_id/ann_id 连续且自洽。"""
    r = triage_by_confidence(_coco(), conf_hi=0.7, conf_lo=0.2)
    assert [im["id"] for im in r.auto["images"]] == [1]
    a = r.auto["annotations"][0]
    assert a["id"] == 1 and a["image_id"] == 1  # ann 指向重排后的 image id
