"""检测打标契约：闭集校验、bbox 合法性、COCO 互转、LLM 幻觉拒绝。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from edge_cam.contracts.schemas.detection import (
    BBox,
    DetImageLabels,
    from_coco,
    to_coco,
    validate_llm_labels,
)


def test_valid_labels_pass() -> None:
    img = DetImageLabels(
        file_name="x.jpg",
        width=640,
        height=480,
        annotations=[{"label": "bird", "bbox": {"x": 10, "y": 10, "w": 100, "h": 80}}],
    )
    assert img.annotations[0].label == "bird"


def test_hallucinated_label_rejected() -> None:
    """LLM 自创类名 → 闭集校验拒。"""
    with pytest.raises(ValidationError):
        DetImageLabels(
            file_name="x.jpg",
            width=640,
            height=480,
            annotations=[{"label": "small_bird", "bbox": {"x": 0, "y": 0, "w": 10, "h": 10}}],
        )


def test_bbox_out_of_image_rejected() -> None:
    with pytest.raises(ValidationError, match="超出图像"):
        DetImageLabels(
            file_name="x.jpg",
            width=100,
            height=100,
            annotations=[{"label": "cat", "bbox": {"x": 50, "y": 50, "w": 80, "h": 80}}],
        )


def test_bbox_nonpositive_rejected() -> None:
    with pytest.raises(ValidationError):
        BBox(x=0, y=0, w=0, h=10)  # w 必须 >0


def test_validate_llm_labels_batch() -> None:
    raw = [
        {
            "file_name": "a.jpg",
            "width": 64,
            "height": 64,
            # dog 属长尾哺乳动物 → 5 类映射为 other_animal（[[ADR-0004]]）
            "annotations": [{"label": "other_animal", "bbox": {"x": 1, "y": 1, "w": 10, "h": 10}}],
        },
        {"file_name": "b.jpg", "width": 64, "height": 64, "annotations": []},
    ]
    out = validate_llm_labels(raw)
    assert len(out) == 2


def test_coco_roundtrip() -> None:
    labels = [
        DetImageLabels(
            file_name="a.jpg",
            width=640,
            height=480,
            annotations=[
                {"label": "bird", "bbox": {"x": 10, "y": 20, "w": 100, "h": 50}},
                {"label": "squirrel", "bbox": {"x": 5, "y": 5, "w": 30, "h": 30}},
            ],
        )
    ]
    coco = to_coco(labels)
    assert {c["name"] for c in coco["categories"]} >= {"bird", "squirrel"}
    assert coco["annotations"][0]["bbox"] == [10, 20, 100, 50]
    back = from_coco(coco)
    assert back[0].file_name == "a.jpg"
    assert [a.label for a in back[0].annotations] == ["bird", "squirrel"]
