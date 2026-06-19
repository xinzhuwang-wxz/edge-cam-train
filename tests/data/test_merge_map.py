"""跨集合并：统一映射、去重、OIV7 非穷尽 ignore 策略（纯函数，不依赖 fiftyone）。"""

from __future__ import annotations

from edge_cam.data.merge_map import (
    MergedSample,
    UnifiedDetection,
    apply_ignore_policy,
    coco_to_unified,
    dedup_by_image_id,
    is_exhaustive,
    oiv7_to_unified,
    unified_label,
)


def test_unified_mapping_coco_oiv7() -> None:
    assert coco_to_unified()["bird"] == "bird"
    assert oiv7_to_unified()["Squirrel"] == "squirrel"
    assert unified_label("bird", "coco") == "bird"
    assert unified_label("Squirrel", "oiv7") == "squirrel"
    # 不在映射表内 → None（应丢弃）
    assert unified_label("toothbrush", "coco") is None


def test_exhaustiveness_flags() -> None:
    assert is_exhaustive("coco") is True
    assert is_exhaustive("oiv7") is False
    assert is_exhaustive("unknown") is False  # 未知源保守


def test_dedup_by_image_id_preserves_order() -> None:
    s = [
        MergedSample("img1", "coco"),
        MergedSample("img2", "coco"),
        MergedSample("img1", "coco"),  # 重复
        MergedSample("img1", "oiv7"),  # 不同源，保留
    ]
    out = dedup_by_image_id(s)
    assert [(x.image_id, x.source) for x in out] == [
        ("img1", "coco"),
        ("img2", "coco"),
        ("img1", "oiv7"),
    ]


def test_ignore_policy_marks_oiv7_non_exhaustive() -> None:
    coco = apply_ignore_policy(
        MergedSample("a", "coco", [UnifiedDetection("bird", (0, 0, 1, 1), "coco")])
    )
    oiv7 = apply_ignore_policy(MergedSample("b", "oiv7"))
    assert coco.exhaustive is True  # COCO 可挖背景负样本
    assert oiv7.exhaustive is False  # OIV7 未标区域不可信 → 不挖负样本
