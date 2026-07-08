"""crop-to-feeder-framing 纯几何测试（round3 §A6，无卡）。"""

from __future__ import annotations

from edge_cam.contracts.schemas.detection_manifest import DetBox, DetImageRecord
from edge_cam.data.feeder_crop import (
    compute_feeder_crop,
    plan_feeder_crops,
    remap_boxes_to_crop,
)


def test_basic_centered_crop() -> None:
    # 1000² 图，中心 100px 框，target 0.5 → 裁窗边 200、居中、框占 0.5
    crop = compute_feeder_crop(1000, 1000, [400, 400, 100, 100], target_box_frac=0.5)
    assert crop == (350, 350, 200, 200)
    assert 100 / crop[2] == 0.5  # box 长边占裁窗一半（裁窗边 = crop[2]）


def test_min_box_px_guard_returns_none() -> None:
    # 30px 框 < min_box_px(48) → 不裁（放大会糊）
    assert compute_feeder_crop(1000, 1000, [0, 0, 30, 30], min_box_px=48.0) is None


def test_clamp_at_corner() -> None:
    # 角落框，裁窗越界 → 夹回图内
    crop = compute_feeder_crop(1000, 1000, [0, 0, 100, 100], target_box_frac=0.5)
    assert crop == (0, 0, 200, 200)


def test_side_clamped_to_image() -> None:
    # 大框 + 小图：期望裁窗超过图 → 夹到 min(img) = 整图
    crop = compute_feeder_crop(300, 300, [50, 50, 200, 200], target_box_frac=0.5)
    assert crop == (0, 0, 300, 300)


def test_remap_box_fully_inside() -> None:
    crop = (350, 350, 200, 200)
    boxes = [DetBox(bbox=[400, 400, 100, 100], category_id=1)]
    out = remap_boxes_to_crop(boxes, crop)
    assert len(out) == 1
    assert out[0].bbox == [50.0, 50.0, 100.0, 100.0]
    assert out[0].category_id == 1


def test_remap_partial_kept_and_clipped() -> None:
    # 框跨裁窗左边界，可见 0.5 ≥ 0.4 → 保留且裁剪
    crop = (350, 350, 200, 200)
    boxes = [DetBox(bbox=[300, 400, 100, 100], category_id=1)]
    out = remap_boxes_to_crop(boxes, crop, min_visible=0.4)
    assert len(out) == 1
    assert out[0].bbox == [0.0, 50.0, 50.0, 100.0]


def test_remap_mostly_outside_dropped() -> None:
    # 框仅一角(10%)在裁窗内 → 低于 min_visible → 丢
    crop = (350, 350, 200, 200)
    boxes = [DetBox(bbox=[540, 540, 100, 100], category_id=1)]  # 只有 10×10 在窗内
    out = remap_boxes_to_crop(boxes, crop, min_visible=0.4)
    assert out == []


def test_remap_keeps_other_visible_animal() -> None:
    # 同图第二只动物也在裁窗里 → 一起带过（喂食器共现场景）
    crop = (350, 350, 200, 200)
    boxes = [
        DetBox(bbox=[400, 400, 100, 100], category_id=1),  # squirrel 主体
        DetBox(bbox=[360, 360, 60, 60], category_id=0),  # 旁边一只 bird
    ]
    out = remap_boxes_to_crop(boxes, crop)
    assert len(out) == 2
    assert {b.category_id for b in out} == {0, 1}


def _rec(boxes: list[DetBox], w: int = 1000, h: int = 1000) -> DetImageRecord:
    return DetImageRecord(path="x.jpg", split="train", width=w, height=h, boxes=boxes)


def test_plan_targets_squirrel_only() -> None:
    # squirrel(id1) 100px 目标框 → 规划 1 个中大近景裁
    r = _rec([DetBox(bbox=[400, 400, 100, 100], category_id=1)])
    plans = plan_feeder_crops([r], {1})
    assert len(plans) == 1
    _src, _crop, nb = plans[0]
    assert any(b.category_id == 1 for b in nb)  # 裁窗里松鼠留住了


def test_plan_skips_non_target_class() -> None:
    # 只有 bird(id0) → 目标是 squirrel/cat → 0 计划
    r = _rec([DetBox(bbox=[400, 400, 100, 100], category_id=0)])
    assert plan_feeder_crops([r], {1, 2}) == []


def test_plan_skips_too_small_box() -> None:
    # 30px 松鼠 < min_box_px(48) → 不裁（放大会糊）
    r = _rec([DetBox(bbox=[400, 400, 30, 30], category_id=1)])
    assert plan_feeder_crops([r], {1}) == []


def test_plan_caps_per_image() -> None:
    boxes = [
        DetBox(bbox=[100, 100, 100, 100], category_id=1),
        DetBox(bbox=[500, 500, 100, 100], category_id=1),
        DetBox(bbox=[800, 800, 100, 100], category_id=1),
    ]
    plans = plan_feeder_crops([_rec(boxes)], {1}, max_crops_per_image=2)
    assert len(plans) == 2
