"""统一 crop 规范：外扩正方形在边界内、resize、最小尺寸门控。"""

from __future__ import annotations

from PIL import Image

from edge_cam.data.crop import crop_with_padding, expand_to_square, passes_min_size


def test_expand_is_square_within_bounds() -> None:
    x1, y1, x2, y2 = expand_to_square((40, 40, 60, 80), padding=0.15, img_w=100, img_h=100)
    assert (x2 - x1) == (y2 - y1)  # 正方形
    assert 0 <= x1 <= x2 <= 100
    assert 0 <= y1 <= y2 <= 100


def test_expand_clipped_at_edge() -> None:
    # 框贴右下角，外扩后仍被裁到图像内
    box = expand_to_square((90, 90, 100, 100), padding=0.5, img_w=100, img_h=100)
    assert box[2] <= 100 and box[3] <= 100


def test_crop_resizes_to_size() -> None:
    img = Image.new("RGB", (200, 200))
    out = crop_with_padding(img, (50, 50, 150, 150), padding=0.1, size=224)
    assert out.size == (224, 224)


def test_crop_no_resize_when_size_zero() -> None:
    img = Image.new("RGB", (200, 200))
    out = crop_with_padding(img, (50, 50, 150, 150), padding=0.0, size=0)
    assert out.size == (100, 100)


def test_min_size_gate() -> None:
    wh = (640, 640)
    assert passes_min_size((0, 0, 200, 200), wh)  # 大框通过
    assert not passes_min_size((0, 0, 20, 20), wh)  # 短边 < 32 → 拒
    assert not passes_min_size((0, 0, 100, 5), wh)  # 面积/短边太小 → 拒
