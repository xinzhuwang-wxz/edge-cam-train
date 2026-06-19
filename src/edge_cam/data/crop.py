"""统一 crop 规范（plan §C.6：训推共用同一函数，缩小 domain gap）。

- padding 各边外扩（默认 15%）→ 取最小外接正方形 → resize。
- 最小尺寸门控：crop 太小/框太小 → 只报 `bird` 不报种（由调用方据返回值决定）。

纯函数（除 PIL resize），训练裁剪与端侧推理裁剪必须调用同一份，避免训推不一致。"""

from __future__ import annotations

from PIL import Image

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2)，像素坐标


def expand_to_square(box: Box, padding: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """外扩 padding → 最小外接正方形 → 裁到图像边界，返回整数像素框。"""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 - bw * padding, x2 + bw * padding
    y1, y2 = y1 - bh * padding, y2 + bh * padding

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1)
    x1, x2 = cx - side / 2, cx + side / 2
    y1, y2 = cy - side / 2, cy + side / 2

    x1 = max(0, min(x1, img_w))
    x2 = max(0, min(x2, img_w))
    y1 = max(0, min(y1, img_h))
    y2 = max(0, min(y2, img_h))
    return (round(x1), round(y1), round(x2), round(y2))


def crop_with_padding(
    image: Image.Image, box: Box, padding: float = 0.15, size: int = 224
) -> Image.Image:
    """按统一规范从整图裁出 crop；size>0 时 resize 到 size×size。"""
    square = expand_to_square(box, padding, image.width, image.height)
    crop = image.crop(square)
    if size:
        crop = crop.resize((size, size))
    return crop


def passes_min_size(
    box: Box,
    img_wh: tuple[int, int],
    min_side: int = 32,
    min_area_frac: float = 0.003,
) -> bool:
    """最小尺寸门控（plan §C.6）：短边 ≥ min_side 且面积占比 ≥ min_area_frac 才报种。"""
    x1, y1, x2, y2 = box
    img_w, img_h = img_wh
    side = min(x2 - x1, y2 - y1)
    area_frac = ((x2 - x1) * (y2 - y1)) / (img_w * img_h) if img_w * img_h else 0.0
    return side >= min_side and area_frac >= min_area_frac
