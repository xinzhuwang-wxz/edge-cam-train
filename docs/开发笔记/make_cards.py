#!/usr/bin/env python
"""给分类数据集 3 张代表图加顶部信息条,做成"数据集卡片"。英文 header 避字体坑。"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

A = Path(__file__).resolve().parent / "assets"
RAW = A / "raw_cls"

CARDS = {
    "naturgucker": [
        "naturgucker (DE) | CC-BY | ~300 sp / 59,023 img",
        "sample: Ardea cinerea (grey heron) | schema: image-level species label",
    ],
    "arter": [
        "arter (DK) | CC-BY | 200 sp / 33,365 img",
        "sample: Buteo buteo (buzzard) | schema: image-level species label",
    ],
    "inat": [
        "inat / GBIF (R&D) | CC0 & CC-BY | 116 sp / 23,141 img",
        "sample: Anas platyrhynchos (mallard) | schema: + lat/lon/observed_at",
    ],
}


def font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            continue
    return ImageFont.load_default()


for src, lines in CARDS.items():
    im0 = Image.open(RAW / f"{src}.jpg").convert("RGB")
    # 统一宽度 720,等比
    w = 720
    h = int(im0.height * w / im0.width)
    im0 = im0.resize((w, h), Image.BILINEAR)
    fs = 19
    f = font(fs)
    bar = len(lines) * (fs + 8) + 10
    canvas = Image.new("RGB", (w, h + bar), (15, 20, 32))
    canvas.paste(im0, (0, bar))
    d = ImageDraw.Draw(canvas)
    y = 7
    for i, t in enumerate(lines):
        d.text((10, y), t, fill=(120, 200, 255) if i == 0 else (220, 220, 220), font=f)
        y += fs + 8
    canvas.save(A / f"card_cls_{src}.png")
    print("card:", src)
print("done")
