"""Open Images V7 直下（绕开 fiftyone）：bbox CSV 过滤 → 加速并行 S3 下图 → COCO json。

为什么不用 fiftyone：py3.8 上 fiftyone-db wheel 构建挂死（见 docs/detect/03 §4）。
为什么能下：AWS S3 直连被限速（~10KB/s），但 AutoDL 学术加速（`source /etc/network_turbo` 设
http_proxy）后 ~9× 提速。本脚本用 urllib，自动走环境里的 http(s)_proxy。

产出：<out>/images/<split>/<ImageID>.jpg + <out>/oiv7_coco.json（categories=OIV7 display name；
仅成功下到的图入 json → 无缺图）。下游 `Oiv7DirectAdapter`(CocoJsonAdapter) 吃这份 json。

用法（box 上，先 `source /etc/network_turbo`）：
  python scripts/fetch_oiv7_direct.py --ann <ann_dir> --out <out_dir> --split train --jobs 32
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 目标 MID → OIV7 display name（核验；只取动物 Mouse，不取 Computer mouse）。
OIV7_MIDS: dict[str, str] = {
    "/m/015p6": "Bird",
    "/m/071qp": "Squirrel",
    "/m/01yrx": "Cat",
    "/m/01g317": "Person",
    "/m/0bt9lr": "Dog",
    "/m/0dq75": "Raccoon",
    "/m/06mf6": "Rabbit",
    "/m/0306r": "Fox",
    "/m/09kx5": "Deer",
    "/m/04rmv": "Mouse",
    "/m/03qrc": "Hamster",
    "/m/0cl4p": "Hedgehog",
    "/m/0km7z": "Skunk",
}

# 每类**含该类图**上限（补 CCT 缺口：bird/person 多，other_animal 少——CCT 已 ~26k）。
DEFAULT_CAPS: dict[str, int] = {
    "Bird": 15000,
    "Person": 8000,
    "Squirrel": 6000,
    "Cat": 5000,
    "Dog": 1500,
    "Raccoon": 1500,
    "Rabbit": 1500,
    "Fox": 1200,
    "Deer": 1500,
    "Mouse": 800,
    "Hamster": 600,
    "Hedgehog": 600,
    "Skunk": 800,
}

S3 = "https://open-images-dataset.s3.amazonaws.com"


def select_images(csv_path: Path, caps: dict[str, int]) -> dict[str, list[tuple[str, list[float]]]]:
    """扫 bbox CSV → {ImageID: [(name, [xmin,xmax,ymin,ymax] 归一化)]}，每类按 caps 限额选图。"""
    per_img: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    counts: dict[str, int] = dict.fromkeys(caps, 0)
    kept: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            mid = row[2]
            name = OIV7_MIDS.get(mid)
            if name is None:
                continue
            iid = row[0]
            # 选图：已选过该图 → 直接累积框；否则该类未满才纳入
            if iid not in kept:
                if counts.get(name, 0) >= caps.get(name, 0):
                    continue
                kept.add(iid)
                counts[name] = counts.get(name, 0) + 1
            box = [float(row[4]), float(row[5]), float(row[6]), float(row[7])]
            per_img[iid].append((name, box))
    print(f"[oiv7] selected {len(per_img)} images; per-class img counts: {counts}", flush=True)
    return per_img


def _download_one(args: tuple[str, str, Path]) -> str | None:
    iid, split, out_img_dir = args
    dst = out_img_dir / f"{iid}.jpg"
    if dst.exists() and dst.stat().st_size > 0:
        return iid
    try:
        urllib.request.urlretrieve(f"{S3}/{split}/{iid}.jpg", dst)  # 走 env http_proxy
        return iid
    except Exception:  # noqa: BLE001 — 个别图失败不致命，跳过
        if dst.exists():
            dst.unlink(missing_ok=True)
        return None


def fetch(ann_dir: Path, out_dir: Path, split: str, caps: dict[str, int], jobs: int) -> Path:
    pat = "*train-annotations-bbox.csv" if split == "train" else f"{split}*bbox.csv"
    bbox_csv = next(ann_dir.glob(pat))
    per_img = select_images(bbox_csv, caps)
    img_dir = out_dir / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)

    ids = list(per_img)
    print(f"[oiv7] downloading {len(ids)} images → {img_dir} (jobs={jobs})", flush=True)
    ok: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for i, res in enumerate(ex.map(_download_one, ((iid, split, img_dir) for iid in ids)), 1):
            if res:
                ok.append(res)
            if i % 1000 == 0:
                print(f"[oiv7] {i}/{len(ids)} done ({len(ok)} ok)", flush=True)
    print(f"[oiv7] downloaded {len(ok)}/{len(ids)}", flush=True)

    return _write_coco(per_img, ok, split, img_dir, out_dir)


def _write_coco(per_img, ok_ids, split, img_dir, out_dir) -> Path:
    from PIL import Image

    name_to_cat = {n: i for i, n in enumerate(sorted(set(OIV7_MIDS.values())), start=1)}
    cats = [{"id": i, "name": n} for n, i in name_to_cat.items()]
    images, anns, ann_id = [], [], 1
    for img_id, iid in enumerate(ok_ids, 1):
        p = img_dir / f"{iid}.jpg"
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:  # noqa: BLE001 — 坏图跳过
            continue
        images.append(
            {"id": img_id, "file_name": f"images/{split}/{iid}.jpg", "width": w, "height": h}
        )
        for name, (xmin, xmax, ymin, ymax) in per_img[iid]:
            anns.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": name_to_cat[name],
                    "bbox": [xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h],
                }
            )
            ann_id += 1
    out_json = out_dir / "oiv7_coco.json"
    out_json.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
    print(f"[oiv7] wrote {out_json}: {len(images)} imgs / {len(anns)} boxes", flush=True)
    return out_json


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Open Images V7 直下 → COCO json")
    ap.add_argument("--ann", required=True, help="含 *-bbox.csv 的目录")
    ap.add_argument("--out", required=True, help="输出根（images/ + oiv7_coco.json）")
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--jobs", type=int, default=32)
    a = ap.parse_args(argv)
    if "http_proxy" not in __import__("os").environ:
        print("[oiv7][WARN] 无 http_proxy；建议先 source /etc/network_turbo", file=sys.stderr)
    fetch(Path(a.ann), Path(a.out), a.split, DEFAULT_CAPS, a.jobs)


if __name__ == "__main__":
    main()
