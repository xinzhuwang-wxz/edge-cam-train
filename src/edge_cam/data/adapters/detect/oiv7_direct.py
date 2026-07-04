"""Open Images V7（官方直下）→ 5 类（[[ADR-0004]]）。**绕开 fiftyone**（py3.8 装不上，见
docs/detect/03 §4）：过滤 bbox CSV + 加速并行 S3 下图 → 生成 `oiv7_coco.json`
（categories=OIV7 display name），本 adapter 用 CocoJsonAdapter 吃它。

获取（ADR-0006 D2）收进本 adapter 的 `_fetch()`（取代旧 `fetch_oiv7_direct.py`，D0 不并存）：
bbox CSV 需先在 raw_dir（AcquireSpec.urls 记来源；国内下 S3 前 `source /etc/network_turbo` 加速），
`acquire()` 据此并行下图 + 写 coco。语义同 `fiftyone_oiv7.py`：非穷尽、CC-BY 可商用 + 署名。
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from edge_cam.data.adapters.detect.base import AcquireSpec, DatasetSpec, register_adapter
from edge_cam.data.adapters.detect.coco_json import CocoJsonAdapter
from edge_cam.data.adapters.detect.fiftyone_oiv7 import OIV7_LABEL_MAP

csv.field_size_limit(10**7)  # OIV6 bbox CSV 大/长行 → 抬 csv 字段上限（默认 128KB 会 _csv.Error）

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
    "Person": 3000,  # person 只需检到有人(不认身份、易检)→ ~3k 够（round2 分布：bird 命门优先）
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

_S3 = "https://open-images-dataset.s3.amazonaws.com"
_BBOX_CSV = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
_CLASSDESC_CSV = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"


def select_images(csv_path: Path, caps: dict[str, int]) -> dict[str, list[tuple[str, list[float]]]]:
    """扫 bbox CSV → {ImageID: [(name, [xmin,xmax,ymin,ymax] 归一化)]}，每类按 caps 限额选图。"""
    per_img: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    counts: dict[str, int] = dict.fromkeys(caps, 0)
    kept: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        # QUOTE_NONE：OIV6 CSV 无引号字段；默认解析遇杂引号会读到下个引号=巨字段（_csv.Error）
        reader = csv.reader(fh, quoting=csv.QUOTE_NONE)
        next(reader, None)  # header
        for row in reader:
            name = OIV7_MIDS.get(row[2])
            if name is None:
                continue
            iid = row[0]
            if iid not in kept:  # 选图：已选过该图直接累积框；否则该类未满才纳入
                if counts.get(name, 0) >= caps.get(name, 0):
                    continue
                kept.add(iid)
                counts[name] = counts.get(name, 0) + 1
            per_img[iid].append(
                (name, [float(row[4]), float(row[5]), float(row[6]), float(row[7])])
            )
    print(f"[oiv7] selected {len(per_img)} images; per-class img counts: {counts}", flush=True)
    return per_img


def _download_one(args: tuple[str, str, Path]) -> str | None:
    iid, split, out_img_dir = args
    dst = out_img_dir / f"{iid}.jpg"
    if dst.exists() and dst.stat().st_size > 0:
        return iid
    try:
        urllib.request.urlretrieve(f"{_S3}/{split}/{iid}.jpg", dst)  # 走 env http_proxy
        return iid
    except Exception:  # noqa: BLE001 — 个别图失败不致命，跳过
        dst.unlink(missing_ok=True)
        return None


def fetch_oiv7(ann_dir: Path, out_dir: Path, split: str, caps: dict[str, int], jobs: int) -> Path:
    """bbox CSV → 选图 → 并行 S3 下图 → 写 oiv7_coco.json（仅成功下到的图入 json，无缺图）。"""
    matches = list(
        ann_dir.glob("*train-annotations-bbox.csv" if split == "train" else f"{split}*bbox.csv")
    )
    if not matches:
        raise FileNotFoundError(
            f"open_images_v7: 缺 bbox CSV（期望在 {ann_dir}）。请下载：{_BBOX_CSV}"
        )
    per_img = select_images(matches[0], caps)
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


class Oiv7DirectAdapter(CocoJsonAdapter):
    """Open Images V7 直下 COCO → 5 类（非穷尽、CC-BY 可商用 + 署名）。"""

    SUBPATH = "commercial/open_images_v7"
    JSON_NAME = "oiv7_coco.json"

    def __init__(
        self,
        raw_root: str,
        *,
        negative_quota: int | None = 0,
        max_per_class: int | None = None,
        jobs: int = 16,
        **spec_overrides,
    ) -> None:
        self._base = Path(raw_root) / self.SUBPATH
        self._jobs = jobs
        spec = DatasetSpec(
            name="open_images_v7",
            raw_format="oiv7_direct_coco",
            label_map=OIV7_LABEL_MAP,
            # 逐图 Flickr 作者在 OIV image-ids CSV（发行前串入）；此处数据集级兜底兑现 CC-BY §4
            default_author="Open Images V7 / Flickr contributors (CC-BY-4.0)",
            license="CC-BY-4.0",
            commercial_safe=True,
            role="train",
            exhaustive=False,  # 按类拉 → 未标类区域不当负样本
            split_unit="image",
            attribution=True,  # 逐图署名清册
            acquire=AcquireSpec(
                method="s3_direct",
                urls=[_BBOX_CSV, _CLASSDESC_CSV, _S3],  # bbox CSV + 类描述 + 图 S3 桶
                version="v7",
            ),
            negative_quota=negative_quota,
            max_per_class=max_per_class,
            **spec_overrides,
        )
        # oiv7_coco.json 的 file_name 已含 images/<split>/ 前缀 → image_root=SUBPATH
        super().__init__(spec, json_path=self._base / self.JSON_NAME, image_root=self.SUBPATH)

    def _fetch(self, dest: Path) -> None:
        """并行 S3 下 train 图 + 写 oiv7_coco.json（bbox CSV 需先在 dest；加速见模块头）。"""
        if "http_proxy" not in __import__("os").environ:
            print("[oiv7][WARN] 无 http_proxy；建议先 source /etc/network_turbo", file=sys.stderr)
        fetch_oiv7(dest, dest, split="train", caps=DEFAULT_CAPS, jobs=self._jobs)


register_adapter("open_images_v7", Oiv7DirectAdapter)
