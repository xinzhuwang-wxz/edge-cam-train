"""② MegaDetector 伪标注 → COCO（保 score，供 ③ 置信分层）。

iNat 观测已筛 Aves → MD 的 **animal** 框即 bird（person 框在 bird 图上多为误检/噪声，丢）。
`build_pseudolabel_coco` 是纯函数（图元信息 + MD preds → COCO），可测；MD 推理 `run_pseudolabel`
复用 `eval.megadetector.run_megadetector`（lazy import pytorch-wildlife，隔离 env/GPU，
AGPL 不进产物）。

产物 annotation 带 `score` 与 `category name=animal`；`InatMdAdapter` 的 label_map `animal→bird`
直接吃。框 `label_provenance` 由分层后写盘阶段决定（md_pseudo/md_human_verified）。
"""

from __future__ import annotations

import json
from pathlib import Path

MD_ANIMAL_CLASSID = 1  # eval.megadetector 折叠后 animal=1（person=2 在 bird 图上丢）
ANIMAL_CATEGORIES = [{"id": 1, "name": "animal"}]


def build_pseudolabel_coco(
    images: list[dict],
    preds: list[dict],
    *,
    keep_category_id: int = MD_ANIMAL_CLASSID,
) -> dict:
    """图元信息 + MD preds → COCO（保 score）。**纯函数可测**。

    images: [{id, file_name, width, height}]（id 与 preds.image_id 对齐）。
    preds:  [{image_id, category_id, bbox[x,y,w,h], score}]（run_megadetector 产）。
    只留 category_id==keep_category_id（iNat：animal→bird）；输出 annotation 统一 category_id=1。
    """
    anns, ann_id = [], 1
    for p in preds:
        if p.get("category_id") != keep_category_id:
            continue
        anns.append(
            {
                "id": ann_id,
                "image_id": p["image_id"],
                "category_id": 1,
                "bbox": [float(v) for v in p["bbox"]],
                "score": float(p.get("score", 0.0)),
            }
        )
        ann_id += 1
    return {"images": images, "annotations": anns, "categories": ANIMAL_CATEGORIES}


def _image_meta(image_dir: Path, photo_ids: list[str]) -> tuple[list[dict], list]:
    """读每张图尺寸 → images 列表 + 对齐的轻量 records（含 .path/.width/.height，喂 MD 推理）。"""
    from dataclasses import dataclass

    from PIL import Image

    @dataclass
    class _Rec:
        path: str
        width: int
        height: int
        boxes: tuple = ()

    images, records = [], []
    for img_id, pid in enumerate(photo_ids):
        p = image_dir / f"{pid}.jpg"
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:  # noqa: BLE001 — 坏图跳过
            continue
        rel = f"{image_dir.name}/{pid}.jpg"
        images.append({"id": img_id, "file_name": rel, "width": w, "height": h})
        records.append(_Rec(path=rel, width=w, height=h))
    return images, records


def run_pseudolabel(
    image_dir: Path,
    *,
    out_json: Path,
    version: str = "MDV6-yolov9-c",
    conf: float = 0.2,
    weights: str | None = None,
    device: str = "cuda",
) -> dict:
    """box/GPU：对 image_dir 下 iNat 图跑 MD → 写 inat_md_coco.json（保 score）。

    conf=0.2 作**下限**（低于此不出框，对齐 triage 的 conf_lo）；分层在 ③ 做。
    `weights` 传本地权重路径 → 离线加载（绕 zenodo 下载，见 run_megadetector）。
    ⚠️ 上 box 前确认隔离 env 装了 pytorch-wildlife，且 version 串与其文档一致。
    """
    from edge_cam.eval.megadetector import run_megadetector

    photo_ids = sorted(p.stem for p in image_dir.glob("*.jpg"))
    images, records = _image_meta(image_dir, [pid for pid in photo_ids])
    # run_megadetector 的 image_id = records 下标 → 与 images[].id 对齐（_image_meta 同序构建）
    for i, img in enumerate(images):
        img["id"] = i
    # file_name 相对 SUBPATH（如 images/<id>.jpg，供 InatMdAdapter image_root=SUBPATH 读）→
    # MD 读图根用 image_dir.parent（=raw_root/commercial/inat_md），不是外层 raw_root
    preds = run_megadetector(
        records, str(image_dir.parent), version, conf=conf, weights=weights, device=device
    )
    coco = build_pseudolabel_coco(images, preds)
    out_json.write_text(json.dumps(coco, ensure_ascii=False))
    print(
        f"[md-pseudo] {len(images)} imgs → {sum(1 for _ in coco['annotations'])} boxes "
        f"(conf≥{conf}) → {out_json}",
        flush=True,
    )
    return coco
