"""Caltech Camera Traps（LILA）→ 5 类（[[ADR-0004]]）。美西南 244k 图，22 类，含 empty 空帧。

两份 JSON（COCO-Camera-Traps）：
  - **bbox 子集** `caltech_bboxes_20200316.json`：带框标注（本 adapter 的框来源）。
  - **图级标注** `caltech_images_20210113.json`：含 `location`/`seq_id` 元数据 + `empty` 空帧标签
    （空帧无框 → 不在 bbox json 里，须从图级 json 补成负样本）。

本 adapter 合二为一：① 从图级 json 把 `location`/`seq_id` join 进 bbox 图记录 →
**按 location 划分**（论文 Terra-Incognita 法，防泄漏；同 location 的 seq 自然不跨 split）；
② 把 `empty` 帧（图级 json 标 empty 的图）补进来当**负样本**（域真实背景，docs/detect/01 §7 主力）。

bbox 子集为穷尽框 → exhaustive=True。许可 CDLA-Permissive → 可商用。

下载（AutoDL；任选 GCP/Azure/AWS）：
  base=https://storage.googleapis.com/public-datasets-lila/caltechcameratraps
  - 图片: {base}/cct_images.tar.gz（解压到 image_root）
  - bbox: {base}/labels/caltech_bboxes_20200316.json
  - 图级: {base}/labels/caltech_camera_traps.json.zip → caltech_images_20210113.json
"""

from __future__ import annotations

import json
from pathlib import Path

from edge_cam.data.adapters.detect.base import DatasetSpec, register_adapter
from edge_cam.data.adapters.detect.coco_json import CocoJsonAdapter

# CCT 原类目（小写，精确）→ 5 类。empty 不映射（无框 → 负样本）；
# car/cow/pig（车辆/牲畜）与 lizard/insect（非哺乳）→ 丢。insect-only 图经 exhaustive 成困难负样本。
CCT_LABEL_MAP: dict[str, str] = {
    "bird": "bird",
    "squirrel": "squirrel",
    "cat": "cat",
    "opossum": "other_animal",
    "raccoon": "other_animal",
    "bobcat": "other_animal",
    "skunk": "other_animal",
    "dog": "other_animal",
    "coyote": "other_animal",
    "rabbit": "other_animal",
    "badger": "other_animal",
    "deer": "other_animal",
    "mountain_lion": "other_animal",
    "fox": "other_animal",
    "bat": "other_animal",
    "rodent": "other_animal",
    # 不映射（丢）：empty(→负样本), car, cow, pig, lizard, insect
}


class CaltechCtAdapter(CocoJsonAdapter):
    """Caltech CT → 5 类（按 location 划分、empty 补负样本、CDLA 可商用）。"""

    SUBPATH = "commercial/caltech_ct"
    IMAGE_SUBDIR = "cct_images"
    BBOX_JSON = "caltech_bboxes_20200316.json"
    META_JSON = "caltech_images_20210113.json"

    def __init__(
        self,
        raw_root: str,
        *,
        negative_quota: int | None = 12000,  # empty 空帧主力负样本（docs/detect/01 §7）
        max_per_class: int | None = None,
        **spec_overrides,
    ) -> None:
        self._base = Path(raw_root) / self.SUBPATH
        spec = DatasetSpec(
            name="caltech_ct",
            raw_format="coco_camera_traps",
            label_map=CCT_LABEL_MAP,
            license="CDLA-Permissive",
            commercial_safe=True,
            role="train",
            exhaustive=True,
            split_unit="location",
            negative_quota=negative_quota,
            max_per_class=max_per_class,
            **spec_overrides,
        )
        super().__init__(
            spec,
            json_path=self._base / self.BBOX_JSON,
            image_root=f"{self.SUBPATH}/{self.IMAGE_SUBDIR}",
            group_key_field="location",
        )

    def _load_coco(self) -> dict:
        """bbox json + 图级 json 合并：join location/seq_id，补 empty 帧当负样本。"""
        coco = super()._load_coco()  # bbox 子集（图 + 框）
        meta_path = self._base / self.META_JSON
        if not meta_path.exists():
            return coco  # 仅 bbox json：location 缺 → 退化为按 path 划分（无 empty 负样本）

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        by_id = {im["id"]: im for im in meta["images"]}
        # ① 把 location/seq_id join 进 bbox 图记录（防泄漏分组键）
        for im in coco["images"]:
            m = by_id.get(im["id"])
            if m is not None:
                if m.get("location") is not None:
                    im.setdefault("location", m["location"])
                if m.get("seq_id") is not None:
                    im.setdefault("seq_id", m["seq_id"])
        # ② 补 empty 空帧（图级 json 标 empty 的图，bbox json 里没有）→ 无框 → 负样本
        empty_cat_ids = {c["id"] for c in meta["categories"] if c["name"] == "empty"}
        empty_img_ids = {
            a["image_id"] for a in meta.get("annotations", []) if a["category_id"] in empty_cat_ids
        }
        present = {im["id"] for im in coco["images"]}
        for iid in empty_img_ids:
            if iid not in present and iid in by_id:
                coco["images"].append(by_id[iid])  # 无对应 annotations → load_raw 标 is_negative
        return coco


register_adapter("caltech_ct", CaltechCtAdapter)
