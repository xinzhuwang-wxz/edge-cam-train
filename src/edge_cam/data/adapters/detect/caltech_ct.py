"""Caltech Camera Traps（LILA / ECCV18 CCT-20）→ 5 类（[[ADR-0004]]）。美西南相机陷阱。

**图源用 ECCV18 下采样集**（实操结论，见 docs/detect/03 §2）：全量 `cct_images.tar.gz` 达 107GB、
逐图 HTTP 404 不可选下；改用 ECCV18 `eccv_18_all_images_sm.tar.gz`（6.2G，57,864 下采样图，
检测 320/416 够用）与其注解 `eccv_18_annotations.tar.gz` 自洽——注解**自带 bbox + location/seq_id +
empty(id 30)**，
UUID 文件名与全量一致。故本 adapter 直接吃 ECCV18 注解（无需 caltech_bboxes/caltech_images 合并、
无需按图存在过滤——注解引用的图就是 sm 里的图）。

处理：
- 合并 5 个 split 注解（train/cis_val/trans_val/cis_test/trans_test）→ 单 coco；**按 location 划分**
  防泄漏（Terra-Incognita；同 location 的 seq 自然不跨 split）。
- `empty`(id 30) 注解无 bbox → CocoJsonAdapter 跳过 → 该图 0 框 → 负样本（域真实背景，§7）；
  `negative_quota=12000` 限额。`car` 等非目标丢。
- 全带框子集 → `exhaustive=True`。许可 CDLA-Permissive → 可商用。

下载（AutoDL）：base=storage.googleapis.com/public-datasets-lila/caltechcameratraps
  - 注解: {base}/eccv_18_annotations.tar.gz（解压到 eccv18/）
  - 图片: {base}/eccv_18_all_images_sm.tar.gz（解压到 IMAGE_SUBDIR）
"""

from __future__ import annotations

import json
from pathlib import Path

from edge_cam.data.adapters.detect.base import AcquireSpec, DatasetSpec, register_adapter
from edge_cam.data.adapters.detect.coco_json import CocoJsonAdapter

_LILA = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps"

# CCT 原类目（小写，精确）→ 5 类。empty/car/cow/pig/lizard/insect 不映射
# （empty→无框负样本；其余 DROP）。insect-only 图经 exhaustive 成困难负样本。
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
}


class CaltechCtAdapter(CocoJsonAdapter):
    """Caltech CT（ECCV18 子集）→ 5 类（按 location 划分、empty 补负样本、CDLA 可商用）。"""

    SUBPATH = "commercial/caltech_ct"
    IMAGE_SUBDIR = "eccv_18_all_images_sm"
    ECCV_DIR = "eccv18/eccv_18_annotation_files"
    ECCV_FILES = (
        "train_annotations.json",
        "cis_val_annotations.json",
        "trans_val_annotations.json",
        "cis_test_annotations.json",
        "trans_test_annotations.json",
    )

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
            raw_format="coco_camera_traps_eccv18",
            label_map=CCT_LABEL_MAP,
            license="CDLA-Permissive",
            commercial_safe=True,
            role="train",
            exhaustive=True,
            split_unit="location",
            acquire=AcquireSpec(  # LILA ECCV18 子集（manual：acquire 校验就位 + 给下载 URL）
                method="manual",
                urls=[
                    f"{_LILA}/eccv_18_annotations.tar.gz",
                    f"{_LILA}/eccv_18_all_images_sm.tar.gz",
                ],
                version="eccv18",
            ),
            negative_quota=negative_quota,
            max_per_class=max_per_class,
            **spec_overrides,
        )
        super().__init__(
            spec,
            json_path=self._base / self.ECCV_DIR / self.ECCV_FILES[0],  # 仅占位，_load_coco 覆盖
            image_root=f"{self.SUBPATH}/{self.IMAGE_SUBDIR}",
            group_key_field="location",
        )

    def _load_coco(self) -> dict:
        """合并 5 个 ECCV18 split 注解 → 单 coco dict（images 去重、annotations 并集）。"""
        images: list[dict] = []
        anns: list[dict] = []
        cats: list[dict] | None = None
        seen: set = set()
        for fn in self.ECCV_FILES:
            p = self._base / self.ECCV_DIR / fn
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            if cats is None:
                cats = d.get("categories")
            for im in d["images"]:
                if im["id"] not in seen:
                    seen.add(im["id"])
                    images.append(im)
            anns.extend(d.get("annotations", []))
        return {"images": images, "annotations": anns, "categories": cats or []}


register_adapter("caltech_ct", CaltechCtAdapter)
