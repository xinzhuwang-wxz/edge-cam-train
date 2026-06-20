"""COCO 2017 → 5 类（**仅可行性**，[[ADR-0004]]/docs/detect/01 §3）。bird/cat/dog/person 带框。

COCO 图片版权杂 → **role=eval_only、commercial_safe=False**：只作泛化/bird 召回信号的额外评估，
**绝不进训练**（许可传染，§4）。标准 COCO instances JSON，类目小写、穷尽标注 → exhaustive=True。
squirrel 不在 COCO（无映射）。

下载（AutoDL）：
  - 标注: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  - 图片: http://images.cocodataset.org/zips/val2017.zip（~1GB，可行性用 val 即可）
"""

from __future__ import annotations

from pathlib import Path

from edge_cam.data.adapters.detect.base import DatasetSpec, register_adapter
from edge_cam.data.adapters.detect.coco_json import CocoJsonAdapter

# COCO 原类目（小写）→ 5 类。dog→other_animal；其余动物（horse/cow/elephant/bear/…）喂食器
# 见不到 → 丢。squirrel COCO 无。
COCO2017_LABEL_MAP: dict[str, str] = {
    "bird": "bird",
    "cat": "cat",
    "person": "person",
    "dog": "other_animal",
}


class Coco2017Adapter(CocoJsonAdapter):
    """COCO 2017 val → 5 类（仅可行性评估，不进训练）。"""

    SUBPATH = "feasibility/coco"
    JSON_NAME = "annotations/instances_val2017.json"
    IMAGE_SUBDIR = "val2017"

    def __init__(
        self,
        raw_root: str,
        *,
        negative_quota: int | None = 0,
        max_per_class: int | None = None,
        **spec_overrides,
    ) -> None:
        base = Path(raw_root) / self.SUBPATH
        spec = DatasetSpec(
            name="coco2017",
            raw_format="coco",
            label_map=COCO2017_LABEL_MAP,
            license="mixed-images-research",
            commercial_safe=False,
            role="eval_only",
            exhaustive=True,
            split_unit="image",
            negative_quota=negative_quota,
            max_per_class=max_per_class,
            **spec_overrides,
        )
        super().__init__(
            spec,
            json_path=base / self.JSON_NAME,
            image_root=f"{self.SUBPATH}/{self.IMAGE_SUBDIR}",
        )


register_adapter("coco2017", Coco2017Adapter)
