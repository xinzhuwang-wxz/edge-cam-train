"""Open Images V7（FiftyOne zoo）→ 5 类（[[ADR-0004]]）。网图带框，全 5 类主力商用源。

FiftyOne `load_zoo_dataset("open-images-v7", classes=[...])` 按类拉子集（自动下载+缓存）。
类目字符串为 OIV7 display name（**首词首字母大写**，经核验）。**按类拉 → 其它类未标 →
exhaustive=False**（未标区域不当负样本，§5.1 防漏标污染）。逐图 CC-BY-4.0 → 可商用，但须逐图
署名清册（attribution=True，docs/detect/01 §3）。

⚠️ `Mouse` = 动物鼠，OIV7 另有独立类 `Computer mouse`（电脑鼠标）——**只取 `Mouse`**。
FiftyOne 框 = `bounding_box` 相对坐标 [x,y,w,h]∈[0,1]，本 adapter 转绝对像素（COCO 惯例）。
record.path 存 FiftyOne 缓存绝对路径（assemble 合并时绝对路径不受 manifest.root 影响）。

无内置 ENA24/CCT loader（仅 OIV7）——故 LILA 两源走 CocoJsonAdapter，OIV7 独走本 adapter。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from edge_cam.data.adapters.detect.base import (
    DatasetSpec,
    DetectionDatasetAdapter,
    RawSample,
    register_adapter,
)

# OIV7 display name（精确，首词大写）→ 5 类。Otter/Mule 等非后院场景不取。
OIV7_LABEL_MAP: dict[str, str] = {
    "Bird": "bird",
    "Squirrel": "squirrel",
    "Cat": "cat",
    "Person": "person",
    "Dog": "other_animal",
    "Raccoon": "other_animal",
    "Rabbit": "other_animal",
    "Fox": "other_animal",
    "Deer": "other_animal",
    "Mouse": "other_animal",  # 动物鼠（≠ "Computer mouse"）
    "Hamster": "other_animal",
    "Hedgehog": "other_animal",
    "Skunk": "other_animal",
}


def _rel_to_abs(bbox_rel: list[float], width: int, height: int) -> list[float]:
    """FiftyOne 相对框 [x,y,w,h]∈[0,1] → 绝对像素 [x,y,w,h]（COCO 惯例）。"""
    x, y, w, h = bbox_rel
    return [x * width, y * height, w * width, h * height]


class FiftyOneOiv7Adapter(DetectionDatasetAdapter):
    """Open Images V7 → 5 类（按类拉、非穷尽、CC-BY 可商用 + 署名）。"""

    def __init__(
        self,
        raw_root: str | None = None,
        *,
        splits: tuple[str, ...] = ("train",),
        max_samples: int | None = None,
        negative_quota: int | None = 0,
        max_per_class: int | None = None,
        **spec_overrides,
    ) -> None:
        spec = DatasetSpec(
            name="open_images_v7",
            raw_format="fiftyone_oiv7",
            label_map=OIV7_LABEL_MAP,
            license="CC-BY-4.0",
            commercial_safe=True,
            role="train",
            exhaustive=False,  # 按类拉 → 未标类区域不当负样本
            split_unit="image",
            attribution=True,  # 逐图署名清册
            negative_quota=negative_quota,
            max_per_class=max_per_class,
            **spec_overrides,
        )
        super().__init__(spec)
        # 缓存到自己子目录（raw_root 是 detect 公共根；build 给所有 adapter 传同一个）
        self.cache_dir = str(Path(raw_root) / "commercial/open_images_v7") if raw_root else None
        self.splits = splits
        self.max_samples = max_samples

    @staticmethod
    def _sample_to_raw(filepath: str, width: int, height: int, dets: list[tuple[str, list[float]]]):
        """(路径, 宽高, [(源标签, 相对框)]) → RawSample（框转绝对像素）。"""
        boxes = [(label, _rel_to_abs(bbox, width, height)) for label, bbox in dets]
        return RawSample(path=filepath, width=width, height=height, boxes=boxes)

    @staticmethod
    def _detections(sample, fo) -> list[tuple[str, list[float]]]:
        """从 FiftyOne 样本取第一个 Detections 字段 → [(label, 相对 bbox)]。"""
        for _name, value in sample.iter_fields():
            if isinstance(value, fo.Detections):
                return [(d.label, list(d.bounding_box)) for d in value.detections]
        return []

    def load_raw(self) -> Iterable[RawSample]:
        import fiftyone as fo
        import fiftyone.zoo as foz

        classes = list(self.spec.label_map)
        for split in self.splits:
            kwargs: dict = {"split": split, "label_types": ["detections"], "classes": classes}
            if self.max_samples is not None:
                kwargs["max_samples"] = self.max_samples
            if self.cache_dir:
                kwargs["dataset_dir"] = self.cache_dir
            ds = foz.load_zoo_dataset("open-images-v7", **kwargs)
            ds.compute_metadata()  # 确保 metadata.width/height（相对→绝对框要用）
            for sample in ds.iter_samples(progress=True):
                meta = sample.metadata
                w = int(meta.width) if meta and meta.width else 0
                h = int(meta.height) if meta and meta.height else 0
                yield self._sample_to_raw(sample.filepath, w, h, self._detections(sample, fo))


# 直下版（oiv7_direct）注册为默认 open_images_v7；fiftyone 版留待 py3.10+ 环境用此别名。
register_adapter("open_images_v7_fiftyone", FiftyOneOiv7Adapter)
