"""检测数据集 manifest（[[ADR-0003]] C5）：与分类 DatasetManifest 对等,带 provenance/split。

替代「直接喂裸 COCO labels.json」——检测也有可移植、可溯源的统一清单:逐图 source/license/
taxon_key(与分类共享 `Provenanced`)+ 固定 split。`to_coco(split)` 产 NanoDet 可读的 COCO dict,
provenance 不丢。新检测源经 ingest adapter(merge_map 的源→11 粗类)产出本 manifest。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from edge_cam.contracts.schemas.dataset import Provenanced, Split, provenance_summary


class DetBox(BaseModel):
    """一个检测框：COCO bbox [x,y,w,h] + 粗类 id。"""

    bbox: list[float]
    category_id: int


class DetImageRecord(Provenanced):
    """单张检测图：路径 + 划分 + 尺寸 + 框 + 溯源（继承 Provenanced）。"""

    path: str
    split: Split
    width: int
    height: int
    boxes: list[DetBox] = Field(default_factory=list)


class DetectionManifest(BaseModel):
    """检测数据集完整清单(可存盘/加载;to_coco 喂 NanoDet)。root 语义同 DatasetManifest。"""

    name: str
    version: str
    seed: int = 0
    root: str | None = None
    categories: dict[str, int]  # 粗类名 → id(11 feeder-cam 闭集)
    records: list[DetImageRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> DetectionManifest:
        ids = set(self.categories.values())
        bad = {b.category_id for r in self.records for b in r.boxes} - ids
        if bad:
            raise ValueError(f"boxes 含未在 categories 声明的 category_id: {sorted(bad)[:5]}")
        return self

    def resolve_path(self, record: DetImageRecord, data_root: str | None = None) -> Path:
        base = data_root or self.root
        return Path(base) / record.path if base else Path(record.path)

    @property
    def num_classes(self) -> int:
        return len(self.categories)

    def counts_by_split(self) -> dict[str, int]:
        c = Counter(r.split for r in self.records)
        return {s: c.get(s, 0) for s in ("train", "val", "test")}

    def provenance(self) -> tuple[list[str], list[str]]:
        """(datasets, licenses) 汇总（与分类共用 provenance_summary）。"""
        return provenance_summary(self.records)

    def to_coco(self, split: Split) -> dict:
        """导出某 split 的 COCO labels dict（NanoDet CocoDataset 可读）。

        category id 用 1-indexed(COCO 惯例);file_name = record.path(相对 root)。"""
        id_shift = 1  # categories 内部 0-based → COCO 1-based
        ordered = sorted(self.categories.items(), key=lambda kv: kv[1])
        cats = [{"id": i + id_shift, "name": n} for n, i in ordered]
        images, annotations, ann_id = [], [], 1
        split_recs = (x for x in self.records if x.split == split)
        for img_id, r in enumerate(split_recs):
            images.append({"id": img_id, "file_name": r.path, "width": r.width, "height": r.height})
            for b in r.boxes:
                x, y, w, h = b.bbox
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": b.category_id + id_shift,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
        return {"images": images, "annotations": annotations, "categories": cats}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> DetectionManifest:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
