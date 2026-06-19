"""分类数据集的 pydantic 契约（engineering §6：含边侧 Literal + 版本）。

manifest 是数据准备的产物，也是训练/评估的输入：固定 seed 划分 + 类索引 +
逐样本溯源（source/license/taxon_key），保证实验可复现与许可可追溯。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Split = Literal["train", "val", "test"]


class SampleRecord(BaseModel):
    """单张样本：路径 + 类标 + 划分 + 溯源。"""

    path: str
    label: str
    split: Split
    taxon_key: str | None = None
    source: str = "unknown"
    license: str = "unknown"


class DatasetManifest(BaseModel):
    """一份数据集的完整清单（可存盘 / 加载，跨实验复用同一划分）。

    可移植：records.path 存**相对 root 的路径**，root 记录 prep 时的数据根。
    换机（如租 GPU 卡）时上传 raw 数据后，用 data_root 覆盖即可复用同一份 manifest，
    无需重跑 prep。root=None 表示旧格式（path 为绝对路径），仍兼容。
    """

    name: str
    version: str
    seed: int
    class_to_idx: dict[str, int]
    root: str | None = None
    records: list[SampleRecord] = Field(default_factory=list)

    def resolve_path(self, record: SampleRecord, data_root: str | None = None) -> Path:
        """把 record 的（相对）路径解析为可读绝对路径。

        优先级：显式 data_root > manifest.root > 把 record.path 当绝对路径（旧格式）。
        """
        base = data_root or self.root
        return Path(base) / record.path if base else Path(record.path)

    @model_validator(mode="after")
    def _check_integrity(self) -> DatasetManifest:
        idxs = sorted(self.class_to_idx.values())
        if idxs != list(range(len(idxs))):
            raise ValueError("class_to_idx 的索引必须是 0..N-1 连续整数")
        unknown = {r.label for r in self.records} - set(self.class_to_idx)
        if unknown:
            raise ValueError(f"records 含未在 class_to_idx 声明的类: {sorted(unknown)[:5]}")
        return self

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    @property
    def num_samples(self) -> int:
        return len(self.records)

    def counts_by_split(self) -> dict[str, int]:
        """{split: 样本数}（按 train/val/test 固定顺序，缺失补 0）。"""
        counter = Counter(r.split for r in self.records)
        return {split: counter.get(split, 0) for split in ("train", "val", "test")}

    def class_counts(self, split: Split | None = None) -> dict[str, int]:
        """{label: 样本数}，可限定某个 split。"""
        return dict(Counter(r.label for r in self.records if split is None or r.split == split))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> DatasetManifest:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
