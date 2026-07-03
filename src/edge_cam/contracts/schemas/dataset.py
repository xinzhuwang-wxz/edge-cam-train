"""分类数据集的 pydantic 契约（engineering §6：含边侧 Literal + 版本）。

manifest 是数据准备的产物，也是训练/评估的输入：固定 seed 划分 + 类索引 +
逐样本溯源（source/license/taxon_key），保证实验可复现与许可可追溯。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Split = Literal["train", "val", "test"]


class Provenanced(BaseModel):
    """逐样本溯源 + 可选蒸馏软标签（两族共享基，[[ADR-0003]] C5）。

    source/license：许可红线全链路可追溯(随产物披露署名清册);taxon_key：[[ADR-0002]] eBird 规范键
    (跨集合并 + 地域 mask + 级联 bird↔species 桥)。soft_label：教师模型软标签,蒸馏时挂在同一
    schema 上 → 蒸馏 = 加字段、数据管线复用(默认 None,不影响现有数据)。"""

    source: str = "unknown"
    license: str = "unknown"  # SPDX 标识优先（ADR-0006 D4）
    taxon_key: str | None = None
    # 逐样本署名（ADR-0006 D4）：兑现 CC-BY 逐图清册（§4）。默认 None → 向后兼容，不影响现有数据。
    author: str | None = None  # 原作者（CC-BY 署名要求）
    original_url: str | None = None  # 原图/原记录 URL（署名 + 可追溯）
    source_media_id: str | None = None  # 源侧媒体 id（如 iNat photo_id）→ 去重/防泄漏 join
    asset_sha256: str | None = None  # 图内容哈希 → 跨源去重 + 完整性
    # 教师 logits/概率:**前瞻 hook**,暂无消费方,待蒸馏(distill #7)落地;默认 None 不影响现有数据。
    soft_label: list[float] | None = None  # noqa: E501 (issue #16 决策:保留为 hook)


class SampleRecord(Provenanced):
    """单张分类样本：路径 + 类标 + 划分 + 溯源（+ 可选软标签）。"""

    path: str
    label: str
    split: Split


def provenance_summary(records: Iterable[Provenanced]) -> tuple[list[str], list[str]]:
    """从逐样本溯源汇出 (datasets, licenses)（去重保序，忽略 unknown）。两族 manifest 共用。"""
    datasets = sorted({r.source for r in records if r.source and r.source != "unknown"})
    licenses = sorted({r.license for r in records if r.license and r.license != "unknown"})
    return datasets, licenses


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
