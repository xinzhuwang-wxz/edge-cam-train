"""分类数据准备编排（engineering §5.5 流水 2）：ingest → split → taxonomy → manifest。

config 驱动、确定性、可复现。产物 = DatasetManifest（可存盘，训练/评估共用）。
本地 CPU 即可跑（纯文件操作 + PIL 元数据）。

CLI:
    python -m edge_cam.data.prep --config configs/data/birds525.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

from edge_cam.contracts.schemas.dataset import DatasetManifest, SampleRecord
from edge_cam.data.ingest import scan_imagefolder
from edge_cam.data.split import stratified_split
from edge_cam.data.taxonomy import EbirdTaxonomy, IdentityTaxonomy, Taxonomy


class SourceSpec(BaseModel):
    """单个有标签数据源（多源合并用，架构审查 B / ADR-0002）。

    每源各自的 ImageFolder 根 + 溯源 + taxonomy 表 → 经 to_taxonomy 归一到 eBird 规范键，
    多源据此合并为同一套物种类。"""

    root: str
    source: str = "unknown"
    license: str = "unknown"
    splits: dict[str, str] | None = None
    taxonomy_csv: str | None = None
    taxonomy_version: str = "ebird-unversioned"

    def to_taxonomy(self) -> Taxonomy:
        if self.taxonomy_csv:
            return EbirdTaxonomy.from_csv(self.taxonomy_csv, version=self.taxonomy_version)
        return IdentityTaxonomy()


class DataPrepConfig(BaseModel):
    """数据准备配置（可从 yaml 加载）。

    两种模式：① 单源（顶层 root/source/license/taxonomy_csv，向后兼容，class=label）；
    ② 多源（sources 列表，架构审查 B）—— 每源各自 taxonomy 归一到 eBird，**按 taxon_key 合并
    物种类**（同种跨集并为一类）、跨源内容去重、未映射样本丢弃。"""

    name: str
    version: str = "v0"
    # 单源
    root: str | None = None
    splits: dict[str, str] | None = None
    source: str = "unknown"
    license: str = "unknown"
    taxonomy_csv: str | None = None
    taxonomy_version: str = "ebird-unversioned"
    # 多源（给定则走合并；与单源 root 二选一）
    sources: list[SourceSpec] | None = None
    dedup: bool = True  # 多源跨集按内容哈希去重
    # 公共
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    seed: int = 0
    min_train_per_class: int = 1
    out_path: str | None = None

    @model_validator(mode="after")
    def _require_source(self) -> DataPrepConfig:
        if not self.root and not self.sources:
            raise ValueError("DataPrepConfig: 需提供 root（单源）或 sources（多源）之一")
        if self.root and self.sources:
            raise ValueError("DataPrepConfig: root 与 sources 不能同时给（单源/多源二选一）")
        return self

    def source_specs(self) -> list[SourceSpec]:
        """归一成 SourceSpec 列表（单源 = 由顶层字段构造一个）。"""
        if self.sources:
            return self.sources
        return [
            SourceSpec(
                root=self.root,  # type: ignore[arg-type]
                source=self.source,
                license=self.license,
                splits=self.splits,
                taxonomy_csv=self.taxonomy_csv,
                taxonomy_version=self.taxonomy_version,
            )
        ]

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataPrepConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def taxonomy_from_config(config: DataPrepConfig) -> Taxonomy:
    """单源 taxonomy adapter（ADR-0002）：有 csv → EbirdTaxonomy，否则 Identity 占位。"""
    return config.source_specs()[0].to_taxonomy()


@dataclass
class _Collected:
    """一条扫描结果（合并前的中间态）。"""

    abs_path: str
    class_key: str  # 训练类标（单源=原 label；多源=eBird taxon_key）
    taxon_key: str | None
    source: str
    license: str
    src_root: str


def _content_hash(path: str) -> str:
    h = hashlib.sha1()  # noqa: S324 — 仅用于去重，非安全用途
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(config: DataPrepConfig, taxonomy: Taxonomy | None = None) -> DatasetManifest:
    """扫描 → （多源）按 eBird 键合并 + 去重 → 分层 split → 组装 manifest（不落盘）。

    taxonomy 参数仅单源时生效（覆盖 config）；多源时每源用自己的 SourceSpec.to_taxonomy。
    """
    specs = config.source_specs()
    multi = len(specs) > 1

    collected: list[_Collected] = []
    dropped = 0
    for spec in specs:
        tax = taxonomy if (taxonomy and not multi) else spec.to_taxonomy()
        src_root = str(Path(spec.root).resolve())
        items = scan_imagefolder(spec.root, splits=spec.splits)
        for path, label in items:
            tkey = tax.to_taxon_key(label)
            # 多源：按 eBird 规范键合并物种类；未映射无法合并 → 丢弃（诚实计数）
            if multi:
                if tkey is None:
                    dropped += 1
                    continue
                class_key = tkey
            else:
                class_key = label
            collected.append(
                _Collected(
                    str(Path(path).resolve()), class_key, tkey, spec.source, spec.license, src_root
                )
            )

    if not collected:
        raise ValueError(
            f"prep: {[s.root for s in specs]} 未扫到可用样本（多源时检查 taxonomy 映射）"
        )

    if multi and config.dedup:
        collected = _dedup_by_content(collected)
    if multi and dropped:
        print(f"[prep] 多源合并：丢弃 {dropped} 张无 eBird 映射样本（未在 taxonomy 表中）")

    items_for_split = [(c.abs_path, c.class_key) for c in collected]
    assignment = stratified_split(
        items_for_split, config.ratios, config.seed, config.min_train_per_class
    )
    class_to_idx = {k: i for i, k in enumerate(sorted({c.class_key for c in collected}))}

    # 单源：相对路径 + 记录 root，保持可移植；多源：多个 root，存绝对路径（root=None）
    single_root = str(Path(specs[0].root).resolve()) if not multi else None
    records = [
        SampleRecord(
            path=os.path.relpath(c.abs_path, single_root) if single_root else c.abs_path,
            label=c.class_key,
            split=assignment[c.abs_path],
            taxon_key=c.taxon_key,
            source=c.source,
            license=c.license,
        )
        for c in collected
    ]
    return DatasetManifest(
        name=config.name,
        version=config.version,
        seed=config.seed,
        class_to_idx=class_to_idx,
        root=single_root,
        records=records,
    )


def _dedup_by_content(collected: list[_Collected]) -> list[_Collected]:
    """跨源按文件内容哈希去重，保留首次出现（确定性）。"""
    seen: set[str] = set()
    out: list[_Collected] = []
    for c in collected:
        digest = _content_hash(c.abs_path)
        if digest not in seen:
            seen.add(digest)
            out.append(c)
    return out


def prepare(config: DataPrepConfig, taxonomy: Taxonomy | None = None) -> DatasetManifest:
    """build_manifest + 可选落盘。"""
    manifest = build_manifest(config, taxonomy)
    if config.out_path:
        out = Path(config.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest.save(out)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="构建分类数据集 manifest")
    parser.add_argument("--config", required=True, help="DataPrepConfig yaml 路径")
    args = parser.parse_args(argv)

    config = DataPrepConfig.from_yaml(args.config)
    manifest = prepare(config)
    print(
        f"[prep] {manifest.name} {manifest.version}: "
        f"{manifest.num_classes} classes / {manifest.num_samples} samples"
    )
    print(f"[prep] splits: {manifest.counts_by_split()}")
    if config.out_path:
        print(f"[prep] saved → {config.out_path}")


if __name__ == "__main__":
    main()
