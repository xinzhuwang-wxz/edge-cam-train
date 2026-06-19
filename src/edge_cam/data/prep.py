"""分类数据准备编排（engineering §5.5 流水 2）：ingest → split → taxonomy → manifest。

config 驱动、确定性、可复现。产物 = DatasetManifest（可存盘，训练/评估共用）。
本地 CPU 即可跑（纯文件操作 + PIL 元数据）。

CLI:
    python -m edge_cam.data.prep --config configs/data/birds525.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from pydantic import BaseModel

from edge_cam.contracts.schemas.dataset import DatasetManifest, SampleRecord
from edge_cam.data.ingest import scan_imagefolder
from edge_cam.data.split import stratified_split
from edge_cam.data.taxonomy import IdentityTaxonomy, Taxonomy


class DataPrepConfig(BaseModel):
    """数据准备配置（可从 yaml 加载）。"""

    name: str
    version: str = "v0"
    root: str
    splits: dict[str, str] | None = None
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    seed: int = 0
    min_train_per_class: int = 1
    source: str = "unknown"
    license: str = "unknown"
    out_path: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataPrepConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def build_manifest(config: DataPrepConfig, taxonomy: Taxonomy | None = None) -> DatasetManifest:
    """扫描 → 分层 split → 归一 taxonomy → 组装 manifest（不落盘）。"""
    taxonomy = taxonomy or IdentityTaxonomy()
    items = scan_imagefolder(config.root, splits=config.splits)
    if not items:
        raise ValueError(f"prep: {config.root} 未扫到任何图片")

    assignment = stratified_split(items, config.ratios, config.seed, config.min_train_per_class)
    class_to_idx = {label: i for i, label in enumerate(sorted({lbl for _, lbl in items}))}
    root = str(Path(config.root).resolve())
    records = [
        SampleRecord(
            # 存相对 root 的路径，保证 manifest 可移植（换机用 data_root 覆盖即可）
            path=os.path.relpath(path, root),
            label=label,
            split=assignment[path],
            taxon_key=taxonomy.to_taxon_key(label),
            source=config.source,
            license=config.license,
        )
        for path, label in items
    ]
    return DatasetManifest(
        name=config.name,
        version=config.version,
        seed=config.seed,
        class_to_idx=class_to_idx,
        root=root,
        records=records,
    )


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
