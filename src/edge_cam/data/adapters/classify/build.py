"""分类数据组装入口：config → 各源 adapter → assemble → 落 manifest + 署名清册。

镜像 `adapters/detect/build.py`，但分类产**单个 DatasetManifest**（split 在 record 里，训练
DataModule 按 split 过滤）。多源都用同一个 `gbif_birds` adapter（源无关），故 config 用 **sources
列表**（不像检测按 adapter 名 keying）：每源给 `adapter` + 子目录 `root` + 透传 overrides。

  python -m edge_cam.data.adapters.classify.build --config configs/data/classify_feeder.yaml

产物（out_dir）：
- manifest.json    —— DatasetManifest（root=raw_root，record.path 相对它，可移植换机）
- license_manifest.csv —— 逐图 path/source/license 署名清册（商用披露，[[ADR-0005]]）
- summary.json     —— 每源/每 split/每类 计数 + license 分布（落库可追溯）
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from edge_cam.contracts.schemas.dataset import DatasetManifest
from edge_cam.data.adapters.classify.base import assemble, build_adapter


@dataclass
class SourceEntry:
    """一个分类数据源：用哪个 adapter + 子目录 + 透传 overrides。"""

    adapter: str = "gbif_birds"
    root: str = ""  # 相对 raw_root 的源子目录（= source 名 / path 前缀）
    source: str | None = None  # provenance；默认取 root
    overrides: dict = field(default_factory=dict)  # max_per_class / taxonomy_csv / license_allow…

    def source_name(self) -> str:
        return self.source or self.root


@dataclass
class ClassifyBuildConfig:
    """分类组装配置。"""

    raw_root: str
    out_dir: str
    name: str = "feeder_classify"
    version: str = "v1"
    sources: list[SourceEntry] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ClassifyBuildConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        raw["sources"] = [SourceEntry(**s) for s in raw.get("sources", [])]
        return cls(**raw)


def _write_license_manifest(manifest: DatasetManifest, path: Path) -> None:
    """逐图 path/source/license 署名清册（商用披露要求，[[ADR-0005]]）。"""
    rows = {(r.path, r.source or "", r.license or "") for r in manifest.records}
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "source", "license"])
        w.writerows(sorted(rows))


def _summary(manifest: DatasetManifest) -> dict:
    """每源 / split / license / 类计数（落库可追溯，诚实计数）。"""
    by_source = Counter(r.source for r in manifest.records)
    by_license = Counter(r.license for r in manifest.records)
    species_per_split = {
        sp: len({r.label for r in manifest.records if r.split == sp})
        for sp in ("train", "val", "test")
    }
    return {
        "num_samples": manifest.num_samples,
        "num_classes": manifest.num_classes,
        "counts_by_split": manifest.counts_by_split(),
        "species_per_split": species_per_split,
        "by_source": dict(by_source),
        "by_license": dict(by_license),
    }


def build(cfg: ClassifyBuildConfig) -> Path:
    """构造各源 adapter → 组装单 manifest → 落盘；返回 out_dir。"""
    raw_root = Path(cfg.raw_root)
    adapters = [
        build_adapter(
            e.adapter,
            str(raw_root / e.root),
            source=e.source_name(),
            path_prefix=e.root,  # 合并后 record.path = <源子目录>/images/...，相对 raw_root
            **e.overrides,
        )
        for e in cfg.sources
    ]
    manifest = assemble(adapters, name=cfg.name, version=cfg.version, root=str(raw_root))

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest.save(out / "manifest.json")
    _write_license_manifest(manifest, out / "license_manifest.csv")
    summary = _summary(manifest)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[classify-build] {out / 'manifest.json'}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="组装分类训练/评估数据（各源 adapter → manifest）")
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)
    build(ClassifyBuildConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
