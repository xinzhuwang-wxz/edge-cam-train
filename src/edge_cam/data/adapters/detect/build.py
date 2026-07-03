"""检测数据组装入口：config → 各 adapter → assemble → 落 manifest + NanoDet labels + 署名清册。

「在 AutoDL 整理数据」= 把各 adapter 下载好的 raw 放到 `raw_root` 下的 layer/dataset 子目录，
跑本模块产出 `data/processed/detect/`（docs/detect/01 §5/§9）。加新源 = 配置里加一行
`datasets: {<name>: {...overrides}}`，无需改本文件（registry + build_adapter 解耦）。

  python -m edge_cam.data.adapters.detect.build --config configs/data/detect_feeder5.yaml
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from edge_cam.contracts.schemas.detection_manifest import DetectionManifest
from edge_cam.data.adapters.detect.base import assemble, build_adapter

_SPLITS = ("train", "val", "test")


@dataclass
class DetectBuildConfig:
    """检测组装配置。datasets: {adapter 名: 透传给 build_adapter 的 overrides}。"""

    raw_root: str
    out_dir: str
    name: str = "feeder_detect"
    version: str = "v1"
    datasets: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DetectBuildConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**raw)


_LICENSE_COLS = [
    "path",
    "source",
    "license",
    "author",
    "original_url",
    "source_media_id",
    "asset_sha256",
]


def _write_license_manifest(manifests: dict[str, DetectionManifest], path: Path) -> None:
    """逐图署名清册（训练集，商用披露要求，docs/detect/01 §3 + ADR-0006 D4）。

    扩到 author/original_url/source_media_id/asset_sha256 → **真正兑现 CC-BY 逐图署名**
    （旧版只 path/source/license 不足以满足 CC-BY 署名要求）。"""
    rows = {
        (
            r.path,
            r.source or "",
            r.license or "",
            r.author or "",
            r.original_url or "",
            r.source_media_id or "",
            r.asset_sha256 or "",
        )
        for key in ("train", "test")
        for r in manifests[key].records
    }
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_LICENSE_COLS)
        w.writerows(sorted(rows))


def build(cfg: DetectBuildConfig) -> Path:
    """构造各 adapter → 组装 → 落盘；返回 out_dir。"""
    adapters = [
        build_adapter(name, cfg.raw_root, **(ov or {})) for name, ov in cfg.datasets.items()
    ]
    manifests = assemble(adapters, name=cfg.name, version=cfg.version)

    out = Path(cfg.out_dir)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for key, m in manifests.items():
        m.root = cfg.raw_root  # 训练时 manifest.root + record.path 定位图片
        m.save(out / f"manifest_{key}.jsonl")  # JSONL + .meta.json sidecar（ADR-0006 D5）
        for split in _SPLITS:
            if any(r.split == split for r in m.records):
                m.write_nanodet_labels(split, out / "labels" / f"{key}_{split}.json")
        summary[key] = m.counts_by_split()

    _write_license_manifest(manifests, out / "license_manifest.csv")
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[detect-build] {out} : {summary}")
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="组装检测训练/评估数据（各 adapter → manifest）")
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)
    build(DetectBuildConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
