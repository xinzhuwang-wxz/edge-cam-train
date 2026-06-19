"""数据准备编排：端到端从 ImageFolder 产出 manifest（CPU 本地可跑）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from edge_cam.data.prep import DataPrepConfig, build_manifest, prepare


def test_build_manifest_flat(flat_imagefolder: Path) -> None:
    config = DataPrepConfig(name="t", root=str(flat_imagefolder), seed=0)
    m = build_manifest(config)
    assert m.num_classes == 3
    assert m.num_samples == 9
    # 每类至少 1 张进 train（含单样本 tiny 类）
    assert m.class_counts(split="train").get("tiny", 0) >= 1


def test_taxon_key_populated(flat_imagefolder: Path) -> None:
    config = DataPrepConfig(name="t", root=str(flat_imagefolder))
    m = build_manifest(config)
    assert all(r.taxon_key for r in m.records)


def test_license_source_propagated(flat_imagefolder: Path) -> None:
    config = DataPrepConfig(
        name="t", root=str(flat_imagefolder), source="kaggle", license="feasibility-only"
    )
    m = build_manifest(config)
    assert {r.license for r in m.records} == {"feasibility-only"}
    assert {r.source for r in m.records} == {"kaggle"}


def test_split_layout(split_imagefolder: Path) -> None:
    config = DataPrepConfig(
        name="t",
        root=str(split_imagefolder),
        splits={"train": "train", "val": "valid", "test": "test"},
    )
    m = build_manifest(config)
    assert m.num_classes == 2
    assert m.num_samples == 11


def test_prepare_saves_and_reloads(flat_imagefolder: Path, tmp_path: Path) -> None:
    out = tmp_path / "manifest.json"
    config = DataPrepConfig(name="t", root=str(flat_imagefolder), out_path=str(out))
    m = prepare(config)
    assert out.exists()
    from edge_cam.contracts.schemas.dataset import DatasetManifest

    assert DatasetManifest.load(out) == m


def test_empty_root_raises(tmp_path: Path) -> None:
    (tmp_path / "emptyclass").mkdir()
    config = DataPrepConfig(name="t", root=str(tmp_path))
    with pytest.raises(ValueError, match="未扫到"):
        build_manifest(config)


def test_config_from_yaml(tmp_path: Path, flat_imagefolder: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        f'name: t\nroot: "{flat_imagefolder}"\nseed: 0\nratios: [0.7, 0.15, 0.15]\n',
        encoding="utf-8",
    )
    config = DataPrepConfig.from_yaml(yaml_path)
    assert config.name == "t"
    assert build_manifest(config).num_samples == 9
