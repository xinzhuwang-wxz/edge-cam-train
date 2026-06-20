"""数据准备编排：端到端从 ImageFolder 产出 manifest（CPU 本地可跑）。"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from edge_cam.data.prep import DataPrepConfig, SourceSpec, build_manifest, prepare

_COLOR = [0]


def _img(path: Path) -> None:
    """每张图唯一颜色 → 字节不同（避免内容去重误删测试样本）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    _COLOR[0] += 7
    c = _COLOR[0] % 256
    Image.new("RGB", (8, 8), (c, (c * 3) % 256, (c * 5) % 256)).save(path)


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


def test_root_xor_sources_validated(flat_imagefolder: Path) -> None:
    with pytest.raises(ValueError, match="root（单源）或 sources"):
        DataPrepConfig(name="t")  # 两者都没给
    with pytest.raises(ValueError, match="不能同时给"):
        DataPrepConfig(name="t", root=str(flat_imagefolder), sources=[SourceSpec(root="x")])


def test_multi_source_merges_by_ebird_key(tmp_path: Path) -> None:
    """两个源、不同俗名拼写、各自 eBird 表 → 按 taxon_key 合并成同一物种类。"""
    # 源 A：House Sparrow（5）+ American Robin（3）
    src_a = tmp_path / "dataset_a"
    for i in range(5):
        _img(src_a / "House Sparrow" / f"a{i}.jpg")
    for i in range(3):
        _img(src_a / "American Robin" / f"r{i}.jpg")
    csv_a = tmp_path / "a.csv"
    csv_a.write_text(
        "label,ebird_code\nHouse Sparrow,houspa\nAmerican Robin,amerob\n", encoding="utf-8"
    )
    # 源 B：拼写不同的同种 House_Sparrow（4）+ 未映射的 Dodo（2，应丢弃）
    src_b = tmp_path / "dataset_b"
    for i in range(4):
        _img(src_b / "House_Sparrow" / f"b{i}.jpg")
    for i in range(2):
        _img(src_b / "Dodo" / f"d{i}.jpg")
    csv_b = tmp_path / "b.csv"
    csv_b.write_text("label,ebird_code\nHouse_Sparrow,houspa\n", encoding="utf-8")

    cfg = DataPrepConfig(
        name="merged",
        sources=[
            SourceSpec(root=str(src_a), source="A", license="cc-by", taxonomy_csv=str(csv_a)),
            SourceSpec(root=str(src_b), source="B", license="cc0", taxonomy_csv=str(csv_b)),
        ],
        min_train_per_class=1,
    )
    m = build_manifest(cfg)
    # 两源的 House Sparrow（5+4）并为一类 houspa；amerob 一类；Dodo 丢弃
    assert set(m.class_to_idx) == {"houspa", "amerob"}
    houspa = [r for r in m.records if r.label == "houspa"]
    assert len(houspa) == 9  # 跨源合并
    assert {r.source for r in houspa} == {"A", "B"}  # 逐源 provenance 保留
    assert all(r.taxon_key in {"houspa", "amerob"} for r in m.records)
    assert m.root is None  # 多源 → 绝对路径


def test_multi_source_dedup_by_content(tmp_path: Path) -> None:
    """两源含字节相同的同一图 → 内容去重只留一份。"""
    a = tmp_path / "a" / "House Sparrow" / "x.jpg"
    b = tmp_path / "b" / "House_Sparrow" / "y.jpg"
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(a)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(b)  # 字节相同
    csv = tmp_path / "m.csv"
    csv.write_text(
        "label,ebird_code\nHouse Sparrow,houspa\nHouse_Sparrow,houspa\n", encoding="utf-8"
    )
    cfg = DataPrepConfig(
        name="d",
        sources=[
            SourceSpec(root=str(tmp_path / "a"), source="A", taxonomy_csv=str(csv)),
            SourceSpec(root=str(tmp_path / "b"), source="B", taxonomy_csv=str(csv)),
        ],
    )
    assert build_manifest(cfg).num_samples == 1  # 去重后只剩一份
