"""DatasetManifest 契约：完整性校验、计数、存盘/加载往返。"""

from __future__ import annotations

from pathlib import Path

import pytest

from edge_cam.contracts.schemas.dataset import DatasetManifest, SampleRecord


def _manifest() -> DatasetManifest:
    records = [
        SampleRecord(path="a/0.jpg", label="a", split="train"),
        SampleRecord(path="a/1.jpg", label="a", split="val"),
        SampleRecord(path="b/0.jpg", label="b", split="train"),
        SampleRecord(path="b/1.jpg", label="b", split="test"),
    ]
    return DatasetManifest(
        name="t", version="v0", seed=0, class_to_idx={"a": 0, "b": 1}, records=records
    )


def test_counts() -> None:
    m = _manifest()
    assert m.num_classes == 2
    assert m.num_samples == 4
    assert m.counts_by_split() == {"train": 2, "val": 1, "test": 1}
    assert m.class_counts() == {"a": 2, "b": 2}
    assert m.class_counts(split="train") == {"a": 1, "b": 1}


def test_non_contiguous_index_rejected() -> None:
    with pytest.raises(ValueError, match="连续整数"):
        DatasetManifest(name="t", version="v0", seed=0, class_to_idx={"a": 0, "b": 2}, records=[])


def test_unknown_label_rejected() -> None:
    with pytest.raises(ValueError, match="未在 class_to_idx"):
        DatasetManifest(
            name="t",
            version="v0",
            seed=0,
            class_to_idx={"a": 0},
            records=[SampleRecord(path="x.jpg", label="ghost", split="train")],
        )


def test_invalid_split_rejected() -> None:
    with pytest.raises(ValueError):
        SampleRecord(path="x.jpg", label="a", split="trainn")  # type: ignore[arg-type]


def test_save_load_roundtrip(tmp_path: Path) -> None:
    m = _manifest()
    out = tmp_path / "manifest.json"
    m.save(out)
    loaded = DatasetManifest.load(out)
    assert loaded == m
