"""scan_imagefolder：扁平 / 带 split 布局、扩展名过滤、确定性、去重。"""

from __future__ import annotations

from pathlib import Path

import pytest

from edge_cam.data.ingest import scan_imagefolder


def test_flat_layout(flat_imagefolder: Path) -> None:
    items = scan_imagefolder(flat_imagefolder)
    labels = sorted({lbl for _, lbl in items})
    assert labels == ["class_a", "class_b", "tiny"]
    assert len(items) == 9  # 5 + 3 + 1


def test_split_layout_merges(split_imagefolder: Path) -> None:
    items = scan_imagefolder(
        split_imagefolder, splits={"train": "train", "val": "valid", "test": "test"}
    )
    # 4+3 (train) + 2+1 (valid) + 1 (test) = 11
    assert len(items) == 11
    assert {lbl for _, lbl in items} == {"a", "b"}


def test_deterministic_order(flat_imagefolder: Path) -> None:
    assert scan_imagefolder(flat_imagefolder) == scan_imagefolder(flat_imagefolder)


def test_non_image_ignored(flat_imagefolder: Path) -> None:
    (flat_imagefolder / "class_a" / "notes.txt").write_text("x")
    items = scan_imagefolder(flat_imagefolder)
    assert all(not p.endswith(".txt") for p, _ in items)


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        scan_imagefolder(tmp_path / "does-not-exist")
