"""共享 fixtures：构造临时 ImageFolder，免依赖真实数据集。"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


def make_image(path: Path, color: tuple[int, int, int] = (120, 120, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


@pytest.fixture
def flat_imagefolder(tmp_path: Path) -> Path:
    """扁平 ImageFolder：class_a(5) / class_b(3) / tiny(1)。"""
    for i in range(5):
        make_image(tmp_path / "class_a" / f"a{i}.jpg")
    for i in range(3):
        make_image(tmp_path / "class_b" / f"b{i}.png")
    make_image(tmp_path / "tiny" / "t0.jpg")
    return tmp_path


@pytest.fixture
def split_imagefolder(tmp_path: Path) -> Path:
    """带 split 的 ImageFolder：train/valid/test → 类目录。"""
    layout = {"train": {"a": 4, "b": 3}, "valid": {"a": 2, "b": 1}, "test": {"a": 1}}
    for split, classes in layout.items():
        for cls, n in classes.items():
            for i in range(n):
                make_image(tmp_path / split / cls / f"{split}_{cls}_{i}.jpg")
    return tmp_path
