"""ImageFolder 读取（engineering §5.5 流水 2 的入口）。

支持两种布局：
- 扁平：root/<label>/<img>
- 带 split：root/<split_dir>/<label>/<img>（如 BIRDS-525 的 train/valid/test）
  → 合并所有 split，后续交 stratified_split 用固定 seed 重新划分（plan §B.1）。
"""

from __future__ import annotations

from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_imagefolder(
    root: str | Path,
    splits: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """扫描 ImageFolder，返回去重且确定性排序的 (path, label) 列表。

    Args:
        root: 数据集根目录。
        splits: 若给定（如 {"train": "train", "val": "valid", "test": "test"}），
            扫描各 split 子目录并合并；否则把 root 下的类目录当扁平 ImageFolder。
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"ingest: 目录不存在 {root}")
    subroots = [root / d for d in splits.values()] if splits else [root]

    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for sub in subroots:
        if not sub.is_dir():
            continue
        for class_dir in sorted(p for p in sub.iterdir() if p.is_dir()):
            label = class_dir.name
            for img in sorted(class_dir.iterdir()):
                if not (img.is_file() and img.suffix.lower() in IMG_EXTS):
                    continue
                path = str(img)
                if path not in seen:
                    seen.add(path)
                    items.append((path, label))
    return items
