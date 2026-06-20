"""固定 seed 的分层 split（plan §B.1 / engineering §5.5）。

纯函数、确定性、与输入顺序无关：同一 (seed, items) 永远产出同一划分，便于实验
可复现。处理长尾小类——每类至少保证 min_train_per_class 进 train（BIRDS-525 每类
仅 ~26 张，缺类是常态）。"""

from __future__ import annotations

import random
from collections import defaultdict

from edge_cam.contracts.schemas.dataset import Split


def stratified_split(
    items: list[tuple[str, str]],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
    min_train_per_class: int = 1,
) -> dict[str, Split]:
    """按类分层把样本划到 train/val/test。

    Args:
        items: (key, label) 列表；key 必须唯一（通常是图片路径）。
        ratios: (train, val, test) 比例，自动归一化。
        seed: 随机种子，决定确定性划分。
        min_train_per_class: 每个类至少保证多少样本进 train（小类兜底）。

    Returns:
        {key: split} 映射，覆盖全部 items，划分互斥且完备。
    """
    if not items:
        return {}
    keys = [k for k, _ in items]
    if len(set(keys)) != len(keys):
        raise ValueError("stratified_split: items 的 key 必须唯一")
    rt, rv, rs = ratios
    total = rt + rv + rs
    if total <= 0:
        raise ValueError("stratified_split: ratios 之和必须 > 0")
    rt, rv = rt / total, rv / total

    by_label: dict[str, list[str]] = defaultdict(list)
    for key, label in items:
        by_label[label].append(key)

    assignment: dict[str, Split] = {}
    for label in sorted(by_label):
        ks = sorted(by_label[label])  # 排序 → 与输入顺序无关
        random.Random(f"{seed}:{label}").shuffle(ks)  # 确定性（version-2 seeding，跨进程稳定）
        n = len(ks)
        n_train = min(max(min(min_train_per_class, n), round(n * rt)), n)
        remaining = n - n_train
        n_val = min(round(n * rv), remaining)
        order: list[Split] = ["train", "val", "test"]
        counts = [n_train, n_val, remaining - n_val]
        splits: list[Split] = [s for s, c in zip(order, counts, strict=True) for _ in range(c)]
        for key, split in zip(ks, splits, strict=True):
            assignment[key] = split
    return assignment
