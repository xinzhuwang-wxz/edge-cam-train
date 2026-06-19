"""消融网格展开（plan §B.0：单变量受控笛卡尔网格）。

把 {key: [候选...]} 展成一串 override dict。key 用 Hydra 点路径（如 'model.name'、
'data.input_size'），便于直接 merge 进 config。纯函数、确定性（保序）。"""

from __future__ import annotations

from itertools import product


def expand_grid(spec: dict[str, list]) -> list[dict]:
    """{key: [v1, v2], ...} → [{key: v1,...}, {key: v2,...}, ...]（笛卡尔积，保序）。

    空 spec → [{}]（单个空 override，即只跑基线）。
    """
    if not spec:
        return [{}]
    keys = list(spec)
    combos = product(*(spec[k] for k in keys))
    return [dict(zip(keys, values, strict=True)) for values in combos]


def label_for(overrides: dict) -> str:
    """给一个网格单元生成可读 label（用于 run 命名 / 表格行）。"""
    if not overrides:
        return "baseline"
    return " ".join(f"{k.split('.')[-1]}={v}" for k, v in overrides.items())
