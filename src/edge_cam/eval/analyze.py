"""选定模型的深度指标分析（plan §B.1：per-class top-1 + 混淆对「易混种」）。

可行性包络（envelope）给的是逐级 top-1/5 汇总；本模块对**选定的最优模型**做细粒度剖析：
每类准确率（揪出最差的种）+ 最易混淆的 (真值→预测) 对。供选型后的深度分析与改进定向。

clean test 上跑（FP32），device 自动选 GPU。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class DeepAnalysis:
    n: int
    top1: float
    top5: float
    per_class_top1: dict[str, float] = field(default_factory=dict)  # class_name → acc
    per_class_n: dict[str, int] = field(default_factory=dict)
    confused_pairs: list[tuple[str, str, int]] = field(default_factory=list)  # (真值, 预测, 次数)

    def worst_classes(self, k: int = 20) -> list[tuple[str, float, int]]:
        """最差 k 个类：(类名, top-1, 样本数)，按 top-1 升序、样本数降序。"""
        items = [(c, self.per_class_top1[c], self.per_class_n[c]) for c in self.per_class_top1]
        return sorted(items, key=lambda x: (x[1], -x[2]))[:k]


@torch.no_grad()
def deep_analyze(
    model: nn.Module,
    loader: DataLoader,
    idx_to_class: dict[int, str],
    *,
    device: str | None = None,
    top_pairs: int = 30,
) -> DeepAnalysis:
    """跑一遍 clean test，统计 top-1/5、per-class 准确率、最易混淆对。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
    top1 = top5 = n = 0
    per_class: dict[int, list[int]] = {}  # idx → [correct, total]
    confusion: Counter[tuple[int, int]] = Counter()  # (true_idx, pred_idx) → count（仅错分）

    for images, targets in loader:
        logits = model(images.to(device)).cpu()
        maxk = min(5, logits.size(1))
        _, pred = logits.topk(maxk, dim=1)
        t = targets.view(-1, 1)
        c1 = pred[:, :1].eq(t).any(dim=1)
        c5 = pred[:, :maxk].eq(t).any(dim=1)
        top1 += int(c1.sum())
        top5 += int(c5.sum())
        n += targets.size(0)
        for tgt, p1, hit in zip(targets.tolist(), pred[:, 0].tolist(), c1.tolist(), strict=True):
            acc = per_class.setdefault(tgt, [0, 0])
            acc[0] += int(hit)
            acc[1] += 1
            if not hit:
                confusion[(tgt, p1)] += 1

    per_class_top1 = {idx_to_class[i]: c / t for i, (c, t) in per_class.items() if t}
    per_class_n = {idx_to_class[i]: t for i, (_, t) in per_class.items()}
    pairs = [
        (idx_to_class[ti], idx_to_class[pi], cnt)
        for (ti, pi), cnt in confusion.most_common(top_pairs)
    ]
    return DeepAnalysis(
        n=n,
        top1=top1 / n if n else 0.0,
        top5=top5 / n if n else 0.0,
        per_class_top1=per_class_top1,
        per_class_n=per_class_n,
        confused_pairs=pairs,
    )


def write_analysis(
    da: DeepAnalysis, out_dir: str | Path, *, model_name: str = ""
) -> tuple[Path, Path]:
    """落盘 analysis.json + analysis.md（最差类表 + 易混淆对表）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "analysis.json"
    json_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "n": da.n,
                "top1": round(da.top1, 4),
                "top5": round(da.top5, 4),
                "per_class_top1": {k: round(v, 4) for k, v in da.per_class_top1.items()},
                "per_class_n": da.per_class_n,
                "confused_pairs": [[t, p, c] for t, p, c in da.confused_pairs],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        f"### 深度分析 · {model_name}（clean test, n={da.n}）",
        f"top-1 **{da.top1:.3f}** · top-5 **{da.top5:.3f}** · 类数 {len(da.per_class_top1)}",
        "",
        "#### 最差 20 类（定向改进/加数据）",
        "| 物种 | top-1 | n |",
        "|---|---|---|",
    ]
    for name, acc, cnt in da.worst_classes(20):
        lines.append(f"| {name} | {acc:.2f} | {cnt} |")
    lines += ["", "#### 最易混淆 (真值 → 误判为) Top", "| 真值 | 误判为 | 次数 |", "|---|---|---|"]
    for true_c, pred_c, cnt in da.confused_pairs:
        lines.append(f"| {true_c} | {pred_c} | {cnt} |")
    md_path = out_dir / "analysis.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
