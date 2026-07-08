"""检测判别 margin（round3 §6）：量化"触发后置信度拉不拉得开"（用户核心诉求）。

每个检测的 **top1 − top2 类分 = margin**：大 = 类判别清晰；小 = "几个 label 置信度都差不多"
（§2 sigmoid 无跨类竞争的症状）。margin 直方图右移 / frac_below 下降 = round3 把类拉开了。

⚠️ 需**全类分向量**（sigmoid 后、argmax 前）。NanoDet NMS 后只留最终类+分 → 须 GPU 定制推理
dump 预 NMS 全类分、匹配回 NMS 存活框（verify G7，~0.5-1 天）。**本模块 = 纯计算，可无卡测**。
"""

from __future__ import annotations

from dataclasses import dataclass


def top1_top2(scores: list[float]) -> tuple[int, float]:
    """全类分向量 → (top1 类 idx, top1−top2 margin)。单类时 margin=该分。"""
    if not scores:
        raise ValueError("scores 不能为空")
    top1_idx = max(range(len(scores)), key=lambda i: scores[i])
    ordered = sorted(scores, reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    return top1_idx, margin


@dataclass
class MarginStats:
    """一批检测的 margin 统计。margins/top1[i] 一一对应。"""

    margins: list[float]
    top1: list[int]
    names: list[str]

    def median(self) -> float:
        if not self.margins:
            return 0.0
        s = sorted(self.margins)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    def frac_below(self, thr: float) -> float:
        """margin < thr 的占比 = "都差不多"的比例（越小越好）。"""
        return sum(m < thr for m in self.margins) / len(self.margins) if self.margins else 0.0

    def per_class_median(self) -> dict[str, float]:
        """按 top1 类分组的 margin 中位数（看 squirrel/cat 判得清不清）。"""
        out: dict[str, float] = {}
        for ci, name in enumerate(self.names):
            ms = sorted(m for m, t in zip(self.margins, self.top1, strict=True) if t == ci)
            if ms:
                n = len(ms)
                out[name] = ms[n // 2] if n % 2 else (ms[n // 2 - 1] + ms[n // 2]) / 2
        return out


def margin_stats(
    detections: list[dict],
    names: list[str],
    *,
    min_top1: float = 0.4,
) -> MarginStats:
    """detections: [{"class_scores":[...]}] → MarginStats。只统计 top1 ≥ min_top1（触发的）。"""
    margins: list[float] = []
    top1: list[int] = []
    for d in detections:
        scores = d["class_scores"]
        idx, margin = top1_top2(scores)
        if scores[idx] < min_top1:
            continue
        margins.append(margin)
        top1.append(idx)
    return MarginStats(margins=margins, top1=top1, names=names)
