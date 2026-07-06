"""层级可用率（Hierarchical Usability）——分类段命门指标（docs/classify/06 §A）。

产品语义（对齐 iNat roll-up / Seek rank-ladder / plan-v2 分级门控）：
每样本输出模型敢确定的**最细层级**（种→属→科→bird），沿层级上滚 + 置信门：
  - 种级置信过门 & 种对        → 可用（报种正确）
  - 种级置信过门 & 种错        → **不可用 + critical_error（自信报错种，产品最伤）**
  - 种级不过门 → 回退：属聚合概率过门 & 属对 → 可用（报属）；否则回退科；
    最终回退 bird（级联已确认是鸟）→ 总可用。
命门 = usable_rate；critical_error（自信错种率）单列，是重罚项。

层级来自 taxonomy registry（ebird_code→genus/family，见 [[taxonomy-kb-cross-session]]）；
本模块只做度量，层级数据由 Hierarchy 传入（测试用小 fixture，真实数据是 R1.1 数据步）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Hierarchy:
    """class index → 祖先键。genus[i]/family[i] 是第 i 类的属/科键（种即 index 本身）。"""

    genus: list[str]
    family: list[str]

    def __post_init__(self) -> None:
        if len(self.genus) != len(self.family):
            raise ValueError("Hierarchy: genus/family 长度须一致（每类一条祖先链）")

    @property
    def num_classes(self) -> int:
        return len(self.genus)


@dataclass
class HierUsabilityMetrics:
    usable_rate: float  # 命门：可用样本占比
    n: int
    species_report: int  # 报到种级的样本数
    species_correct: int  # 报种且正确
    critical_error: int  # 自信报错种（重罚项）
    report_genus: int  # 回退到属级
    report_family: int  # 回退到科级
    report_bird: int  # 回退到 bird（兜底，总可用）


def _aggregate(probs: torch.Tensor, keys: list[str]) -> tuple[torch.Tensor, list[str]]:
    """把种级概率按祖先键聚合（sum）。probs (N,C) → (N,U)，返回 (聚合概率, 有序唯一键)。"""
    uniq = sorted(set(keys))
    idx = {k: i for i, k in enumerate(uniq)}
    m = torch.zeros(probs.size(1), len(uniq), dtype=probs.dtype)
    for c, k in enumerate(keys):
        m[c, idx[k]] = 1.0
    return probs @ m, uniq


def hierarchical_usability(
    logits: torch.Tensor,
    targets: torch.Tensor,
    hier: Hierarchy,
    tau_species: float = 0.5,
    tau_genus: float = 0.5,
    tau_family: float = 0.5,
) -> HierUsabilityMetrics:
    """计算层级可用率（命门）。tau_* 为各层置信门（v0 单值，后续可 per-species/per-region）。"""
    if logits.size(1) != hier.num_classes:
        raise ValueError(f"logits 类数 {logits.size(1)} != hierarchy {hier.num_classes}")
    probs = logits.softmax(dim=1)
    n = probs.size(0)
    genus_p, genus_u = _aggregate(probs, hier.genus)
    family_p, family_u = _aggregate(probs, hier.family)
    tgt_genus = [hier.genus[t] for t in targets.tolist()]
    tgt_family = [hier.family[t] for t in targets.tolist()]
    sp_conf, sp_idx = probs.max(dim=1)

    c = dict(
        usable=0,
        species_report=0,
        species_correct=0,
        critical_error=0,
        report_genus=0,
        report_family=0,
        report_bird=0,
    )
    for i in range(n):
        t = targets[i].item()
        if sp_conf[i].item() >= tau_species:
            c["species_report"] += 1
            if sp_idx[i].item() == t:
                c["species_correct"] += 1
                c["usable"] += 1
            else:
                c["critical_error"] += 1  # 自信报错种
            continue
        gp, gi = genus_p[i].max(dim=0)
        if gp.item() >= tau_genus:
            c["report_genus"] += 1
            if genus_u[int(gi.item())] == tgt_genus[i]:
                c["usable"] += 1
            continue
        fp, fi = family_p[i].max(dim=0)
        if fp.item() >= tau_family:
            c["report_family"] += 1
            if family_u[int(fi.item())] == tgt_family[i]:
                c["usable"] += 1
            continue
        c["report_bird"] += 1  # 兜底：级联已确认是鸟，回退 bird 总可用
        c["usable"] += 1

    return HierUsabilityMetrics(
        usable_rate=c["usable"] / n if n else 0.0,
        n=n,
        species_report=c["species_report"],
        species_correct=c["species_correct"],
        critical_error=c["critical_error"],
        report_genus=c["report_genus"],
        report_family=c["report_family"],
        report_bird=c["report_bird"],
    )
