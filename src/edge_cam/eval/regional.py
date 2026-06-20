"""地域过滤（plan §5.4：推理期 eBird likely-species mask，最大精度杠杆）。

训练用全局头；评估/推理时按地区把候选从全局 N 缩到区域 n —— 不在清单的类 logit
置 -inf。这里做**消融**：对比 mask on/off 的 top-1/5，量化「地域过滤值多少分」。

mask 以 taxon_key 表达（与训练解耦、可 OTA）→ 经 manifest 的 label→idx 落到 logit 列。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

NEG_INF = float("-inf")


class RegionalMask:
    """一组「在场」类的索引集合，可作用于 logits。"""

    def __init__(self, allowed_idx: set[int], num_classes: int) -> None:
        if not allowed_idx:
            raise ValueError("RegionalMask: 允许类集合不能为空")
        if max(allowed_idx) >= num_classes or min(allowed_idx) < 0:
            raise ValueError("RegionalMask: 索引越界")
        self.allowed_idx = set(allowed_idx)
        self.num_classes = num_classes

    @classmethod
    def from_taxon_keys(
        cls,
        allowed_keys: set[str],
        class_to_idx: dict[str, int],
        taxon_of: dict[str, str],
    ) -> RegionalMask:
        """由地域 taxon_key 清单 + manifest 映射构建。

        **契约（显式校验，非仅注释）**：allowed_keys 与 taxon_of 的值必须是**同一套规范键**
        （应为 eBird/Clements key）。若交集为空，几乎总是因为 taxonomy 仍是占位
        （IdentityTaxonomy 产小写俗名，对不上 eBird 清单）—— 报清楚而非抛「集合为空」。

        Args:
            allowed_keys: 区域「在场」物种的 taxon_key 集合。
            class_to_idx: manifest 的 label→idx。
            taxon_of: label→taxon_key（与 manifest 同源，防 index 漂移）。
        """
        manifest_keys = set(taxon_of.values())
        if not (allowed_keys & manifest_keys):
            raise ValueError(
                f"RegionalMask: 区域清单({len(allowed_keys)} keys)与 manifest taxon_key"
                f"({len(manifest_keys)} keys)交集为 0。检查 taxonomy 是否已解析到 eBird 规范键"
                "（IdentityTaxonomy 占位产小写俗名，对不上 eBird 清单）。"
            )
        # label 必在 class_to_idx（taxon_of 与 class_to_idx 同源 manifest）；防御性 guard
        idx = {
            class_to_idx[label]
            for label, key in taxon_of.items()
            if key in allowed_keys and label in class_to_idx
        }
        return cls(idx, num_classes=len(class_to_idx))

    @classmethod
    def from_json(
        cls, path: str | Path, class_to_idx: dict[str, int], taxon_of: dict[str, str]
    ) -> RegionalMask:
        """从 json（taxon_key 字符串数组）加载区域清单。"""
        keys = set(json.loads(Path(path).read_text(encoding="utf-8")))
        return cls.from_taxon_keys(keys, class_to_idx, taxon_of)

    def as_transform(self):
        """返回作用于 logits 的 callable（disallowed 列置 -inf）。"""
        keep = torch.zeros(self.num_classes, dtype=torch.bool)
        keep[list(self.allowed_idx)] = True

        def _mask(logits: torch.Tensor) -> torch.Tensor:
            out = logits.clone()
            out[:, ~keep] = NEG_INF
            return out

        return _mask

    @property
    def coverage(self) -> float:
        """区域类数 / 全局类数（候选缩小倍数的倒数）。"""
        return len(self.allowed_idx) / self.num_classes


@torch.no_grad()
def evaluate_regional(model, loader, mask: RegionalMask, device: str = "cpu") -> dict:
    """**只在 in-region 子集(真值∈区域)** 上比 mask on/off → 真实地域增益(修 issue#11)。

    旧口径在全局 test 上加 mask,把外地真值压成 -inf 必错 → artifact(非真增益)。正确口径:
    部署场景是"鸟确实是本地种",故只看 in-region 样本的 top-1 off vs on。
    返回 {in_region_n, top1_off, top1_on, gain}。"""
    model = model.eval().to(device)
    transform = mask.as_transform()
    n = off = on = 0
    for images, targets in loader:
        logits = model(images.to(device)).cpu()
        masked = transform(logits)
        for i, gt in enumerate(targets.tolist()):
            if gt not in mask.allowed_idx:
                continue
            n += 1
            off += int(logits[i].argmax().item() == gt)
            on += int(masked[i].argmax().item() == gt)
    return {
        "in_region_n": n,
        "top1_off": off / n if n else 0.0,
        "top1_on": on / n if n else 0.0,
        "gain": (on - off) / n if n else 0.0,
    }
