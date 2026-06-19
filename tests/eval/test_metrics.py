"""评估指标：top-1/5、per-class、logit_transform 钩子。"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from edge_cam.eval.metrics import evaluate_torch


class _IdentityModel(torch.nn.Module):
    """直接把输入当 logits（输入即 (N, num_classes)）→ 便于精确断言。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _loader(logits: torch.Tensor, targets: torch.Tensor, bs: int = 2) -> DataLoader:
    return DataLoader(TensorDataset(logits, targets), batch_size=bs)


def test_perfect_and_partial() -> None:
    logits = torch.tensor([[2.0, 0, 0], [0, 2.0, 0], [2.0, 0, 0], [0, 0, 2.0]])
    targets = torch.tensor([0, 1, 1, 2])  # 第三个错（真值1，预测0）
    m = evaluate_torch(_IdentityModel(), _loader(logits, targets))
    assert m.n == 4
    assert m.top1 == 0.75
    assert m.top5 == 1.0  # 类数<5 → 全命中
    assert m.per_class_top1[1] == 0.5  # 类1：2 个里对 1 个


def test_logit_transform_applied() -> None:
    # mask 掉类0 → 原本预测类0 的样本被迫改判
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    targets = torch.tensor([1])

    def mask0(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[:, 0] = float("-inf")
        return out

    m = evaluate_torch(_IdentityModel(), _loader(logits, targets), logit_transform=mask0)
    assert m.top1 == 1.0  # mask 后改判类1，命中
