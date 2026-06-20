"""Classifier：top-k 计数、forward 形状、训练步产出 loss。"""

from __future__ import annotations

import pytest
import torch

from edge_cam.eval.metrics import topk_hits
from edge_cam.train.classify.module import Classifier


def test_topk_correct_logic() -> None:
    # 2 个样本，5 类
    logits = torch.tensor([[0.1, 0.9, 0.2, 0.0, 0.3], [0.5, 0.1, 0.2, 0.9, 0.0]])
    target = torch.tensor([1, 0])  # 样本0 top-1 命中(1)；样本1 真值0 排第2
    hits = topk_hits(logits, target, (1, 5))
    assert hits[1] == 1  # 仅样本0 top-1 命中
    assert hits[5] == 2  # top-5 都命中


def test_topk_caps_at_num_classes() -> None:
    logits = torch.randn(3, 4)
    hits = topk_hits(logits, torch.tensor([0, 1, 2]), (1, 5))
    assert 0 <= hits[1] <= 3
    assert hits[5] == 3  # 类数<5 → top-5 等于全命中


@pytest.mark.slow
def test_forward_shape() -> None:
    model = Classifier(model_name="efficientnet_lite0", num_classes=7, pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 7)


@pytest.mark.slow
def test_training_step_returns_loss() -> None:
    model = Classifier(model_name="efficientnet_lite0", num_classes=7, pretrained=False)
    batch = (torch.randn(2, 3, 224, 224), torch.tensor([0, 3]))
    loss = model.training_step(batch, 0)
    assert loss.requires_grad
    assert loss.ndim == 0
