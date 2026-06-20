"""分类评估指标：top-1/top-5/per-class（plan §B.1）。

evaluate_torch / evaluate_onnx 共用同一套统计；可选 logit_transform 用于地域 mask
（slice 4 消融），避免 metrics 与 regional 循环依赖。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

LogitTransform = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class EvalMetrics:
    top1: float
    top5: float
    n: int
    per_class_top1: dict[int, float] = field(default_factory=dict)


def topk_hits(
    logits: torch.Tensor, targets: torch.Tensor, ks: tuple[int, ...] = (1, 5)
) -> dict[int, int]:
    """{k: top-k 命中数}。单一 top-k 计数实现，训练(module)与评估共用（架构审查 D）。"""
    maxk = min(max(ks), logits.size(1))
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(targets.view(-1, 1))
    return {k: int(correct[:, : min(k, maxk)].any(dim=1).sum().item()) for k in ks}


def _batch_hits(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[int, int, dict[int, list[int]]]:
    """返回 (top1 命中, top5 命中, {label: [correct, total]})。"""
    hits = topk_hits(logits, targets, (1, 5))
    maxk = min(5, logits.size(1))
    _, pred = logits.topk(maxk, dim=1)
    hit1 = pred.eq(targets.view(-1, 1))[:, :1].any(dim=1)
    per_class: dict[int, list[int]] = {}
    for t, h in zip(targets.tolist(), hit1.tolist(), strict=True):
        acc = per_class.setdefault(t, [0, 0])
        acc[0] += int(h)
        acc[1] += 1
    return hits[1], hits[5], per_class


def _finalize(top1: int, top5: int, n: int, pc: dict[int, list[int]]) -> EvalMetrics:
    per_class = {k: c / t for k, (c, t) in pc.items() if t} if pc else {}
    return EvalMetrics(
        top1=top1 / n if n else 0.0, top5=top5 / n if n else 0.0, n=n, per_class_top1=per_class
    )


@torch.no_grad()
def evaluate_torch(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    logit_transform: LogitTransform | None = None,
) -> EvalMetrics:
    model = model.eval().to(device)
    top1 = top5 = n = 0
    pc: dict[int, list[int]] = {}
    for images, targets in loader:
        logits = model(images.to(device)).cpu()
        if logit_transform is not None:
            logits = logit_transform(logits)
        b1, b5, b_pc = _batch_hits(logits, targets)
        top1 += b1
        top5 += b5
        n += targets.size(0)
        for k, (c, t) in b_pc.items():
            acc = pc.setdefault(k, [0, 0])
            acc[0] += c
            acc[1] += t
    return _finalize(top1, top5, n, pc)


def evaluate_onnx(
    onnx_path: str,
    loader: DataLoader,
    logit_transform: LogitTransform | None = None,
) -> EvalMetrics:
    """用 onnxruntime 跑 ONNX（FP32 或量化后）评估，口径与 torch 版一致。

    ONNX 为**静态 batch=1**（NPU 单帧）→ 逐样本喂入，与端侧推理一致。"""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    top1 = top5 = n = 0
    pc: dict[int, list[int]] = {}
    for images, targets in loader:
        arr = images.numpy().astype(np.float32)
        per_sample = [
            np.asarray(sess.run(None, {input_name: arr[i : i + 1]})[0]) for i in range(arr.shape[0])
        ]
        logits = torch.from_numpy(np.concatenate(per_sample, axis=0))
        if logit_transform is not None:
            logits = logit_transform(logits)
        b1, b5, b_pc = _batch_hits(logits, targets)
        top1 += b1
        top5 += b5
        n += targets.size(0)
        for k, (c, t) in b_pc.items():
            acc = pc.setdefault(k, [0, 0])
            acc[0] += c
            acc[1] += t
    return _finalize(top1, top5, n, pc)
