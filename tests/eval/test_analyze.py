"""深度分析：per-class top-1 + 混淆对（合成数据，不依赖训练/GPU）。"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from edge_cam.eval.analyze import deep_analyze, write_analysis


class _FixedLogits(torch.nn.Module):
    """把输入直接当 logits 返回（测控制可预测的预测结果）。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_deep_analyze_perclass_and_confusion() -> None:
    # 3 类。样本: 真值[0,0,1,2]; logits 让 pred=[0,1,1,2] → 类0 一对一错(误判为1), 类1/2 全对
    logits = torch.tensor([[9.0, 0, 0], [0, 9.0, 0], [0, 9.0, 0], [0, 0, 9.0]])
    targets = torch.tensor([0, 0, 1, 2])
    loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
    idx_to_class = {0: "sparrow", 1: "robin", 2: "crow"}
    da = deep_analyze(_FixedLogits(), loader, idx_to_class, device="cpu")

    assert da.n == 4
    assert da.top1 == 0.75  # 3/4 对
    assert da.per_class_top1["sparrow"] == 0.5  # 类0: 1对1错
    assert da.per_class_top1["robin"] == 1.0
    assert da.per_class_n["sparrow"] == 2
    # 混淆对: sparrow 被误判为 robin 一次
    assert ("sparrow", "robin", 1) in da.confused_pairs
    # 最差类: sparrow 排第一
    assert da.worst_classes(1)[0][0] == "sparrow"


def test_write_analysis(tmp_path: Path) -> None:
    logits = torch.tensor([[9.0, 0], [0, 9.0]])
    targets = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
    da = deep_analyze(_FixedLogits(), loader, {0: "a", 1: "b"}, device="cpu")
    jp, mp = write_analysis(da, tmp_path, model_name="m")
    assert jp.exists() and mp.exists()
    assert "深度分析" in mp.read_text(encoding="utf-8")
    assert "最差" in mp.read_text(encoding="utf-8")
