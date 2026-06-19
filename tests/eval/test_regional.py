"""地域 mask：构建、置 -inf、覆盖率、消融效果。"""

from __future__ import annotations

import pytest
import torch

from edge_cam.eval.regional import RegionalMask


def test_from_taxon_keys() -> None:
    class_to_idx = {"sparrow": 0, "robin": 1, "penguin": 2}
    taxon_of = {"sparrow": "k_sparrow", "robin": "k_robin", "penguin": "k_penguin"}
    mask = RegionalMask.from_taxon_keys({"k_sparrow", "k_robin"}, class_to_idx, taxon_of)
    assert mask.allowed_idx == {0, 1}
    assert mask.coverage == pytest.approx(2 / 3)


def test_transform_masks_disallowed() -> None:
    mask = RegionalMask({0, 2}, num_classes=3)
    out = mask.as_transform()(torch.tensor([[1.0, 5.0, 1.0]]))
    assert out[0, 1] == float("-inf")  # 类1 被屏蔽
    assert out[0, 0] == 1.0 and out[0, 2] == 1.0


def test_regional_improves_topk() -> None:
    # 真值类2，但全局里类1 logit 最高（不在区域）→ mask 后类2 胜出
    mask = RegionalMask({0, 2}, num_classes=3)
    logits = torch.tensor([[0.1, 9.0, 0.5]])
    assert logits.argmax().item() == 1  # 全局错
    masked = mask.as_transform()(logits)
    assert masked.argmax().item() == 2  # 区域内对


def test_empty_and_oob_rejected() -> None:
    with pytest.raises(ValueError):
        RegionalMask(set(), num_classes=3)
    with pytest.raises(ValueError):
        RegionalMask({5}, num_classes=3)


def test_key_mismatch_raises_diagnostic() -> None:
    """区域清单与 manifest taxon_key 不同套（如 eBird vs 占位小写名）→ 报清楚的契约错。"""
    class_to_idx = {"sparrow": 0, "robin": 1}
    taxon_of = {"sparrow": "sparrow", "robin": "robin"}  # IdentityTaxonomy 占位风格
    with pytest.raises(ValueError, match="交集为 0"):
        # 区域清单用 eBird code，对不上占位键
        RegionalMask.from_taxon_keys({"houspa", "amerob"}, class_to_idx, taxon_of)
