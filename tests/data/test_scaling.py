"""数据量 scaling 子集：确定性抽 train、val/test 固定、嵌套（20%⊂50%⊂100%）。"""

from __future__ import annotations

import math

from edge_cam.contracts.schemas.detection_manifest import (
    FEEDER5_CATEGORIES,
    DetBox,
    DetectionManifest,
    DetImageRecord,
)
from edge_cam.data.scaling import subsample_train


def _rec(i, split):
    return DetImageRecord(
        path=f"{split}_{i}.jpg",
        split=split,
        width=100,
        height=100,
        boxes=[DetBox(bbox=[10, 10, 20, 20], category_id=FEEDER5_CATEGORIES["bird"])],
    )


def _mani(n_train=50, n_val=10, n_test=10):
    recs = [_rec(i, "train") for i in range(n_train)]
    recs += [_rec(i, "val") for i in range(n_val)]
    recs += [_rec(i, "test") for i in range(n_test)]
    return DetectionManifest(
        name="t", version="v0", categories=dict(FEEDER5_CATEGORIES), records=recs
    )


def _train_paths(m):
    return {r.path for r in m.records if r.split == "train"}


def test_subsample_keeps_val_test_full():
    """只抽 train，val/test 全保留（同一 held-out 才可比）。"""
    m = _mani()
    m2 = subsample_train(m, 0.2)
    assert sum(r.split == "val" for r in m2.records) == 10
    assert sum(r.split == "test" for r in m2.records) == 10
    assert sum(r.split == "train" for r in m2.records) == math.ceil(0.2 * 50)  # 10


def test_subsample_nested():
    """嵌套：20% ⊂ 50% ⊂ 100%（加数据是真加）。"""
    m = _mani()
    t20 = _train_paths(subsample_train(m, 0.2))
    t50 = _train_paths(subsample_train(m, 0.5))
    t100 = _train_paths(subsample_train(m, 1.0))
    assert t20 < t50 < t100  # 真子集
    assert t100 == _train_paths(m)  # 100% = 全量


def test_subsample_deterministic():
    """确定性：同 frac 两次结果一致（可复现）。"""
    m = _mani()
    assert _train_paths(subsample_train(m, 0.3)) == _train_paths(subsample_train(m, 0.3))


def test_subsample_frac_bounds():
    """frac≥1 原样；frac 非法抛。"""
    import pytest

    m = _mani()
    assert subsample_train(m, 1.0) is m
    with pytest.raises(ValueError, match="frac"):
        subsample_train(m, 0.0)
    with pytest.raises(ValueError, match="frac"):
        subsample_train(m, 1.5)
