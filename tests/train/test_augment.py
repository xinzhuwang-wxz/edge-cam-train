"""退化/评估变换：输出形状、dtype、归一化一致。"""

from __future__ import annotations

import torch
from PIL import Image

from edge_cam.train.classify.augment import (
    build_eval_transform,
    build_field_transform,
    build_train_transform,
)


def _img() -> Image.Image:
    return Image.new("RGB", (300, 200), (90, 140, 70))


def test_train_transform_shape_dtype() -> None:
    out = build_train_transform(size=224)(_img())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_eval_transform_shape() -> None:
    out = build_eval_transform(size=192)(_img())
    assert out.shape == (3, 192, 192)


def test_field_transform_shape() -> None:
    out = build_field_transform(size=224)(_img())
    assert out.shape == (3, 224, 224)


def test_degradation_strength_zero_is_clean_geom() -> None:
    # strength=0 关闭退化概率，仍输出合法张量
    out = build_train_transform(size=128, degradation_strength=0.0)(_img())
    assert out.shape == (3, 128, 128)
