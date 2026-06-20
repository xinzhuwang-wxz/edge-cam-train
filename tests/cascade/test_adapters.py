"""级联真 adapter（#12）：decode_nanodet 数学(合成,严格)+ OnnxClassifier 真 onnx(slow)。"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from edge_cam.cascade.adapters import decode_nanodet


def test_decode_nanodet_known_box() -> None:
    """合成 NanoDet 输出 → 已知框,严格验证 priors/Integral/box/缩放/NMS。"""
    isz, strides, reg_max, ncls = 32, (8, 16, 32), 7, 2
    # N = 4²+2²+1² = 21;通道 = ncls + 4*(reg_max+1) = 2 + 32 = 34
    out = np.full((1, 21, 34), -10.0, np.float32)  # 全低分(sigmoid≈0,被 conf 过滤)
    # 目标 anchor:stride8、(col=1,row=1)→cx=cy=8,index=row*4+col=5
    i = 5
    out[0, i, 0] = 10.0  # class0 高分
    reg = out[0, i, 2:].reshape(4, 8)
    reg[:, 1] = 10.0  # 每边分布质量在 bin1 → Integral≈1 → 距离=1*stride=8
    dets = decode_nanodet(
        out,
        (32, 32),
        input_size=isz,
        strides=strides,
        reg_max=reg_max,
        num_classes=ncls,
        conf_thr=0.3,
        nms_iou=0.6,
    )
    assert len(dets) == 1, f"应只 1 检测,得 {len(dets)}"
    d = dets[0]
    assert d.class_id == 0 and d.score > 0.99
    # cx=cy=8, 各边距离=8 → box=[0,0,16,16](orig=input,缩放=1)
    assert d.box == pytest.approx((0.0, 0.0, 16.0, 16.0), abs=1e-3)


def test_decode_scales_to_original() -> None:
    """orig 尺寸≠input 时框按比例缩放。"""
    out = np.full((1, 21, 34), -10.0, np.float32)
    out[0, 5, 0] = 10.0
    out[0, 5, 2:].reshape(4, 8)[:, 1] = 10.0
    dets = decode_nanodet(
        out, (64, 64), input_size=32, strides=(8, 16, 32), num_classes=2, conf_thr=0.3
    )
    # input 上 [0,0,16,16] → orig 64/32=2x → [0,0,32,32]
    assert dets[0].box == pytest.approx((0.0, 0.0, 32.0, 32.0), abs=1e-3)


def test_prior_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="priors"):
        decode_nanodet(
            np.zeros((1, 99, 34), np.float32),
            (32, 32),
            input_size=32,
            strides=(8, 16, 32),
            num_classes=2,
        )


@pytest.mark.slow
def test_onnx_classifier_real(tmp_path) -> None:
    """临时导一个小分类 onnx → OnnxClassifier 真跑(真 adapter,非 fake)。"""
    from torch import nn

    from edge_cam.cascade.adapters import OnnxClassifier
    from edge_cam.onnx_artifact import export_onnx

    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 5))
    onnx = export_onnx(model, tmp_path / "clf.onnx", input_size=224, simplify=False)
    clf = OnnxClassifier(str(onnx), input_size=224)
    top1, conf, top5 = clf.classify(Image.new("RGB", (300, 200), (120, 130, 140)))
    assert 0 <= top1 < 5 and 0.0 <= conf <= 1.0 and len(top5) == 5
