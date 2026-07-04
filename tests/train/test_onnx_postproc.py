"""检测 ONNX 出 logits（[[ADR-0007]]）：剥 cls 头 Sigmoid、不误删 SE 块 Sigmoid、契约门防呆。"""

from __future__ import annotations

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402

from edge_cam.train.detect.onnx_postproc import (  # noqa: E402
    assert_cls_logits,
    strip_cls_sigmoid,
)


def _model_with_cls_and_se_sigmoid():
    """cls 头：Split→Sigmoid→Concat（该剥）；SE 块：Sigmoid→Mul（不该动）。"""
    nodes = [
        helper.make_node("Split", ["X"], ["cls", "reg"], axis=2),
        helper.make_node("Sigmoid", ["cls"], ["cls_sig"], name="cls_head_sigmoid"),
        helper.make_node("Concat", ["cls_sig", "reg"], ["out1"], axis=2),
        helper.make_node("Sigmoid", ["se_in"], ["se_sig"], name="se_sigmoid"),
        helper.make_node("Mul", ["se_sig", "X"], ["out2"]),
    ]
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 37])
    se_in = helper.make_tensor_value_info("se_in", TensorProto.FLOAT, [1, 4, 37])
    out1 = helper.make_tensor_value_info("out1", TensorProto.FLOAT, [1, 4, 37])
    out2 = helper.make_tensor_value_info("out2", TensorProto.FLOAT, [1, 4, 37])
    g = helper.make_graph(nodes, "t", [X, se_in], [out1, out2])
    return helper.make_model(g)


def test_assert_cls_logits_rejects_sigmoid_before_strip():
    """导 sigmoid'd 检测 ONNX → 契约门报错（防呆，同 FP32-only 门）。"""
    m = _model_with_cls_and_se_sigmoid()
    with pytest.raises(ValueError, match="ADR-0007"):
        assert_cls_logits(m)


def test_strip_removes_only_cls_head_sigmoid():
    """只剥 cls 头 Sigmoid（消费者=Concat）；SE 块 Sigmoid（消费者=Mul）保留。"""
    m = _model_with_cls_and_se_sigmoid()
    n = strip_cls_sigmoid(m)
    assert n == 1  # 只删 1 个（cls 头）
    ops = [nd.op_type for nd in m.graph.node]
    assert ops.count("Sigmoid") == 1  # SE 块那个还在
    # Concat 现在直接吃 Split 的原始 cls（logits），不再经 Sigmoid
    concat = next(nd for nd in m.graph.node if nd.op_type == "Concat")
    assert "cls" in concat.input and "cls_sig" not in concat.input
    # SE 块完好
    assert any(nd.op_type == "Mul" for nd in m.graph.node)


def test_assert_cls_logits_passes_after_strip():
    """剥完 → 契约门通过（cls 头无 Sigmoid）。"""
    m = _model_with_cls_and_se_sigmoid()
    strip_cls_sigmoid(m)
    assert_cls_logits(m)  # 不抛


def test_strip_idempotent():
    """再剥一次 = 0（幂等，无 cls 头 Sigmoid 可删）。"""
    m = _model_with_cls_and_se_sigmoid()
    assert strip_cls_sigmoid(m) == 1
    assert strip_cls_sigmoid(m) == 0


@pytest.mark.slow
def test_strip_real_feeder_onnx_to_logits():
    """真实 feeder ONNX 回归（round1 §7 喷框源）：剥 4 个 cls Sigmoid → cls 输出变 logits。"""
    from pathlib import Path

    import numpy as np

    np_ort = pytest.importorskip("onnxruntime")
    from edge_cam.train.detect.onnx_postproc import _cls_head_sigmoids

    root = Path(__file__).resolve().parents[2]
    p = root / "results/detect/feeder_416/weights/feeder_416_op13.onnx"
    if not p.exists():
        pytest.skip("feeder_416 ONNX 不在（DVC 未拉）")
    m = onnx.load(str(p))
    assert len(_cls_head_sigmoids(m.graph)) == 4  # 剥前：4 FPN level 各一
    assert strip_cls_sigmoid(m) == 4
    assert_cls_logits(m)  # 剥后干净
    out_p = root / "results/detect/feeder_416/weights/_logits_tmp.onnx"
    onnx.save(m, str(out_p))
    try:
        sess = np_ort.InferenceSession(str(out_p), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"data": np.zeros((1, 3, 416, 416), np.float32)})[0]
        cls = out[0, :, :5]
        assert cls.min() < 0.0  # logits 会有负值（sigmoid'd 时恒 ≥0）
    finally:
        out_p.unlink(missing_ok=True)
