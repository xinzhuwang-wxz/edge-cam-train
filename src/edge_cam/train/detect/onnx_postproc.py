"""检测 ONNX 导出后处理（[[ADR-0007]]）：剥 cls 头 Sigmoid → ONNX 出 **logits**，sigmoid 留 A7 CPU。

背景（round1 §7）：NanoDet `export_onnx.py` 把 sigmoid 烤进图（cls 输出 [0,1]），而 `decode_nanodet`
假设 logits、又 sigmoid 一次 → 双重 → 背景 0→0.5 也过阈值 → 喷 ~1300 框/图、查准≈0。ADR-0007 定：
ONNX 只出 logits，sigmoid+decode+NMS 全留 CPU。本模块 = 导出后**图手术**（不 fork §3）+ 契约门。

- `strip_cls_sigmoid(model)`：删 **cls 头模式**的 Sigmoid（`Split→Sigmoid→Concat`，消费者是 Concat；
  SE 块 sigmoid 消费者是 Mul，不误删）。返回删除个数。
- `assert_cls_logits(model)`：契约门——图里仍有 cls 头 Sigmoid（消费者含 Concat）则报错，防呆导出
  sigmoid'd 检测 ONNX（同 FP32-only 门思路）。
"""

from __future__ import annotations

from pathlib import Path


def _cls_head_sigmoids(graph) -> list:
    """cls 头 Sigmoid = 输出被 Concat 消费的 Sigmoid（区别于 SE 块 sigmoid→Mul）。"""
    out = []
    for node in graph.node:
        if node.op_type != "Sigmoid":
            continue
        consumers = [n for n in graph.node if node.output[0] in n.input]
        if consumers and all(c.op_type == "Concat" for c in consumers):
            out.append(node)
    return out


def strip_cls_sigmoid(model) -> int:
    """把 cls 头 Sigmoid 旁路删除（Concat 改吃 Sigmoid 的输入=原始 logits）。返回删除个数。"""
    graph = model.graph
    targets = _cls_head_sigmoids(graph)
    for s in targets:
        s_in, s_out = s.input[0], s.output[0]
        for n in graph.node:  # 把所有消费 s_out 的输入改成 s_in
            n.input[:] = [s_in if i == s_out else i for i in n.input]
        graph.node.remove(s)
    return len(targets)


def assert_cls_logits(model) -> None:
    """契约门：检测 ONNX 的 cls 头**不得含 Sigmoid**（应出 logits，sigmoid 留 CPU，ADR-0007）。"""
    remain = _cls_head_sigmoids(model.graph)
    if remain:
        raise ValueError(
            f"检测 ONNX 违反 ADR-0007：cls 头仍有 {len(remain)} 个 Sigmoid（应出 logits）。"
            f"导出后需 strip_cls_sigmoid。节点：{[s.name or s.output[0] for s in remain]}"
        )


def strip_and_verify(onnx_path: str | Path) -> int:
    """就地把检测 ONNX 的 cls Sigmoid 剥掉 + checker + 契约门。返回删除个数。导出路调用。"""
    import onnx

    p = Path(onnx_path)
    model = onnx.load(str(p))
    n = strip_cls_sigmoid(model)
    onnx.checker.check_model(model)
    assert_cls_logits(model)  # 剥完必须干净
    onnx.save(model, str(p))
    return n
