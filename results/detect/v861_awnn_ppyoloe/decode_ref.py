"""PP-YOLOE-s @ V861 NPU —— 板端 CPU 后处理参考实现（自包含，只依赖 numpy）。

NPU 出 6 个裸 logits（3 尺度 × cls/reg），CPU 做：sigmoid(cls) + DFL(softmax+积分) + anchor 解码 + 逐类 NMS。

板端 I/O 契约
-------------
输入 : RGB, HWC, uint8 0-255, 640x640   (归一化 /255 已折进 NPU 的 ImageProcess 层)
       ⚠ 注意是 **RGB**（nanodet 那版是 BGR），别搞反
输出 : 6 个 FP32 CHW blob
       conv2d_81.tmp_0 [5, 80, 80]   cls  stride 8
       conv2d_84.tmp_0 [68,80, 80]   reg  stride 8
       conv2d_74.tmp_0 [5, 40, 40]   cls  stride 16
       conv2d_77.tmp_0 [68,40, 40]   reg  stride 16
       conv2d_67.tmp_0 [5, 20, 20]   cls  stride 32
       conv2d_70.tmp_0 [68,20, 20]   reg  stride 32
       reg 68 通道 = 4 边 × 17 个 DFL bin (reg_max=16)

用法
----
    from decode_ref import decode_ppyoloe
    dets = decode_ppyoloe(cls_list, reg_list, orig_w, orig_h)   # → [{label,score,box}]
"""

from __future__ import annotations

import numpy as np

NAMES = ["bird", "squirrel", "cat", "person", "other_animal"]  # 行号=类id
INPUT = 640
STRIDES = (8, 16, 32)
REG_MAX = 16  # 17 个 bin: 0..16


def _softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _nms(boxes, scores, thr):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep


def decode_ppyoloe(cls_list, reg_list, orig_w, orig_h, conf=0.45, nms_thr=0.50):
    """cls_list/reg_list: 按 stride 8/16/32 顺序的 numpy 数组 (CHW)。→ 原图像素坐标框。"""
    boxes_all, scores_all = [], []
    for cls, reg, stride in zip(cls_list, reg_list, STRIDES):
        C, H, W = cls.shape  # [5,H,W]
        # cls → sigmoid
        s = 1.0 / (1.0 + np.exp(-cls.astype(np.float32)))
        s = s.reshape(C, -1).T  # [H*W, 5]

        # reg [68,H,W] → [4,17,H,W] → softmax(bins) → 积分 = 到四边距离(单位: cell)
        r = reg.astype(np.float32).reshape(4, REG_MAX + 1, H, W)
        p = _softmax(r, axis=1)
        bins = np.arange(REG_MAX + 1, dtype=np.float32).reshape(1, -1, 1, 1)
        d = (p * bins).sum(axis=1)  # [4,H,W] = l,t,r,b
        d = d.reshape(4, -1).T  # [H*W, 4]

        # anchor 中心 (grid_cell_offset=0.5)
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        cx = (xx.reshape(-1) + 0.5) * stride
        cy = (yy.reshape(-1) + 0.5) * stride

        boxes_all.append(np.stack([
            cx - d[:, 0] * stride, cy - d[:, 1] * stride,
            cx + d[:, 2] * stride, cy + d[:, 3] * stride,
        ], axis=1))
        scores_all.append(s)

    boxes = np.concatenate(boxes_all, 0)    # [8400,4] @ 640 坐标
    scores = np.concatenate(scores_all, 0)  # [8400,5]

    # 缩回原图（预处理是直接 resize 到 640，非 letterbox）
    boxes[:, [0, 2]] *= orig_w / INPUT
    boxes[:, [1, 3]] *= orig_h / INPUT
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)

    dets = []
    for c in range(len(NAMES)):
        sc = scores[:, c]
        m = sc >= conf
        if not m.any():
            continue
        b, s = boxes[m], sc[m]
        for i in _nms(b, s, nms_thr):
            dets.append({"label": NAMES[c], "score": float(s[i]),
                         "box": [round(float(v), 1) for v in b[i]]})
    return sorted(dets, key=lambda d: -d["score"])


if __name__ == "__main__":
    # 自测：用板端 dump 的 6 个 FP32 blob 解码
    import sys
    from pathlib import Path

    ref = Path(sys.argv[1] if len(sys.argv) > 1 else "ref")
    ow, oh = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (640, 640)

    def load(name, c, hw):
        return np.fromfile(ref / f"{name}_awnn_chw_fp32.bin", dtype=np.float32).reshape(c, hw, hw)

    cls_list = [load("conv2d_81.tmp_0", 5, 80), load("conv2d_74.tmp_0", 5, 40), load("conv2d_67.tmp_0", 5, 20)]
    reg_list = [load("conv2d_84.tmp_0", 68, 80), load("conv2d_77.tmp_0", 68, 40), load("conv2d_70.tmp_0", 68, 20)]

    for d in decode_ppyoloe(cls_list, reg_list, ow, oh):
        print(f"  {d['label']:14} {d['score']:.3f}  {d['box']}")
