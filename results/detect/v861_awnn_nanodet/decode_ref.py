"""NanoDet-Plus-m(416, 5 类)· V861 板端 CPU 后处理参考 · 自包含(只依赖 numpy)。

板端链路：AWNN Runtime 加载 `_ipu` → 喂 0-255 BGR(归一化焊进 NPU)→ 取 blob `output`
= **logits** `[1, 3598, 37]`(**sigmoid 前**)→ 本文件在 CPU(玄铁 RISC-V)做 `sigmoid + DFL + NMS`。

⚠ 与移动端 `mobile_handoff_nanodet/nanodet_detect.py` 的唯一差别：板端导出是 **logits**(按 ADR-0007
剥了 sigmoid),所以这里 **对前 5 类先 sigmoid**;移动端 onnx/tflite 已焊 sigmoid、不用再做。DFL/NMS 一致。

本文件**不 import edge_cam、不依赖仓库其它文件**——把整个 `v861_awnn_nanodet/` 拷走即可用;
移植到 C/RISC-V 时照这 ~40 行翻译即可(与仓库内 `src/edge_cam/cascade/adapters.py:decode_nanodet` 逐值等价)。

用法(自测，需 numpy;读图尺寸用 PIL，没有就传 W H):
    python decode_ref.py round2/ref/demo_bird_output_fp32.bin round2/ref/demo_bird.jpg
    python decode_ref.py <logits.bin> <W> <H>
"""

import sys

import numpy as np

LABELS = ["bird", "squirrel", "cat", "person", "other_animal"]  # 行号=类id，与 labels.txt 同序
STRIDES = [8, 16, 32, 64]
REG_MAX = 7
INPUT = 416


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _anchors():
    cx, cy, st = [], [], []
    for s in STRIDES:
        fh = (INPUT + s - 1) // s
        for y in range(fh):
            for x in range(fh):
                cx.append(x * s)
                cy.append(y * s)
                st.append(s)
    return np.array(cx), np.array(cy), np.array(st, np.float32)


_ANCH = _anchors()


def _nms(boxes, scores, iou_th):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_th]
    return keep


def decode(logits, orig_wh, conf_thr=0.4, nms_iou=0.5):
    """logits: [3598,37] 或 [1,3598,37]（板端 `output`，sigmoid 前）。返回 [{label,score,box:[x1,y1,x2,y2]}]，原图坐标。"""
    out = np.asarray(logits, np.float32).reshape(-1, 5 + 4 * (REG_MAX + 1))
    W, H = orig_wh
    cls = _sigmoid(out[:, :5])  # ← 板端关键：logits 先 sigmoid
    reg = out[:, 5:].reshape(-1, 4, REG_MAX + 1)
    cxa, cya, sta = _ANCH
    dist = (_softmax(reg) * np.arange(REG_MAX + 1)).sum(-1) * sta[:, None]  # DFL 期望
    sx, sy = W / INPUT, H / INPUT
    x1 = np.clip((cxa - dist[:, 0]) * sx, 0, W)
    y1 = np.clip((cya - dist[:, 1]) * sy, 0, H)
    x2 = np.clip((cxa + dist[:, 2]) * sx, 0, W)
    y2 = np.clip((cya + dist[:, 3]) * sy, 0, H)
    boxes = np.stack([x1, y1, x2, y2], 1)
    score, label = cls.max(1), cls.argmax(1)
    m = score > conf_thr
    boxes, score, label = boxes[m], score[m], label[m]
    dets = []
    for c in range(5):
        idx = np.where(label == c)[0]
        if not len(idx):
            continue
        for k in _nms(boxes[idx], score[idx], nms_iou):
            dets.append({"label": LABELS[c], "score": float(score[idx][k]),
                         "box": [round(float(v), 1) for v in boxes[idx][k]]})
    return sorted(dets, key=lambda d: -d["score"])


if __name__ == "__main__":
    logits = np.fromfile(sys.argv[1], np.float32)  # 板端 dump 的 output_fp32.bin
    if len(sys.argv) >= 4 and sys.argv[2].isdigit():
        wh = (int(sys.argv[2]), int(sys.argv[3]))
    else:  # 传图片路径 → 用 PIL 读尺寸
        from PIL import Image
        im = Image.open(sys.argv[2])
        wh = im.size  # (W, H)
    dets = decode(logits, wh)
    print(f"orig={wh}  {len(dets)} detections:")
    for d in dets[:10]:
        print(" ", d)
