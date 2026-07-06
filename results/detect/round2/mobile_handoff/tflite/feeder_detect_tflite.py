"""Feeder 检测 · TFLite 参考（**焊入版，接口与 ncnn 完全一致**）。
模型 feeder_416.{fp32,fp16}.tflite —— sigmoid + 归一化**已焊进图**，和 ncnn 一样：
  喂 0-255 BGR（resize 416）→ 概率 [3598,37]，decode/NMS 与 feeder_detect_ref.py 逐行相同。
唯一区别 = runtime（tflite interpreter 换掉 ncnn），预处理/后处理/decode 全不变。
用法: python tflite/feeder_detect_tflite.py <image> [fp32|fp16] [conf] [nms] [out.jpg]（从包根跑）
"""

import json
import os
import sys

import cv2
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except Exception:
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception:
        from tensorflow.lite import Interpreter

LABELS = ["bird", "squirrel", "cat", "person", "other_animal"]
STRIDES = [8, 16, 32, 64]
REG_MAX = 7
INPUT = 416
MODEL_DIR = os.environ.get("FEEDER_MODEL_DIR", os.path.dirname(os.path.abspath(__file__)))


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def make_anchors():
    cx, cy, st = [], [], []
    for s in STRIDES:
        fh = (INPUT + s - 1) // s
        for y in range(fh):
            for x in range(fh):
                cx.append(x * s)
                cy.append(y * s)
                st.append(s)
    return np.array(cx), np.array(cy), np.array(st, np.float32)


def nms(boxes, scores, iou_th):
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
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_th]
    return keep


_ANCH = make_anchors()
_ITP = {}


def detect(img, prec="fp32", conf=0.4, nms_th=0.5):
    H, W = img.shape[:2]
    if prec not in _ITP:
        it = Interpreter(model_path=f"{MODEL_DIR}/feeder_416.{prec}.tflite")
        it.allocate_tensors()
        _ITP[prec] = (it, it.get_input_details()[0], it.get_output_details()[0])
    it, ii, oo = _ITP[prec]
    x = cv2.resize(img, (INPUT, INPUT)).astype(np.float32)[None]  # 0-255 BGR NHWC（不归一化，已焊）
    it.set_tensor(ii["index"], x)
    it.invoke()
    out = np.array(it.get_tensor(oo["index"])).reshape(-1, 37)  # 概率（cls 已 sigmoid，已焊）
    cls = out[:, :5]  # 直接是概率（与 ncnn 一样）
    reg = out[:, 5:].reshape(-1, 4, REG_MAX + 1)
    cxa, cya, sta = _ANCH
    dist = (softmax(reg) * np.arange(REG_MAX + 1)).sum(-1) * sta[:, None]
    sx, sy = W / INPUT, H / INPUT
    x1 = np.clip((cxa - dist[:, 0]) * sx, 0, W)
    y1 = np.clip((cya - dist[:, 1]) * sy, 0, H)
    x2 = np.clip((cxa + dist[:, 2]) * sx, 0, W)
    y2 = np.clip((cya + dist[:, 3]) * sy, 0, H)
    boxes = np.stack([x1, y1, x2, y2], 1)
    score = cls.max(1)
    label = cls.argmax(1)
    m = score > conf
    boxes, score, label = boxes[m], score[m], label[m]
    dets = []
    for c in range(len(LABELS)):
        idx = np.where(label == c)[0]
        if not len(idx):
            continue
        for k in nms(boxes[idx], score[idx], nms_th):
            b = boxes[idx][k]
            dets.append(
                {
                    "label": LABELS[c],
                    "score": round(float(score[idx][k]), 4),
                    "box": [round(float(v), 1) for v in b],
                }
            )
    return sorted(dets, key=lambda d: -d["score"])


if __name__ == "__main__":
    path = sys.argv[1]
    prec = sys.argv[2] if len(sys.argv) > 2 else "fp32"
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
    nms_th = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    out_jpg = sys.argv[5] if len(sys.argv) > 5 else None
    img = cv2.imread(path)
    dets = detect(img, prec, conf, nms_th)
    for d in dets:
        print(json.dumps(d, ensure_ascii=False))
    print(f"# {len(dets)} detections ({prec})", file=sys.stderr)
    if out_jpg is not None:
        for d in dets:
            a, b, c, e = (int(v) for v in d["box"])
            cv2.rectangle(img, (a, b), (c, e), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{d['label']} {d['score']:.2f}",
                (a, max(0, b - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        cv2.imwrite(out_jpg, img)
