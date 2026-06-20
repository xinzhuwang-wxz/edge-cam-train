"""级联真实 adapter（#12）：ORT 跑 ONNX,实现 Detector/Classifier seam。

- OnnxClassifier：分类 onnx(crop→top1/conf/top5),预处理 = resize+ImageNet 归一(对齐训练)。
- OnnxDetector：检测 onnx(裸 head 输出 1×N×43)+ standalone numpy NanoDet-Plus 解码
  (GFL 分布→bbox + grid priors + 逐类 NMS),输出原图坐标。decode_nanodet 是纯函数,可单测。

⚠️ 真 onnx 端到端需检测 onnx(实验1 在板外 GPU 跑过);decode 数学由合成输出单测保证。
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from edge_cam.cascade.pipeline import Detection

_IM_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IM_STD = np.array([0.229, 0.224, 0.225], np.float32)
# NanoDet 输入归一(BGR,plan/config)
_DET_MEAN = np.array([103.53, 116.28, 123.675], np.float32)
_DET_STD = np.array([57.375, 57.12, 58.395], np.float32)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """逐类贪心 NMS,返回保留下标。boxes: (n,4) x1y1x2y2。"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def decode_nanodet(
    output: np.ndarray,
    orig_wh: tuple[int, int],
    *,
    input_size: int = 416,
    strides: tuple[int, ...] = (8, 16, 32, 64),
    reg_max: int = 7,
    num_classes: int = 11,
    conf_thr: float = 0.3,
    nms_iou: float = 0.6,
) -> list[Detection]:
    """NanoDet-Plus 裸 head 输出 (1,N,43) → 原图坐标检测(纯函数,可单测)。

    通道:[:num_classes]=cls logits(sigmoid);[num_classes:]=4×(reg_max+1) 分布。
    priors = 各 stride 网格中心(arange*stride,与 NanoDet get_single_level_center_priors 一致)。
    距离 = Integral(softmax(reg)·arange(reg_max+1))×stride;box=[cx-l,cy-t,cx+r,cy+b]→缩放回原图。"""
    out = output[0]  # (N, 43)
    cls = _sigmoid(out[:, :num_classes])
    reg = out[:, num_classes:].reshape(-1, 4, reg_max + 1)
    dist = (_softmax(reg, axis=-1) * np.arange(reg_max + 1, dtype=np.float32)).sum(-1)  # (N,4)

    # 网格中心 priors（与各 stride 的 featmap 顺序拼接,需与导出时一致）
    cx, cy, st = [], [], []
    for s in strides:
        h = w = int(np.ceil(input_size / s))
        xs = (np.arange(w, dtype=np.float32)) * s
        ys = (np.arange(h, dtype=np.float32)) * s
        gx, gy = np.meshgrid(xs, ys)
        cx.append(gx.ravel())
        cy.append(gy.ravel())
        st.append(np.full(h * w, s, np.float32))
    cx, cy, st = np.concatenate(cx), np.concatenate(cy), np.concatenate(st)
    if len(cx) != len(out):
        raise ValueError(
            f"priors({len(cx)}) 与输出 anchor 数({len(out)})不符,检查 strides/input_size"
        )

    d = dist * st[:, None]  # 像素距离 (N,4): l,t,r,b
    x1, y1 = cx - d[:, 0], cy - d[:, 1]
    x2, y2 = cx + d[:, 2], cy + d[:, 3]
    boxes_in = np.stack([x1, y1, x2, y2], 1)
    sx, sy = orig_wh[0] / input_size, orig_wh[1] / input_size
    boxes = boxes_in * np.array([sx, sy, sx, sy], np.float32)

    score = cls.max(1)
    cid = cls.argmax(1)
    dets: list[Detection] = []
    for c in range(num_classes):
        m = (cid == c) & (score >= conf_thr)
        if not m.any():
            continue
        b, s = boxes[m], score[m]
        for k in _nms(b, s, nms_iou):
            x1, y1, x2, y2 = b[k]
            dets.append(
                Detection(
                    box=(float(x1), float(y1), float(x2), float(y2)), class_id=c, score=float(s[k])
                )
            )
    return dets


class OnnxClassifier:
    """分类 ONNX adapter(Classifier seam)。crop(PIL RGB)→(top1, 置信, top5)。"""

    def __init__(self, onnx_path: str, input_size: int = 224) -> None:
        import onnxruntime as ort

        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        self.input_size = input_size

    def classify(self, crop: Image.Image) -> tuple[int, float, list[int]]:
        im = crop.convert("RGB").resize((self.input_size, self.input_size), Image.BILINEAR)
        x = (np.asarray(im, np.float32) / 255.0 - _IM_MEAN) / _IM_STD
        x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)
        logits = self.sess.run(None, {self.iname: x})[0][0]
        prob = _softmax(logits)
        top5 = prob.argsort()[::-1][:5].tolist()
        return int(top5[0]), float(prob[top5[0]]), top5


class OnnxDetector:
    """检测 ONNX adapter(Detector seam)。image(PIL)→ list[Detection](原图坐标)。"""

    def __init__(
        self,
        onnx_path: str,
        input_size: int = 416,
        num_classes: int = 11,
        conf_thr: float = 0.3,
        nms_iou: float = 0.6,
    ) -> None:
        import onnxruntime as ort

        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        self.input_size = input_size
        self.num_classes = num_classes
        self.conf_thr = conf_thr
        self.nms_iou = nms_iou

    def detect(self, image: Image.Image) -> list[Detection]:
        rgb = image.convert("RGB").resize((self.input_size, self.input_size), Image.BILINEAR)
        arr = np.asarray(rgb, np.float32)[:, :, ::-1]  # RGB→BGR(NanoDet)
        x = (arr - _DET_MEAN) / _DET_STD
        x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)
        out = self.sess.run(None, {self.iname: x})[0]
        return decode_nanodet(
            out,
            (image.width, image.height),
            input_size=self.input_size,
            num_classes=self.num_classes,
            conf_thr=self.conf_thr,
            nms_iou=self.nms_iou,
        )
