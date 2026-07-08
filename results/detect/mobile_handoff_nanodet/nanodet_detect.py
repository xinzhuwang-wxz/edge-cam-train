"""NanoDet-Plus-m(ShuffleNetV2 1.0x, 416, 5 类)移动端参考实现 · ONNX / TFLite 通用。

接口对齐 round2 / ppyoloe 交接包:`detect(image_bgr) -> [{"label","score","box":[x1,y1,x2,y2]}]`,
喂**任意尺寸 BGR 像素(0-255,cv2.imread 原样)**,输出**原图像素坐标**框。默认 conf=0.4 / nms=0.5
(阈值是 **CPU decode 侧的旋钮、非焊进模型**,移动端/板端可随时调)。

模型内部(已封装,调用方无感):sigmoid + 归一化焊进图;输入 416(**onnx=NCHW / tflite=NHWC**,自动适配);
输出 `[3598,37]` = 5 类概率 + 4×(REG_MAX+1) DFL 回归;decode(DFL+anchor)+ NMS 在本文件 CPU 侧。
换 onnx/tflite 只换 runner,decode 不变(已数值验证两者 ≡)。

用法:
    from nanodet_detect import NanoDet
    det = NanoDet("round3/p2/onnx/p2_416.onnx")          # 或 round3/p2/tflite/p2_416.fp16.tflite
    for d in det.detect(cv2.imread("x.jpg")): print(d)   # {'label','score','box':[x1,y1,x2,y2]}
命令行自测:python nanodet_detect.py round3/p2/onnx/p2_416.onnx some.jpg
"""
import cv2, numpy as np
LABELS = ["bird", "squirrel", "cat", "person", "other_animal"]  # 行号=类id,与 labels.txt / round2 同序
STRIDES = [8, 16, 32, 64]; REG_MAX = 7; INPUT = 416


def _softmax(x): e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def _anchors():
    cx, cy, st = [], [], []
    for s in STRIDES:
        fh = (INPUT + s - 1) // s
        for y in range(fh):
            for x in range(fh): cx.append(x * s); cy.append(y * s); st.append(s)
    return np.array(cx), np.array(cy), np.array(st, np.float32)


_ANCH = _anchors()


def _nms(boxes, scores, iou_th):
    x1, y1, x2, y2 = boxes.T; areas = (x2 - x1) * (y2 - y1); order = np.argsort(-scores, kind="stable"); keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9); order = order[1:][iou <= iou_th]
    return keep


class NanoDet:
    """ONNX / TFLite 通用。按扩展名自动选 runtime;decode+NMS 相同。"""

    def __init__(self, path, conf=0.4, nms=0.5):
        self.conf, self.nms = conf, nms
        self.is_onnx = path.endswith(".onnx")
        if self.is_onnx:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.iname = self.sess.get_inputs()[0].name
        else:
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite import Interpreter
            self.it = Interpreter(model_path=path); self.it.allocate_tensors()
            self.ind, self.outd = self.it.get_input_details(), self.it.get_output_details()

    def _infer(self, x):  # x: resize 后 BGR 0-255 float32 [416,416,3]
        if self.is_onnx:
            blob = np.transpose(x, (2, 0, 1))[None]  # NCHW [1,3,416,416]
            return self.sess.run(None, {self.iname: blob})[0].reshape(-1, 37)
        d = self.ind[0]  # tflite:onnx2tf 转 NHWC [1,416,416,3]
        self.it.set_tensor(d["index"], x[None].astype(d["dtype"]))
        self.it.invoke()
        return self.it.get_tensor(self.outd[0]["index"]).reshape(-1, 37)

    def detect(self, img):
        H, W = img.shape[:2]
        x = cv2.resize(img, (INPUT, INPUT)).astype(np.float32)  # BGR 0-255
        out = self._infer(x)
        cls = out[:, :5]; reg = out[:, 5:].reshape(-1, 4, REG_MAX + 1)
        cxa, cya, sta = _ANCH
        dist = (_softmax(reg) * np.arange(REG_MAX + 1)).sum(-1) * sta[:, None]
        sx, sy = W / INPUT, H / INPUT
        x1 = np.clip((cxa - dist[:, 0]) * sx, 0, W); y1 = np.clip((cya - dist[:, 1]) * sy, 0, H)
        x2 = np.clip((cxa + dist[:, 2]) * sx, 0, W); y2 = np.clip((cya + dist[:, 3]) * sy, 0, H)
        boxes = np.stack([x1, y1, x2, y2], 1); score = cls.max(1); label = cls.argmax(1)
        m = score > self.conf; boxes, score, label = boxes[m], score[m], label[m]
        dets = []
        for c in range(5):
            idx = np.where(label == c)[0]
            if not len(idx): continue
            for k in _nms(boxes[idx], score[idx], self.nms):
                dets.append({"label": LABELS[c], "score": float(score[idx][k]), "box": [round(float(v), 1) for v in boxes[idx][k]]})
        return sorted(dets, key=lambda d: -d["score"])


NanoDetONNX = NanoDet  # 向后兼容别名(旧代码 import NanoDetONNX)

if __name__ == "__main__":
    import sys
    det = NanoDet(sys.argv[1])
    out = det.detect(cv2.imread(sys.argv[2]))
    print(f"{len(out)} detections:")
    for d in out[:10]:
        print(" ", d)
