"""Feeder 检测器 · ppyoloe_plus_crn_s(5 类)· ONNX / TFLite 参考实现。

接口对齐 round2 NanoDet 交接包:`detect(image_bgr) -> [{"label","score","box":[x1,y1,x2,y2]}]`,
喂**任意尺寸 BGR 像素(0-255,cv2.imread 原样)**,输出**原图像素坐标**框。默认 conf=0.50 / nms=0.50
(round3 用户要求比 round2 的 0.4 稍高;阈值是 **CPU decode 侧的旋钮、非焊进模型**,移动端/板端都可随时调 0.4/0.5/0.6)。

与 round2 的唯一内部差异(已封装,调用方无感):输入 640(非 416)、预处理 RGB+÷255(PP-YOLOE)、
模型输出 boxes[8400,4]+scores[5,8400]、decode 用本文件的 NMS。换 onnx/tflite 只换 runner,decode 不变。

用法:
    from ppyoloe_detect import PPYoloeDetector
    det = PPYoloeDetector("ppyoloe_s_640.onnx")            # 或 .tflite
    for d in det.detect(cv2.imread("x.jpg")): print(d)
"""

import cv2
import numpy as np

NAMES = ["bird", "squirrel", "cat", "person", "other_animal"]  # 行号=类id,与 labels.txt / round2 同序
INPUT = 640


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """标准 NMS(单类),boxes=[N,4] xyxy。返回保留 index。"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


class PPYoloeDetector:
    def __init__(self, model_path: str, conf: float = 0.50, nms: float = 0.50):
        self.conf, self.nms = conf, nms
        self.is_onnx = model_path.endswith(".onnx")
        if self.is_onnx:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        else:
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite import Interpreter
            self.it = Interpreter(model_path=model_path)
            self.it.allocate_tensors()
            self.ind, self.outd = self.it.get_input_details(), self.it.get_output_details()

    def _preprocess(self, im_bgr):
        h0, w0 = im_bgr.shape[:2]
        im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (INPUT, INPUT), interpolation=cv2.INTER_LINEAR).astype("float32") / 255.0
        blob = np.transpose(im, (2, 0, 1))[None]  # NCHW [1,3,640,640]
        sf = np.array([[INPUT / h0, INPUT / w0]], "float32")  # PP-YOLOE scale_factor;模型据此把框缩回原图
        return blob, sf

    def _infer(self, blob, sf):
        """→ boxes[8400,4](原图坐标), scores[5,8400](已 sigmoid)。onnx/tflite 输出一致(已三方验证)。"""
        if self.is_onnx:
            boxes, scores = self.sess.run(None, {"image": blob, "scale_factor": sf})
            return boxes[0], scores[0]
        for d in self.ind:  # tflite:onnx2tf 转 NHWC,按 shape 喂
            sh = list(d["shape"])
            if len(sh) == 4 and sh[-1] == 3:
                self.it.set_tensor(d["index"], np.transpose(blob, (0, 2, 3, 1)).astype(d["dtype"]))
            elif len(sh) == 4:
                self.it.set_tensor(d["index"], blob.astype(d["dtype"]))
            elif tuple(sh) == (1, 2):
                self.it.set_tensor(d["index"], sf.astype(d["dtype"]))
        self.it.invoke()
        outs = {tuple(o["shape"]): self.it.get_tensor(o["index"]) for o in self.outd}
        boxes = next(v for k, v in outs.items() if k[-1] == 4)[0]
        scores = next(v for k, v in outs.items() if 5 in k and k[-1] != 4)[0]
        return boxes, scores

    def detect(self, im_bgr) -> list[dict]:
        blob, sf = self._preprocess(im_bgr)
        boxes, scores = self._infer(blob, sf)  # [8400,4], [5,8400]
        scores = scores.T  # [8400,5]
        dets = []
        for c in range(len(NAMES)):
            sc = scores[:, c]
            m = sc >= self.conf
            if not m.any():
                continue
            b, s = boxes[m], sc[m]
            for i in _nms(b, s, self.nms):
                dets.append({"label": NAMES[c], "score": float(s[i]),
                             "box": [round(float(x), 1) for x in b[i]]})
        return sorted(dets, key=lambda d: -d["score"])


if __name__ == "__main__":
    import sys
    det = PPYoloeDetector(sys.argv[1])
    out = det.detect(cv2.imread(sys.argv[2]))
    print(f"{len(out)} detections:")
    for d in out[:10]:
        print(" ", d)
