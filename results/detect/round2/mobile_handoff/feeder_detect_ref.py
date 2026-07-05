"""Feeder 检测 · Python 参考实现（移动端 decode+NMS 交接用；照抄 NanoDet 官方后处理）。
模型: nanodet_feeder5_mobile_416.ncnn —— sigmoid+归一化已焊入，输出 [3598,37]=[5类概率, 32框分布(DFL)]。
decode 规格(源自 nanodet 源码)：
  center = (x*stride, y*stride)  # 无 +0.5
  dist[l,t,r,b] = Σ softmax(每边8bin)·[0..7] · stride
  box = [cx-l, cy-t, cx+r, cy+b]  (416 输入空间) → 缩放回原图
用法: python feeder_detect_ref.py <image> [conf=0.4] [nms=0.5] [out.jpg]
"""
import os, sys, json
import numpy as np, cv2, ncnn

LABELS = ["bird", "squirrel", "cat", "person", "other_animal"]
STRIDES = [8, 16, 32, 64]
REG_MAX = 7
INPUT = 416
EXPORTS = os.environ.get("FEEDER_MODEL_DIR", ".")   # 模型目录（默认当前目录）
PARAM = f"{EXPORTS}/nanodet_feeder5_mobile_416.param"
BIN = f"{EXPORTS}/nanodet_feeder5_mobile_416.bin"


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def make_anchors():
    cx, cy, st = [], [], []
    for s in STRIDES:
        fh = (INPUT + s - 1) // s          # ceil(416/s): 52,26,13,7
        fw = fh
        for y in range(fh):                # 与模型 flatten 顺序一致：y 外 x 内
            for x in range(fw):
                cx.append(x * s); cy.append(y * s); st.append(s)
    return np.array(cx), np.array(cy), np.array(st, np.float32)


def nms(boxes, scores, iou_th):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores, kind="stable")   # 稳定排序：平票按原(anchor)序，与 C++ stable_sort 一致
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None); h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_th]
    return keep


_NET_CACHE = {}


def _get_net(use_fp16):
    if use_fp16 not in _NET_CACHE:
        net = ncnn.Net()
        net.opt.use_fp16_storage = use_fp16      # 移动端默认 True（体积/内存减半、近无损）
        net.opt.use_fp16_arithmetic = use_fp16
        net.opt.use_fp16_packed = use_fp16
        net.load_param(PARAM); net.load_model(BIN)
        _NET_CACHE[use_fp16] = net
    return _NET_CACHE[use_fp16]


def detect(img, conf=0.4, nms_th=0.5, use_fp16=True):
    H, W = img.shape[:2]
    net = _get_net(use_fp16)                     # 模型只加载一次、复用
    mat = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.PIXEL_BGR, W, H, INPUT, INPUT)
    ex = net.create_extractor(); ex.input("in0", mat)
    _, out = ex.extract("out0")
    out = np.array(out)                                   # (3598,37)
    cls = out[:, :5]                                      # 概率(已sigmoid)
    reg = out[:, 5:].reshape(-1, 4, REG_MAX + 1)          # (3598,4,8)
    cxa, cya, sta = make_anchors()
    dist = (softmax(reg) * np.arange(REG_MAX + 1)).sum(-1) * sta[:, None]  # (3598,4) px, 416 空间
    # 与 C++ 同序：decode → 缩放到原图 + 裁剪到图像边界 → 再 NMS（NMS 在原图空间做）
    sx, sy = W / INPUT, H / INPUT
    x1 = np.clip((cxa - dist[:, 0]) * sx, 0, W); y1 = np.clip((cya - dist[:, 1]) * sy, 0, H)
    x2 = np.clip((cxa + dist[:, 2]) * sx, 0, W); y2 = np.clip((cya + dist[:, 3]) * sy, 0, H)
    boxes = np.stack([x1, y1, x2, y2], 1)
    score = cls.max(1); label = cls.argmax(1)
    m = score > conf
    boxes, score, label = boxes[m], score[m], label[m]
    dets = []
    for c in range(len(LABELS)):                          # 逐类 NMS
        idx = np.where(label == c)[0]
        if not len(idx): continue
        for k in nms(boxes[idx], score[idx], nms_th):
            b = boxes[idx][k]
            dets.append({
                "label": LABELS[c], "score": round(float(score[idx][k]), 4),
                "box": [round(float(b[0]), 1), round(float(b[1]), 1),
                        round(float(b[2]), 1), round(float(b[3]), 1)],
            })
    return sorted(dets, key=lambda d: -d["score"])


if __name__ == "__main__":
    path = sys.argv[1]
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    nms_th = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    out_jpg = sys.argv[4] if len(sys.argv) > 4 else None
    img = cv2.imread(path)
    dets = detect(img, conf, nms_th)
    for d in dets:
        print(json.dumps(d, ensure_ascii=False))
    print(f"# {len(dets)} detections", file=sys.stderr)
    if out_jpg:
        for d in dets:
            x1, y1, x2, y2 = map(int, d["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{d["label"]} {d["score"]:.2f}', (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(out_jpg, img)
        print(f"# drawn -> {out_jpg}", file=sys.stderr)
