"""P5 框级复现：INT8(AWNN simulate) vs FP32(onnxruntime) 在同一预处理输入上比检测框。

用 AWNN simulate 自己 dump 的 data_fp32.bin(预处理后输入) 喂 onnxruntime，保证两边预处理逐字一致：
  FP32 参考 = onnxruntime(main_416_fp32_logits.onnx, data_fp32.bin)
  INT8      = output.npy (simulate 输出, 已 dequant)
各自 decode_nanodet → 比框一致率/IoU/类一致。全在 416 空间(orig_wh=(416,416))直接可比。
"""
import sys, glob, os
from pathlib import Path
import numpy as np

V = str(Path(__file__).resolve().parent)  # results/detect/round2/v861_awnn
sys.path.insert(0, str(Path(V).parents[3] / "src"))  # 仓库根/src
from edge_cam.cascade.adapters import decode_nanodet  # noqa: E402
import onnxruntime as ort  # noqa: E402

ONNX = f"{V}/onnx/main_416_fp32_logits.onnx"
# 可传入 results 目录(默认 logits/results); 例: python decode_repro.py build_out/logits_minmax/results
RES = f"{V}/{sys.argv[1]}" if len(sys.argv) > 1 else f"{V}/build_out/logits/results"
CONF, IOU = 0.40, 0.50  # 交接包默认阈值

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(fp, q):
    """以 FP32 为真值，贪心匹配 INT8：同类 + IoU>=0.5。返回(命中数, IoU列表, 分差列表)。"""
    used = [False] * len(q)
    ious, dsc = [], []
    for f in fp:
        best, bj = 0.0, -1
        for j, g in enumerate(q):
            if used[j] or g.class_id != f.class_id:
                continue
            v = iou(f.box, g.box)
            if v > best:
                best, bj = v, j
        if bj >= 0 and best >= 0.5:
            used[bj] = True
            ious.append(best)
            dsc.append(abs(f.score - q[bj].score))
    return len(ious), ious, dsc


tot_fp = tot_i8 = tot_m = 0
all_iou, all_dsc, cos_all = [], [], []
print(f"{'image':40s} {'FP32det':>7} {'INT8det':>7} {'matched':>7} {'meanIoU':>8} {'cos(logit)':>10}")
for d in sorted(glob.glob(f"{RES}/data_*.jpg")):
    inp = np.fromfile(f"{d}/data_fp32.bin", dtype=np.float32).reshape(1, 3, 416, 416)
    out_fp = sess.run(None, {iname: inp})[0].reshape(1, 3598, 37)
    out_i8 = np.load(f"{d}/output.npy").reshape(1, 3598, 37)
    # logit 空间 cos-sim(整体)
    a, b = out_fp.ravel(), out_i8.ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    cos_all.append(cos)
    dfp = decode_nanodet(out_fp, (416, 416), num_classes=5, conf_thr=CONF, nms_iou=IOU)
    di8 = decode_nanodet(out_i8, (416, 416), num_classes=5, conf_thr=CONF, nms_iou=IOU)
    m, ious, dsc = match(dfp, di8)
    tot_fp += len(dfp); tot_i8 += len(di8); tot_m += m
    all_iou += ious; all_dsc += dsc
    mi = np.mean(ious) if ious else 0.0
    print(f"{os.path.basename(d):40s} {len(dfp):7d} {len(di8):7d} {m:7d} {mi:8.3f} {cos:10.4f}")

print("-" * 84)
rec = tot_m / tot_fp if tot_fp else 1.0
prec = tot_m / tot_i8 if tot_i8 else 1.0
print(f"合计 FP32框={tot_fp} INT8框={tot_i8} 匹配={tot_m}")
print(f"框召回(INT8命中FP32)={rec:.3f}  框精度(INT8无多余)={prec:.3f}")
print(f"匹配框 平均IoU={np.mean(all_iou) if all_iou else 0:.3f}  平均|分差|={np.mean(all_dsc) if all_dsc else 0:.4f}")
print(f"logit空间 平均cos-sim={np.mean(cos_all):.4f}")
