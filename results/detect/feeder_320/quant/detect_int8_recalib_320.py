"""控制变量实验：增大 calib 校准样本数能否压回 feeder_320 int8 量化掉点。

在同一个固定 val 子集（前 N_SUBSET=1600 张，shuffle=False 确定性）上跑三遍 COCOeval：
  1. fp32（子集 baseline）
  2. int8 @ calib=120（复现旧设置）
  3. int8 @ calib=1000（增大 calib）

量化均 per-channel / opset13 / QInt8（同蓝本 detect_int8_eval_320.py）。
delogit 双 sigmoid 修正、post_process、CalibReader 沿用蓝本。

COCOeval 限定 params.imgIds = 子集 img_ids（控制变量 + 提速；否则 GT 全量会
把缺失图当漏检，子集 mAP 失真）。

CPU 限制（同机 416 训练在跑，绝不抢）：
  OMP_NUM_THREADS=4 / ort intra=4 inter=1 / DataLoader workers=2。
脚本由 `nice -n 19` 启动。
"""
import contextlib
import copy
import io
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import onnx
import onnxruntime as ort
import torch
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from onnx import version_converter
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from pycocotools.cocoeval import COCOeval

CFG = "outputs/detect/feeder_320.yml"
FP32 = "outputs/detect/feeder_320.onnx"
PTH = "outputs/detect/feeder_320/model_best/nanodet_model_best.pth"
SAVE = "outputs/detect/int8_recalib_320"
INT8_120 = os.path.join(SAVE, "feeder_320.int8.calib120.onnx")
INT8_1000 = os.path.join(SAVE, "feeder_320.int8.calib1000.onnx")

N_SUBSET = 1600          # 固定 val 子集大小（前 N 张）
CALIB_SMALL = 120
CALIB_LARGE = 1000
THREADS = 4

os.makedirs(SAVE, exist_ok=True)
torch.set_num_threads(THREADS)

t0 = time.time()
load_config(cfg, CFG)
NUM_CLASSES = cfg.model.arch.head.num_classes
print("num_classes=%d class_names=%s" % (NUM_CLASSES, list(cfg.class_names)), flush=True)

val_dataset = build_dataset(cfg.data.val, "test")
# workers=2 限 CPU；shuffle=False 保证「前 N 张」确定性、可复现
loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=2,
    collate_fn=naive_collate, drop_last=False,
)
N_VAL = len(val_dataset)
N_EVAL = min(N_SUBSET, N_VAL)
print("val images total=%d -> eval subset=%d" % (N_VAL, N_EVAL), flush=True)

model = build_model(cfg.model)
_logger = Logger(-1, SAVE, False)
load_model_weight(model, torch.load(PTH, map_location="cpu"), _logger)
model.eval()

# 复用 dataset 上的 coco_api / cat_ids（results2json + COCOeval 都要）
COCO_API = val_dataset.coco_api
CAT_IDS = val_dataset.cat_ids


def prep(b):
    if isinstance(b["img"], list):
        b["img"] = stack_batch_img(b["img"], divisible=32)
    return b


EPS = 1e-7
def delogit_cls(preds):
    # 本 nanodet 版 _forward_onnx 已对 cls 通道 sigmoid，post_process 内 get_bboxes
    # 又 sigmoid 一次 -> 双 sigmoid。这里对前 NUM_CLASSES 通道反 sigmoid 还原 raw logits。
    p = np.clip(preds[..., :NUM_CLASSES], EPS, 1.0 - EPS)
    preds = preds.copy()
    preds[..., :NUM_CLASSES] = np.log(p / (1.0 - p))
    return preds


class CalibReader(CalibrationDataReader):
    """从 val loader 取前 n 张做校准（与蓝本一致）。"""

    def __init__(self, n):
        self.data = []
        for i, b in enumerate(loader):
            if i >= n:
                break
            self.data.append({"data": prep(b)["img"].numpy().astype(np.float32)})
        self.it = iter(self.data)
        print("  calib samples collected = %d" % len(self.data), flush=True)

    def get_next(self):
        return next(self.it, None)


# --- 准备 opset>=13 的 fp32 图（per-channel 需要） ---
FP32Q = FP32
m = onnx.load(FP32)
if m.opset_import[0].version < 13:
    m = version_converter.convert_version(m, 13)
    FP32Q = os.path.join(SAVE, "feeder_320_op13.onnx")
    onnx.save(m, FP32Q)
    print("opset升级到13 -> %s" % FP32Q, flush=True)


def quantize(calib_n, out_path):
    print("[quantize] calib=%d -> %s" % (calib_n, out_path), flush=True)
    quantize_static(
        FP32Q, out_path, CalibReader(calib_n),
        quant_format=QuantFormat.QDQ, per_channel=True, weight_type=QuantType.QInt8,
    )
    print("[quantize] done calib=%d" % calib_n, flush=True)


# --- ORT session：限 CPU 线程，绝不抢 416 ---
def make_session(onnx_path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    return ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])


def results2json(results):
    out = []
    for image_id, dets in results.items():
        for label, bboxes in dets.items():
            category_id = int(CAT_IDS[label])
            for bbox in bboxes:
                out.append(dict(
                    image_id=int(image_id),
                    category_id=category_id,
                    bbox=[float(bbox[0]), float(bbox[1]),
                          float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])],
                    score=float(bbox[4]),
                ))
    return out


def coco_eval_subset(results, subset_img_ids, tag):
    """对固定子集做 COCOeval：限定 params.imgIds = 子集 id。"""
    rj = results2json(results)
    if not rj:
        print("[%s] EMPTY detections" % tag, flush=True)
        return {"mAP": 0.0, "AP_50": 0.0}
    jpath = os.path.join(SAVE, "results_%s.json" % tag)
    json.dump(rj, open(jpath, "w"))
    coco_dets = COCO_API.loadRes(jpath)
    ev = COCOeval(copy.deepcopy(COCO_API), copy.deepcopy(coco_dets), "bbox")
    ev.params.imgIds = sorted(int(i) for i in subset_img_ids)  # ★ 限子集
    ev.evaluate()
    ev.accumulate()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.summarize()
    mAP = float(ev.stats[0])    # AP@[.5:.95]
    AP50 = float(ev.stats[1])   # AP@.5
    print("[%s] mAP@.5:.95=%.4f AP50=%.4f (subset n=%d)"
          % (tag, mAP, AP50, len(ev.params.imgIds)), flush=True)
    return {"mAP": mAP, "AP_50": AP50}


def infer_subset(onnx_path, tag, capture_ids=False):
    """跑前 N_EVAL 张，返回 results；capture_ids 时同时返回子集 img_ids。"""
    sess = make_session(onnx_path)
    iname = sess.get_inputs()[0].name
    results = {}
    subset_ids = []
    for k, b in enumerate(loader):
        if k >= N_EVAL:
            break
        b = prep(b)
        preds = sess.run(None, {iname: b["img"].numpy().astype(np.float32)})[0]
        preds = delogit_cls(preds)
        res = model.head.post_process(torch.from_numpy(preds), b)
        results.update(res)
        if capture_ids:
            subset_ids.extend(res.keys())
        if (k + 1) % 400 == 0:
            print("  [%s] %d/%d  (%.0fs)" % (tag, k + 1, N_EVAL, time.time() - t0), flush=True)
    if capture_ids:
        return results, subset_ids
    return results


# --- sanity：1 张图核对 onnx(delogit) vs pytorch eval（raw logits）数值一致 ---
def sanity_one():
    b = prep(next(iter(loader)))
    inp = b["img"].numpy().astype(np.float32)
    with torch.no_grad():
        pt = model(torch.from_numpy(inp)).numpy()
    sess = make_session(FP32)
    on = sess.run(None, {sess.get_inputs()[0].name: inp})[0]
    on_fixed = delogit_cls(on)
    cls_diff = np.abs(pt[..., :NUM_CLASSES] - on_fixed[..., :NUM_CLASSES]).max()
    reg_diff = np.abs(pt[..., NUM_CLASSES:] - on_fixed[..., NUM_CLASSES:]).max()
    print("SANITY cls_logit_maxdiff=%.5f reg_maxdiff=%.5f" % (cls_diff, reg_diff), flush=True)


sanity_one()

# 1) 量化两个 int8（各自独立校准）
quantize(CALIB_SMALL, INT8_120)
quantize(CALIB_LARGE, INT8_1000)

# 2) fp32 跑子集 + 锁定 subset img_ids
print("=== eval fp32 (capture subset ids) ===", flush=True)
res_fp32, subset_ids = infer_subset(FP32, "fp32", capture_ids=True)
subset_ids = sorted(set(int(i) for i in subset_ids))
print("subset img_ids captured = %d (unique)" % len(subset_ids), flush=True)

# 3) 两个 int8 跑同一固定子集（同 N_EVAL 张，shuffle=False 确定性一致）
print("=== eval int8@calib120 ===", flush=True)
res_i120 = infer_subset(INT8_120, "int8_120")
print("=== eval int8@calib1000 ===", flush=True)
res_i1000 = infer_subset(INT8_1000, "int8_1000")

# 4) COCOeval（限子集 img_ids）
r_fp32 = coco_eval_subset(res_fp32, subset_ids, "fp32")
r_i120 = coco_eval_subset(res_i120, subset_ids, "int8_calib120")
r_i1000 = coco_eval_subset(res_i1000, subset_ids, "int8_calib1000")

drop120 = r_fp32["mAP"] - r_i120["mAP"]
drop1000 = r_fp32["mAP"] - r_i1000["mAP"]
drop120_50 = r_fp32["AP_50"] - r_i120["AP_50"]
drop1000_50 = r_fp32["AP_50"] - r_i1000["AP_50"]

print("\n================ RESULT (subset n=%d, drops are within-subset relative) ================"
      % len(subset_ids), flush=True)
print("quant_mode=per-channel(opset13)/QDQ/QInt8  threads=%d" % THREADS, flush=True)
print("              mAP@.5:.95     AP50", flush=True)
print("fp32        : %8.4f   %8.4f" % (r_fp32["mAP"], r_fp32["AP_50"]), flush=True)
print("int8@cal120 : %8.4f   %8.4f   (drop mAP=%.4f / %.2fpt, AP50=%.4f / %.2fpt)"
      % (r_i120["mAP"], r_i120["AP_50"], drop120, drop120 * 100, drop120_50, drop120_50 * 100), flush=True)
print("int8@cal1000: %8.4f   %8.4f   (drop mAP=%.4f / %.2fpt, AP50=%.4f / %.2fpt)"
      % (r_i1000["mAP"], r_i1000["AP_50"], drop1000, drop1000 * 100, drop1000_50, drop1000_50 * 100), flush=True)
recover = (drop120 - drop1000) * 100
print("RECALIB EFFECT: mAP drop %.2fpt(calib120) -> %.2fpt(calib1000)  | 压回 %.2fpt"
      % (drop120 * 100, drop1000 * 100, recover), flush=True)
print("RECALIB EFFECT: AP50 drop %.2fpt(calib120) -> %.2fpt(calib1000) | 压回 %.2fpt"
      % (drop120_50 * 100, drop1000_50 * 100, (drop120_50 - drop1000_50) * 100), flush=True)
print("total wall=%.0fs" % (time.time() - t0), flush=True)
print("=== DETECT INT8 RECALIB DONE ===", flush=True)
