"""每类召回率（fp32 ONNX），口径对齐实验1：conf>=0.3 / IoU>=0.5 / 类别正确。
val 800 图子集（与实验1 的 64.5% bird 召回可比）。严格限 CPU（4 线程 / nice -19）。
"""
import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import json
import numpy as np
import onnxruntime as ort
import torch
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config, load_model_weight, Logger

CFG = "outputs/detect/feeder_320.yml"
FP32 = "outputs/detect/feeder_320.int8.onnx"
PTH = "outputs/detect/feeder_320/model_best/nanodet_model_best.pth"
GT_JSON = "/root/autodl-tmp/detect_raw/processed/labels/train_val.json"
SAVE = "outputs/detect/bird_recall_int8_320"
N_IMG = 800
SCORE_THR = 0.3
IOU_THR = 0.5
os.makedirs(SAVE, exist_ok=True)

load_config(cfg, CFG)
NUM_CLASSES = cfg.model.arch.head.num_classes
CLASS_NAMES = list(cfg.class_names)
print("num_classes=%d class_names=%s" % (NUM_CLASSES, CLASS_NAMES), flush=True)

# --- GT 类映射：COCO category_id(按 id 升序) -> 0..4，与 nanodet CocoDataset 一致 ---
gt = json.load(open(GT_JSON))
cats_sorted = sorted(gt["categories"], key=lambda c: c["id"])
catid2cls = {c["id"]: i for i, c in enumerate(cats_sorted)}
cls2name = {i: c["name"] for i, c in enumerate(cats_sorted)}
print("catid2cls=%s" % catid2cls, flush=True)
# 名字一致性检查（防顺序漂移）
assert [cls2name[i] for i in range(NUM_CLASSES)] == CLASS_NAMES, "class order mismatch!"

# GT: image_id -> list[(cls, [x1,y1,x2,y2])]
gt_by_img = {}
for a in gt["annotations"]:
    cls = catid2cls[a["category_id"]]
    x, y, w, h = a["bbox"]
    gt_by_img.setdefault(a["image_id"], []).append((cls, [x, y, x + w, y + h]))

val_dataset = build_dataset(cfg.data.val, "test")
loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=2,
    collate_fn=naive_collate, drop_last=False,
)
print("val images total = %d, 评估子集 = 前 %d 张" % (len(val_dataset), N_IMG), flush=True)

model = build_model(cfg.model)
_logger = Logger(-1, SAVE, False)
load_model_weight(model, torch.load(PTH, map_location="cpu"), _logger)
model.eval()

EPS = 1e-7
def delogit_cls(preds):
    # ONNX 前 NUM_CLASSES 通道是已 sigmoid 的概率；post_process 期望 raw logits
    p = np.clip(preds[..., :NUM_CLASSES], EPS, 1.0 - EPS)
    preds = preds.copy()
    preds[..., :NUM_CLASSES] = np.log(p / (1.0 - p))
    return preds

def prep(b):
    if isinstance(b["img"], list):
        b["img"] = stack_batch_img(b["img"], divisible=32)
    return b

so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.inter_op_num_threads = 1
sess = ort.InferenceSession(FP32, sess_options=so, providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name

# pred: image_id -> {cls: list[(score,[x1,y1,x2,y2])]}
pred_by_img = {}
eval_img_ids = set()
for k, b in enumerate(loader):
    if k >= N_IMG:
        break
    b = prep(b)
    out = sess.run(None, {iname: b["img"].numpy().astype(np.float32)})[0]
    out = delogit_cls(out)
    res = model.head.post_process(torch.from_numpy(out), b)  # {img_id:{cls:[[x1,y1,x2,y2,score]..]}}
    for img_id, per_cls in res.items():
        img_id = int(img_id)
        eval_img_ids.add(img_id)
        d = {}
        for cls, boxes in per_cls.items():
            keep = [(float(box[4]), box[:4]) for box in boxes if float(box[4]) >= SCORE_THR]
            if keep:
                d[int(cls)] = keep
        pred_by_img[img_id] = d
    if (k + 1) % 200 == 0:
        print("  推理 %d/%d" % (k + 1, N_IMG), flush=True)

print("实际评估图数 = %d" % len(eval_img_ids), flush=True)

def iou_xyxy(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ba = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = aa + ba - inter
    return inter / u if u > 0 else 0.0

# 每类内贪心匹配：pred 按 score 降序，类别正确 + IoU>=0.5，一个 GT 最多匹配一次
gt_cnt = [0] * NUM_CLASSES
tp_cnt = [0] * NUM_CLASSES
for img_id in eval_img_ids:
    gts = [g for g in gt_by_img.get(img_id, [])]
    # per-class GT boxes
    gt_per_cls = {c: [] for c in range(NUM_CLASSES)}
    for cls, box in gts:
        gt_per_cls[cls].append(box)
        gt_cnt[cls] += 1
    preds = pred_by_img.get(img_id, {})
    for cls in range(NUM_CLASSES):
        gboxes = gt_per_cls[cls]
        if not gboxes:
            continue
        used = [False] * len(gboxes)
        plist = sorted(preds.get(cls, []), key=lambda x: -x[0])
        for score, pbox in plist:
            best_iou, best_j = 0.0, -1
            for j, gbox in enumerate(gboxes):
                if used[j]:
                    continue
                iou = iou_xyxy(pbox, gbox)
                if iou >= IOU_THR and iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                used[best_j] = True
                tp_cnt[cls] += 1

print("\n=== 每类召回率（fp32 ONNX, score>=%.2f, IoU>=%.2f, 类别正确, 子集 %d 图）===" % (
    SCORE_THR, IOU_THR, len(eval_img_ids)), flush=True)
print("| 类 | GT | TP | 召回率 |", flush=True)
print("|---|---|---|---|", flush=True)
tot_gt = tot_tp = 0
for c in range(NUM_CLASSES):
    rec = tp_cnt[c] / gt_cnt[c] if gt_cnt[c] else 0.0
    tot_gt += gt_cnt[c]; tot_tp += tp_cnt[c]
    print("| %s | %d | %d | %.4f |" % (cls2name[c], gt_cnt[c], tp_cnt[c], rec), flush=True)
overall = tot_tp / tot_gt if tot_gt else 0.0
print("| **总体** | %d | %d | %.4f |" % (tot_gt, tot_tp, overall), flush=True)

bird_rec = tp_cnt[0] / gt_cnt[0] if gt_cnt[0] else 0.0
print("\nRESULT bird_recall=%.4f (GT=%d TP=%d) | overall_recall=%.4f | imgs=%d | score_thr=%.2f iou_thr=%.2f" % (
    bird_rec, gt_cnt[0], tp_cnt[0], overall, len(eval_img_ids), SCORE_THR, IOU_THR), flush=True)
print("=== DETECT BIRD RECALL 320 DONE ===", flush=True)
