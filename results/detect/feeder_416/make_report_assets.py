#!/usr/bin/env python
"""feeder_416 检测实验报告资产生成：CSV 机读指标 + loss 曲线解析 + 图表。
数据源：eval_results.txt / train_full.log / quant/*.log（均已 download 到本目录）。
图表用英文标签，避免 matplotlib 中文缺字。"""
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
QUANT = ROOT / "quant"

C = {
    "bird": "#d62728",
    "squirrel": "#ff7f0e",
    "cat": "#2ca02c",
    "person": "#9467bd",
    "other_animal": "#1f77b4",
}
FP32, INT8 = "#1f77b4", "#d62728"

# ------------------------------------------------------------------ 数据（从日志抽出，已核对）
EPOCHS = [10, 20, 30, 40, 50, 60, 70]
OVERALL = {  # COCO bbox, 逐 10ep（eval_results.txt）
    "mAP": [0.4086, 0.4572, 0.4680, 0.4789, 0.4880, 0.4958, 0.4978],
    "AP50": [0.6186, 0.6711, 0.6867, 0.6971, 0.7061, 0.7136, 0.7161],
    "AP75": [0.4356, 0.4940, 0.5047, 0.5176, 0.5291, 0.5339, 0.5407],
    "AP_small": [0.0099, 0.0146, 0.0158, 0.0162, 0.0161, 0.0169, 0.0166],
    "AP_medium": [0.0781, 0.0907, 0.0920, 0.1002, 0.1036, 0.1043, 0.1123],
    "AP_large": [0.4775, 0.5296, 0.5412, 0.5532, 0.5630, 0.5716, 0.5741],
}
PC_MAP = {  # 每类 mAP@.5:.95 (%) 逐 10ep（train_full.log per-class 表）
    "bird": [32.1, 35.2, 35.9, 36.4, 36.9, 37.2, 37.3],
    "squirrel": [51.2, 59.0, 59.4, 61.1, 62.8, 64.2, 64.1],
    "cat": [54.2, 60.9, 63.2, 64.4, 65.2, 65.8, 65.7],
    "person": [16.9, 18.9, 19.5, 19.9, 20.2, 20.8, 21.3],
    "other_animal": [49.9, 54.6, 56.1, 57.6, 58.9, 59.9, 60.5],
}
PC_AP50 = {
    "bird": [54.1, 58.5, 59.5, 60.6, 61.2, 61.6, 61.6],
    "squirrel": [73.3, 81.1, 82.8, 83.9, 85.5, 86.0, 86.4],
    "cat": [76.4, 82.4, 84.3, 85.2, 85.9, 86.7, 86.3],
    "person": [34.3, 37.5, 38.7, 39.5, 39.9, 41.0, 41.7],
    "other_animal": [71.2, 76.1, 78.1, 79.3, 80.6, 81.6, 82.1],
}
# 量化 fp32 -> int8（ORT-QDQ per-channel/opset13/calib120），best=ep70
QUANT_OVERALL = {  # (fp32, int8)
    "mAP": (0.4979, 0.4573),
    "AP50": (0.7161, 0.6762),
    "AP75": (0.5409, 0.4914),
    "AP_small": (0.0166, 0.0120),
}
QUANT_PC_MAP = {  # (fp32, int8) mAP %
    "bird": (37.28, 34.46),
    "squirrel": (64.12, 60.68),
    "cat": (65.73, 60.30),
    "person": (21.34, 18.70),
    "other_animal": (60.47, 54.50),
}
QUANT_PC_AP50 = {
    "bird": (61.64, 57.19),
    "squirrel": (86.40, 84.31),
    "cat": (86.26, 82.40),
    "person": (41.69, 37.36),
    "other_animal": (82.05, 76.82),
}
# 召回 (val 800 子集, conf>=0.3/IoU>=0.5/类别正确): GT, fp32 TP, int8 TP
RECALL = {
    "bird": (201, 184, 177),
    "squirrel": (114, 101, 96),
    "cat": (17, 15, 14),
    "other_animal": (530, 506, 471),
}
RECALL_OVERALL = (862, 806, 758)
# 320 vs 416 对比（feeder_320 README vs 本跑）
CMP = {
    "mAP (fp32)": (0.459, 0.4978),
    "AP50 (fp32)": (0.679, 0.7161),
    "bird mAP (fp32)": (0.341, 0.373),
    "bird recall (fp32)": (0.8756, 0.9154),
    "bird recall (int8)": (0.8408, 0.8806),
    "mAP (int8)": (0.413, 0.4573),
}


# ------------------------------------------------------------------ 解析 loss 曲线
def parse_loss(log_path):
    pat = re.compile(
        r"Epoch(\d+)/\d+\|Iter\d+\((\d+)/(\d+)\).*?"
        r"loss_qfl:([\d.]+).*?loss_bbox:([\d.]+).*?loss_dfl:([\d.]+)"
    )
    rows = []
    if not log_path.exists():
        return rows
    for line in log_path.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if not m:
            continue
        ep, it, tot, qfl, bbox, dfl = m.groups()
        ep, it, tot = int(ep), int(it), int(tot)
        step = (ep - 1) * tot + it
        rows.append((step, ep, float(qfl), float(bbox), float(dfl)))
    return rows


LOSS = parse_loss(ROOT / "train_full.log")


# ------------------------------------------------------------------ 写 CSV
def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


write_csv(
    ROOT / "overall_metrics.csv",
    ["epoch", "mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"],
    [[EPOCHS[i]] + [OVERALL[k][i] for k in ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]] for i in range(len(EPOCHS))],
)
write_csv(
    ROOT / "per_class_ap.csv",
    ["epoch", "class", "AP50", "mAP"],
    [[EPOCHS[i], c, PC_AP50[c][i], PC_MAP[c][i]] for i in range(len(EPOCHS)) for c in C],
)
if LOSS:
    write_csv(ROOT / "train_loss_curve.csv", ["step", "epoch", "loss_qfl", "loss_bbox", "loss_dfl"], LOSS)
write_csv(
    QUANT / "per_class_fp32_vs_int8.csv",
    ["class", "fp32_AP50", "int8_AP50", "drop_AP50", "fp32_mAP", "int8_mAP", "drop_mAP"],
    [[c, QUANT_PC_AP50[c][0], QUANT_PC_AP50[c][1], round(QUANT_PC_AP50[c][0] - QUANT_PC_AP50[c][1], 2),
      QUANT_PC_MAP[c][0], QUANT_PC_MAP[c][1], round(QUANT_PC_MAP[c][0] - QUANT_PC_MAP[c][1], 2)] for c in C],
)
write_csv(
    QUANT / "per_class_recall_fp32_vs_int8.csv",
    ["class", "GT", "fp32_TP", "fp32_recall", "int8_TP", "int8_recall", "drop"],
    [[c, RECALL[c][0], RECALL[c][1], round(RECALL[c][1] / RECALL[c][0], 4),
      RECALL[c][2], round(RECALL[c][2] / RECALL[c][0], 4),
      round((RECALL[c][1] - RECALL[c][2]) / RECALL[c][0], 4)] for c in RECALL]
    + [["overall", RECALL_OVERALL[0], RECALL_OVERALL[1], round(RECALL_OVERALL[1] / RECALL_OVERALL[0], 4),
        RECALL_OVERALL[2], round(RECALL_OVERALL[2] / RECALL_OVERALL[0], 4),
        round((RECALL_OVERALL[1] - RECALL_OVERALL[2]) / RECALL_OVERALL[0], 4)]],
)
print("CSV 写出完成")


# ------------------------------------------------------------------ 图表
def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig:", name)


# 1. 整体指标曲线
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for k, mk in [("AP50", "o-"), ("mAP", "s-"), ("AP75", "^-")]:
    ax.plot(EPOCHS, OVERALL[k], mk, label=k, linewidth=2)
ax.set_xlabel("epoch")
ax.set_ylabel("COCO AP")
ax.set_title("feeder_416 — Overall detection AP vs epoch (val 6633)")
ax.grid(alpha=0.3)
ax.legend()
for k in ("mAP", "AP50"):
    ax.annotate(f"{OVERALL[k][-1]:.3f}", (EPOCHS[-1], OVERALL[k][-1]), textcoords="offset points", xytext=(-6, 6), fontsize=9)
save(fig, "fig1_overall_curve.png")

# 2. 每类 mAP 曲线
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for c in C:
    ax.plot(EPOCHS, PC_MAP[c], "o-", color=C[c], label=c, linewidth=2)
ax.set_xlabel("epoch")
ax.set_ylabel("mAP@.5:.95 (%)")
ax.set_title("feeder_416 — Per-class mAP vs epoch")
ax.grid(alpha=0.3)
ax.legend()
save(fig, "fig2_perclass_map_curve.png")

# 3. loss 曲线
if LOSS:
    steps = [r[0] for r in LOSS]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(steps, [r[2] for r in LOSS], label="loss_qfl", alpha=0.8)
    ax.plot(steps, [r[3] for r in LOSS], label="loss_bbox", alpha=0.8)
    ax.plot(steps, [r[4] for r in LOSS], label="loss_dfl", alpha=0.8)
    ax.plot(steps, [r[2] + r[3] + r[4] for r in LOSS], label="total", color="k", linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"feeder_416 — Training loss ({len(LOSS)} logged points, 70 epochs)")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "fig3_loss_curve.png")

# 4. 量化 per-class mAP fp32 vs int8
import numpy as np

cls = list(C)
x = np.arange(len(cls))
w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar(x - w / 2, [QUANT_PC_MAP[c][0] for c in cls], w, label="fp32", color=FP32)
b2 = ax.bar(x + w / 2, [QUANT_PC_MAP[c][1] for c in cls], w, label="int8 (ORT-QDQ)", color=INT8)
ax.set_xticks(x)
ax.set_xticklabels(cls, rotation=12)
ax.set_ylabel("mAP@.5:.95 (%)")
ax.set_title("feeder_416 — Per-class mAP: FP32 vs INT8")
ax.grid(alpha=0.3, axis="y")
ax.legend()
for c, xi in zip(cls, x):
    ax.annotate(f"-{QUANT_PC_MAP[c][0]-QUANT_PC_MAP[c][1]:.1f}", (xi, max(QUANT_PC_MAP[c]) + 1), ha="center", fontsize=8, color="dimgray")
save(fig, "fig4_quant_perclass_map.png")

# 5. 召回 fp32 vs int8
cls_r = list(RECALL) + ["overall"]
fp32_r = [RECALL[c][1] / RECALL[c][0] * 100 for c in RECALL] + [RECALL_OVERALL[1] / RECALL_OVERALL[0] * 100]
int8_r = [RECALL[c][2] / RECALL[c][0] * 100 for c in RECALL] + [RECALL_OVERALL[2] / RECALL_OVERALL[0] * 100]
x = np.arange(len(cls_r))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - w / 2, fp32_r, w, label="fp32", color=FP32)
ax.bar(x + w / 2, int8_r, w, label="int8", color=INT8)
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n(GT={RECALL[c][0] if c in RECALL else RECALL_OVERALL[0]})" for c in cls_r])
ax.set_ylabel("recall (%)")
ax.set_ylim(0, 100)
ax.set_title("feeder_416 — Per-class recall (val 800 subset): FP32 vs INT8")
ax.axhline(88, ls="--", color="gray", alpha=0.6)
ax.grid(alpha=0.3, axis="y")
ax.legend()
for xi, v in zip(x, fp32_r):
    ax.annotate(f"{v:.1f}", (xi - w / 2, v + 1), ha="center", fontsize=8)
for xi, v in zip(x, int8_r):
    ax.annotate(f"{v:.1f}", (xi + w / 2, v + 1), ha="center", fontsize=8)
save(fig, "fig5_recall_fp32_vs_int8.png")

# 6. 320 vs 416 对比
keys = list(CMP)
v320 = [CMP[k][0] for k in keys]
v416 = [CMP[k][1] for k in keys]
x = np.arange(len(keys))
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.bar(x - w / 2, v320, w, label="feeder_320 (30ep)", color="#8c8c8c")
ax.bar(x + w / 2, v416, w, label="feeder_416 (70ep)", color="#2ca02c")
ax.set_xticks(x)
ax.set_xticklabels(keys, rotation=15, ha="right")
ax.set_ylabel("score (0-1)")
ax.set_title("Detection envelope — feeder_320 vs feeder_416")
ax.grid(alpha=0.3, axis="y")
ax.legend()
for xi, a, b in zip(x, v320, v416):
    ax.annotate(f"+{(b-a)*100:.1f}", (xi + w / 2, b + 0.01), ha="center", fontsize=8, color="green")
save(fig, "fig6_320_vs_416.png")

print("全部资产生成完成 ->", FIG)
