#!/usr/bin/env python
"""分类实验报告资产：解析 V1/V2 训练曲线 + 画 训练对比 / envelope对比 / 鸟框占比 图。
英文标签避免 matplotlib 缺字。"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def parse_val(csv_path):
    """metrics.csv → {epoch: (val_top1, val_top5)}（val 行=train 列空）。"""
    out = {}
    with open(csv_path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r.get("val_top1"):
                ep = int(r["epoch"])
                out[ep] = (float(r["val_top1"]), float(r["val_top5"]))
    return out


V1 = parse_val(ROOT / "v1_fullimage" / "metrics_v2.csv")  # 整图
V2 = parse_val(ROOT / "v2_crop" / "metrics_cropv2.csv")  # 检测裁框

# envelope (top1, top5)
ENV = {
    "V1 full-image": {"fp32": (0.596, 0.777), "int8": (0.567, 0.762), "field": (0.370, 0.594)},
    "V2 detector-crop": {"fp32": (0.748, 0.866), "int8": (0.750, 0.867), "field": (0.626, 0.797)},
}
# 6 张样本鸟框占画面比例（裁框动机）
COVERAGE = [
    ("robin", 3), ("mallard", 11), ("greylag", 16), ("buzzard", 27),
    ("canada goose", 30), ("grey heron", 38),
]
C1, C2 = "#8c8c8c", "#2ca02c"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig:", name)


# fig1: 训练 val_top1 曲线 V1 vs V2
fig, ax = plt.subplots(figsize=(7.5, 4.6))
e1 = sorted(V1)
e2 = sorted(V2)
ax.plot(e1, [V1[e][0] for e in e1], "o-", color=C1, label="V1 full-image  top1", lw=2)
ax.plot(e2, [V2[e][0] for e in e2], "s-", color=C2, label="V2 detector-crop  top1", lw=2)
ax.plot(e1, [V1[e][1] for e in e1], "o--", color=C1, alpha=0.5, label="V1 top5")
ax.plot(e2, [V2[e][1] for e in e2], "s--", color=C2, alpha=0.5, label="V2 top5")
ax.axhline(max(V1[e][0] for e in e1), ls=":", color=C1, alpha=0.6)
ax.set_xlabel("epoch")
ax.set_ylabel("val accuracy")
ax.set_title("Classify val accuracy — full-image (V1) vs detector-crop (V2)")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
b1, b2 = max(V1[e][0] for e in e1), max(V2[e][0] for e in e2)
ax.annotate(f"{b1:.3f}", (e1[-1], b1), textcoords="offset points", xytext=(-4, -14), color=C1, fontsize=9)
ax.annotate(f"{b2:.3f}  (+{(b2 - b1) * 100:.0f}pt)", (e2[-1], b2), textcoords="offset points", xytext=(-50, 6), color=C2, fontsize=10, fontweight="bold")
save(fig, "fig1_train_v1_vs_v2.png")

# fig2: envelope 对比（top1 grouped bar：fp32/int8/field）
levels = ["fp32", "int8", "field"]
x = np.arange(len(levels))
w = 0.38
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.bar(x - w / 2, [ENV["V1 full-image"][lv][0] for lv in levels], w, label="V1 full-image", color=C1)
ax.bar(x + w / 2, [ENV["V2 detector-crop"][lv][0] for lv in levels], w, label="V2 detector-crop", color=C2)
ax.set_xticks(x)
ax.set_xticklabels(["fp32 (val)", "int8 sim (test)", "field degraded"])
ax.set_ylabel("top-1")
ax.set_ylim(0, 0.85)
ax.set_title("Classify envelope top-1 — V1 vs V2 (INT8 is the edge number)")
ax.grid(alpha=0.3, axis="y")
ax.legend()
for xi, lv in zip(x, levels):
    a, b = ENV["V1 full-image"][lv][0], ENV["V2 detector-crop"][lv][0]
    ax.annotate(f"{a:.3f}", (xi - w / 2, a + 0.01), ha="center", fontsize=8)
    ax.annotate(f"{b:.3f}\n+{(b - a) * 100:.0f}pt", (xi + w / 2, b + 0.01), ha="center", fontsize=8, color="green")
save(fig, "fig2_envelope_v1_vs_v2.png")

# fig3: 鸟框占画面比例（裁框动机）
fig, ax = plt.subplots(figsize=(7.5, 3.8))
names = [c[0] for c in COVERAGE]
vals = [c[1] for c in COVERAGE]
bars = ax.barh(names, vals, color=["#d62728" if v < 15 else "#1f77b4" for v in vals])
ax.set_xlabel("bird box area as % of full image")
ax.set_title("Why crop: the bird often fills only a small part of the photo")
ax.axvline(100, ls=":", color="gray")
for b, v in zip(bars, vals):
    ax.annotate(f"{v}%", (v + 1, b.get_y() + b.get_height() / 2), va="center", fontsize=9)
ax.set_xlim(0, 45)
save(fig, "fig3_bird_coverage.png")

# fig4: 地域 mask off vs on（按 coverage 排序，看"区域越窄增益越大"）
import json

reg_path = ROOT / "regional" / "regional_results.json"
if reg_path.exists():
    reg = json.loads(reg_path.read_text())
    reg.sort(key=lambda r: -r["coverage"])  # 宽→窄
    labels_r = [f"{r['region']}\ncov {r['coverage']:.2f}\n({r['region_classes']}/360)" for r in reg]
    xr = np.arange(len(reg))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(xr - w / 2, [r["top1_off"] for r in reg], w, label="mask OFF", color="#8c8c8c")
    ax.bar(xr + w / 2, [r["top1_on"] for r in reg], w, label="mask ON (regional)", color="#1f77b4")
    ax.set_xticks(xr)
    ax.set_xticklabels(labels_r, fontsize=9)
    ax.set_ylabel("in-region top-1")
    ax.set_ylim(0.7, 0.85)
    ax.set_title("Regional mask gain scales with how much the region narrows candidates")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    for xi, r in zip(xr, reg):
        ax.annotate(f"+{r['gain'] * 100:.1f}pt", (xi + w / 2, r["top1_on"] + 0.004), ha="center", fontsize=9, color="#1f77b4", fontweight="bold")
    save(fig, "fig4_regional_mask.png")

# CSV：envelope 对比机读
with open(ROOT / "envelope_v1_vs_v2.csv", "w", newline="") as f:
    w_ = csv.writer(f)
    w_.writerow(["level", "V1_top1", "V1_top5", "V2_top1", "V2_top5", "delta_top1"])
    for lv in levels:
        a = ENV["V1 full-image"][lv]
        b = ENV["V2 detector-crop"][lv]
        w_.writerow([lv, a[0], a[1], b[0], b[1], round(b[0] - a[0], 4)])
print("V1 best top1:", round(b1, 4), "V2 best top1:", round(b2, 4))
print("done ->", FIG)
