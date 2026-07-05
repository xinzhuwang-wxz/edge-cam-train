"""round2 训练+评测报告图表（PNG，配色与数据集报告一致，顶刊简洁风）→ report_charts/。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体（mac）
for f in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc"]:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150,
                     "font.size": 12, "axes.titleweight": "bold"})

COL = {"bird": "#3cc83c", "squirrel": "#f0a028", "cat": "#3c78f0", "person": "#f03c3c", "other_animal": "#9b59b6"}
TEAL, GREY, BLUE2 = "#0d7d72", "#888888", "#3c78f0"
OUT = os.path.join(os.path.dirname(__file__), "report_charts")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("→", name)


# ① 主模型训练曲线
bird = [80.8,81.8,80.5,80.5,83.8,80.3,86.4,86.5,87.2,88.1,89.5,89.3,89.4,90.0,90.6,90.5,91.1,91.3,91.7,91.5,91.6,91.5,91.8,91.8]
ap50 = [52.8,56.7,61.8,64.1,70.9,67.1,70.5,74.6,74.8,76.9,82.1,81.9,80.6,82.4,83.7,84.1,84.7,84.7,85.2,85.6,85.7,85.6,85.8,85.8]
ep = list(range(1, 25))
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(ep, bird, "-o", color=COL["bird"], lw=2.4, ms=5, label="bird（命门）")
ax.plot(ep, ap50, "--s", color=GREY, lw=1.6, ms=3.5, label="整体 5类")
ax.axhline(77.4, color="#c0392b", ls=":", lw=1.6)
ax.annotate("round1 基线 77.4", (1.2, 77.4), color="#c0392b", va="bottom", fontsize=10)
ax.annotate("91.8", (24, 91.8), color=COL["bird"], va="bottom", ha="right", fontweight="bold")
ax.annotate("85.8", (24, 85.8), color=GREY, va="bottom", ha="right", fontsize=9)
ax.set(xlabel="Epoch", ylabel="AP50 (%)", title="主模型训练曲线：NanoDet-Plus-m-416, 1.0x, 24ep",
       xlim=(0.5, 24.5), ylim=(48, 95))
ax.legend(loc="lower right", frameon=False)
save(fig, "01_train_curve.png")

# ② 数据量 scaling（验证 bird AP50）
pct = [12.5, 25, 50, 100]
d_val = [84.3, 87.1, 89.7, 91.8]
fig, ax = plt.subplots(figsize=(5.8, 4.2))
ax.plot(pct, d_val, "-o", color=COL["bird"], lw=2.6, ms=8)
for x, y in zip(pct, d_val):
    ax.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10, color=COL["bird"])
ax.set(xlabel="训练数据量 (%)", ylabel="验证 bird AP50 (%)", title="数据量 scaling（1.0x 固定）",
       xticks=pct, ylim=(82, 94))
ax.annotate("单调升、边际递减\n（50%→100% +2.1pt）", (38, 90.5), fontsize=10, color=GREY)
save(fig, "02_scaling_data.png")

# ③ 参数 scaling（验证 bird AP50）
w = [0.5, 1.0, 1.5]
p_val = [87.6, 91.8, 91.6]
fig, ax = plt.subplots(figsize=(5.8, 4.2))
ax.plot(w, p_val, "-o", color=COL["bird"], lw=2.6, ms=8)
ax.plot(1.5, 91.9, "D", color="#e67e22", ms=9, label="A: full-COCO 1.5x")
for x, y in zip(w, p_val):
    ax.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10, color=COL["bird"])
ax.set(xlabel="模型宽度倍率", ylabel="验证 bird AP50 (%)", title="参数 scaling（100% 数据固定）",
       xticks=w, ylim=(85, 94))
ax.annotate("1.0x 甜点\n（1.5x 不再涨 · 0.5x 掉 4pt）", (0.72, 88.3), fontsize=10, color=BLUE2)
ax.legend(loc="lower right", frameon=False, fontsize=9)
save(fig, "03_scaling_param.png")

# ④ 416 vs 320（逐类，验证集）
cls = ["bird", "squirrel", "cat", "person", "other_animal"]
v416 = [91.8, 90.7, 95.2, 55.1, 96.0]
v320 = [90.6, 90.0, 95.1, 52.1, 94.9]
import numpy as np
x = np.arange(len(cls)); bw = 0.38
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.bar(x - bw/2, v416, bw, label="416", color=[COL[c] for c in cls])
ax.bar(x + bw/2, v320, bw, label="320", color=[COL[c] for c in cls], alpha=0.5, hatch="//")
ax.annotate("bird 仅 -1.2", (0, 92.5), fontsize=9, color=COL["bird"], ha="center")
ax.set(ylabel="验证 AP50 (%)", title="416 vs 320 分辨率消融（验证集）",
       xticks=x, ylim=(0, 108))
ax.set_xticklabels(cls, rotation=12)
ax.legend(frameon=False)
save(fig, "04_res_416_320.png")

# ⑤ worksite 逐类 AP30/50
w50 = [94.9, 79.3, 89.0, 92.0, 92.6]
w30 = [95.8, 83.2, 89.2, 93.0, 94.4]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.bar(x - bw/2, w30, bw, label="AP30", color=[COL[c] for c in cls], alpha=0.55)
ax.bar(x + bw/2, w50, bw, label="AP50", color=[COL[c] for c in cls])
for i, (a, b) in enumerate(zip(w30, w50)):
    ax.annotate(f"{b}", (i + bw/2, b + 1), ha="center", fontsize=8.5)
ax.set(ylabel="AP (%)", title="观鸟器工作场景集 · 逐类 AP（main 模型）",
       xticks=x, ylim=(0, 105))
ax.set_xticklabels(cls, rotation=12)
ax.legend(frameon=False)
save(fig, "05_worksite_ap.png")

# ⑥ bird 按尺度（工作场景集）
sizes = ["medium\n(32²–96²)", "large\n(>96², 工作点)", "all"]
w_size = [69.7, 98.5, 94.9]
xs = np.arange(len(sizes))
fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.bar(xs, w_size, 0.55, color=COL["bird"])
for i, v in enumerate(w_size):
    ax.annotate(f"{v}", (i, v + 1.8), ha="center", fontsize=11, color=COL["bird"], fontweight="bold")
ax.set(ylabel="bird AP50 (%)", title="bird 按尺度（工作场景集）：大目标近满分", xticks=xs, ylim=(0, 108))
ax.set_xticklabels(sizes)
save(fig, "06_bird_by_size.png")

# ⑦ 负样本误报率
thr = [0.30, 0.40, 0.50]
rate = [17.5, 7.5, 3.3]
fig, ax = plt.subplots(figsize=(5.4, 4.0))
ax.plot(thr, rate, "-o", color="#c0392b", lw=2.4, ms=7)
for x0, y0 in zip(thr, rate):
    ax.annotate(f"{y0}%", (x0, y0), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax.set(xlabel="置信度阈值", ylabel="空帧误报率 (%)", title="负样本误报（120 空帧）",
       xticks=thr, ylim=(0, 20))
ax.annotate("部署建议 ≥0.4\n+ 时序滤波", (0.4, 7.5), xytext=(0.42, 13), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=1))
save(fig, "07_negatives.png")

# ⑧ 每类框数（train+val）—— 替换 §1 的 train+val vs test 图
cb = [("bird", 16802), ("other_animal", 8509), ("person", 8316), ("cat", 3814), ("squirrel", 3429)]
cn = [c[0] for c in cb]; cv = [c[1] for c in cb]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
bars = ax.bar(cn, cv, color=[COL[n] for n in cn], width=0.62)
for b, v in zip(bars, cv):
    ax.annotate(f"{v:,}", (b.get_x() + b.get_width() / 2, v + 250), ha="center", fontsize=11, fontweight="bold")
ax.set(ylabel="框数", title="每类框数（train+val 共 40,870 框）", ylim=(0, 18500))
ax.set_xticks(range(len(cn))); ax.set_xticklabels(cn, rotation=10)
save(fig, "class_boxes_trainval.png")

print("\n全部图表 →", OUT)
