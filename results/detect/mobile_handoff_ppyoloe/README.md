# Feeder 检测器 · ppyoloe_plus_crn_s（round3 M 线）· 移动端交接包（onnx / tflite）

喂食台粗检测模型（**PP-YOLOE+ crn_s，640，5 类**）的部署包。**同一权威模型两种格式，接口与 round2 NanoDet 包一致**——喂 BGR 像素、输出 `{label, score, box}`。挑一个格式用即可。

> 检测 5 类：`bird / squirrel / cat / person / other_animal`（行号 = id，见 `labels.txt`，与 round2 同序）。
> **这是 round3 M 线的 ppyoloe，比 round2 NanoDet 精度更高**（见文末对比），但更大更慢；是否替换 NanoDet 看帧率需求。

---

## 用哪个？

| 你的场景 | 用哪个 | 进哪个目录 |
|---|---|---|
| Android / iOS | **tflite**（`fp16` 移动端推荐，体积/内存减半）| [`round3/tflite/`](round3/tflite/) |
| onnxruntime / 板端 / 转别的格式 | **onnx** | [`round3/onnx/`](round3/onnx/) |

**两者同一模型、接口一致**，换格式只换 runtime，`ppyoloe_detect.py` 一行不改（已三方数值验证，见下）。

---

## 目录结构

**同类型 = 同接口**，`ppyoloe_detect.py` 对所有轮通用；**按 round 分层**（ppyoloe 首见于 round3）。

```
mobile_handoff_ppyoloe/
├── README.md              ← 本文件
├── labels.txt             ← 5 类名（行号 = id，与 round2 同序）
├── ppyoloe_detect.py      ← Python 参考（onnx + tflite 通用，含预处理+decode+NMS）
└── round3/
    ├── onnx/ppyoloe_s_640.onnx        · 29M，onnxruntime / 转格式源
    ├── tflite/
    │   ├── ppyoloe_s_640.fp16.tflite · 14M，移动端部署推荐（体积/内存减半）
    │   └── ppyoloe_s_640.fp32.tflite · 29M，最高精度、与 onnx 逐值一致
    ├── ckpt/                          · 续训权重 best_model.{pdparams,pdopt,pdema}（PaddleDet 2.6，见 ckpt/README）
    └── demos/                         · 5 类样例图（自测用）
```

---

## I/O 契约（最重要，与 round2 一致的部分）

**输入**：任意尺寸 **BGR 像素（0-255，`cv2.imread` 原样）**。`detect()` 内部做：resize **640×640（直接拉伸，非等比）** → BGR→RGB → **÷255** → NCHW。**预处理已封装在 `ppyoloe_detect.py`，调用方只管喂 BGR、拿检测。**

**输出**：检测列表，每条 `{"label": "bird", "score": 0.965, "box": [x1,y1,x2,y2]}`
- `box`：**原图像素坐标**（已缩放回你传入的尺寸）；`score`：0~1；**默认阈值 `conf=0.50, nms=0.50`（暂定）**——最终推荐值待 **0.4/0.5/0.6 性能测评**（precision/recall/对角权衡）后定，届时更新本包。
- **阈值是 CPU decode 侧的旋钮、不焊进模型**——ONNX/TFLite 输出的是原始分数,阈值在 `detect.py` 里施加。移动端/板端都可随时调（`PPYoloeDetector(model, conf=0.4)`），不用重导模型。

### ⚠️ 与 round2 NanoDet 的内部差异（已封装，调用方无感）

| | round2 NanoDet | 本包 ppyoloe |
|---|---|---|
| 输入尺寸 | 416 | **640** |
| 归一化 | BGR，mean/std 焊进图 | **RGB + ÷255**（在 detect.py 里做）|
| 模型原始输出 | 单张量 `[1,3598,37]` | **两个张量**：`boxes[1,8400,4]` ＋ `scores[1,5,8400]`（已 sigmoid）|
| decode/NMS | CPU | **CPU（在 detect.py 里，同样逻辑）** |

调用接口 `detect(image_bgr) -> [{label,score,box}]` **完全一致**,同事换模型只改模型路径。

### 🚨 自己用 Java / Kotlin / C++ 重写 decode 时——上面三处差异**不再"无感",必须逐一照抄 `ppyoloe_detect.py`**

> 若拿 nanodet 的 Java 移植当模板改本包,**接口一样但底层三处全不同**,是"nanodet 转换 OK、ppyoloe 一直不行"的根因。已用真模型逐一复现:

| # | nanodet 的做法（你可能照抄了） | ppyoloe 必须这样（否则的后果） |
|---|---|---|
| **1 预处理** | 喂 **BGR、0-255 原始像素**（归一化焊进图，啥都不用做）| 必须 **BGR→RGB + 每通道 ÷255.0**（归一化**没**焊进图）。**漏 ÷255 → 模型输出所有类分数塌到 ~0.01,`≥0.5` 一个都过不了 = 能跑但永远检不到**（实测检测数直接归零）。**这是最可能的病根。** |
| **2 模型输入** | **单输入** `image[1,3,416,416]` | **双输入**:`image[1,3,640,640]` **＋ `scale_factor[1,2] = [640/原图高, 640/原图宽]`**。模型内部用它把框缩回原图。**漏喂 → onnxruntime 直接报 "missing input scale_factor";乱填 [1,1] 或高宽写反 → 框坐标全错、越界到画面外**（非正方形图实测越界）。 |
| **3 模型输出** | **一个**张量 `[1,3598,37]`,要做**重型 DFL decode**（softmax 8bin＋anchor＋stride）| **两个独立张量**：`scores[1,5,8400]`（**已 sigmoid**）＋ `boxes[1,8400,4]`（**已是解好的 xyxy 框、已在原图坐标**）。**不要**套 nanodet 的 DFL/anchor;只需：`scores` **转置成 [8400,5]** → 每 anchor 取 max 类 → conf 过滤 → 逐类 NMS,框直接从 boxes 按同下标取。注意 scores 原始布局是 **[类,anchor]**,不转置 argmax 维度就错。 |

> **⚠️ 两个易错点(同事实际踩到的)**：
> 1. **runtime 输出带 batch 维**：本文档为简洁写 `scores[5,8400]`/`boxes[8400,4]`,但 onnxruntime/tflite 实际吐的是 **`[1,5,8400]` / `[1,8400,4]`**(前面多个 batch=1)。先 `[0]` 去掉 batch 再按上面处理。
> 2. **框在"另一个"输出里、不在 scores 里**：模型有**两个输出张量**。`[1,5,8400]` 只是分数(那"一大堆数"=5×8400=42000 个分)。**框是第二个输出 `[1,8400,4]`**,得单独接出来。配对规则:分数第 `i` 号候选框(anchor)的框 = **`boxes[0, i, :]` = `[x1,y1,x2,y2]`**,同下标一一对应,**已是原图坐标、不用再缩放/解码**。
>
> 语言无关伪代码(照抄即可)：
> ```
> scores = out_scores[0]          # [5,8400]  ← 去 batch
> boxes  = out_boxes[0]           # [8400,4]  ← 去 batch, 就是那个"另一个"输出
> dets = []
> for i in 0..8399:               # 遍历 8400 个候选框
>     c    = argmax over 5 类 of scores[:, i]   # 该框最强的类
>     conf = scores[c, i]                        # 已 sigmoid, 直接比阈值
>     if conf >= 0.5:
>         x1,y1,x2,y2 = boxes[i]                  # ← 关键: 同一个 i 去 boxes 取框
>         dets.push({label:NAMES[c], score:conf, box:[x1,y1,x2,y2]})
> dets = perClassNMS(dets, 0.5)   # 逐类 NMS
> ```

**一句话自查**（给移动端同事）：先打印模型**原始输出**分数峰值——
- 峰值全 `<0.05` → **预处理错(99% 是没 ÷255 或没转 RGB)**；
- onnxruntime 报缺输入 → **漏了 `scale_factor`**；
- 检到框但都跑到画面外 → **`scale_factor` 值/顺序错，或 `scores` 没转置**。

decode 的**权威逐行参照**就是本目录 `ppyoloe_detect.py`（`_preprocess` + `detect`,不到 60 行),照它翻译即可,别照 nanodet。

---

## 用法

```python
import cv2
from ppyoloe_detect import PPYoloeDetector

det = PPYoloeDetector("round3/tflite/ppyoloe_s_640.fp16.tflite")   # 或 round3/onnx/ppyoloe_s_640.onnx
for d in det.detect(cv2.imread("photo.jpg")):
    print(d)   # {'label': 'bird', 'score': 0.96, 'box': [x1,y1,x2,y2]}
```
命令行自测：`python ppyoloe_detect.py round3/onnx/ppyoloe_s_640.onnx round3/demos/demo_bird.jpg`

依赖：`onnxruntime`（onnx）或 `ai_edge_litert`/`tensorflow`（tflite）+ `opencv-python` + `numpy`。**Python ≥ 3.7**（脚本已 `from __future__ import annotations`，与 nanodet 包同样对 3.7/3.8/3.9+ 通用；早期版本曾用 `list[...]` 注解，在 Python 3.8 上 import 即报 `TypeError: 'type' object is not subscriptable`，已修）。

---

## 可靠性（三方交叉验证，同输入）

| 对比 | boxes | scores |
|---|---|---|
| ONNX vs Paddle 原模型 | cos 1.000000 (Δ6e-4) | cos 1.000000 (Δ2e-6) |
| TFLite fp32 vs ONNX | cos 1.000000 (Δ1e-3) | cos 1.000000 (Δ4e-6) |
| TFLite fp16 vs ONNX | cos 1.000000 (Δ0.7px) | cos 0.999998 (Δ9e-4) |

端到端（真图 bird）：三格式 bird 0.965、box IoU vs GT **0.99**、逐值一致。**Paddle→ONNX→TFLite 全链可靠。**

---

## 性能 / 精度（决策参考）

**V861 部署（AWNN 支持表）**：ppyoloe_s **4.24 fps / 235ms / 18.15MB NPU 内存**（NanoDet 更快 @416）。内存 fits V861；帧率是能否替换 NanoDet 的关键门槛。

**精度（同 test，vs round2 NanoDet P0）**：feeder 部署域整体 AP50 **90.5 vs 87.7（+2.8）**；相机陷阱域 **75.1 vs 65.5（+9.6）**，大胜在小/远目标。逐类同口径见 round3 飞书报告。
