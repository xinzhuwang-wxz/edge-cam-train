# Feeder 检测器 · 移动端交接包（ncnn / onnx / tflite）

喂食台粗检测模型（NanoDet-Plus-m 416, 5 类）的部署包。**同一个权威模型三种格式，接口完全一致**——喂 0-255 BGR、输出概率、**同一套 decode**。挑一个格式用即可。

> **⚠️ 布局说明（round2 与 round3 的唯一区别）**：本包重组后，**onnx/tflite 的 Python decode 参考已上移到上层 `../nanodet_detect.py`（round2/round3 通用，本目录不再单独放 `feeder_detect_tflite.py`）**。跑 round2 的 onnx/tflite 用 `python ../nanodet_detect.py round2/onnx/feeder_416.onnx <img>`（从包根 `mobile_handoff_nanodet/` 跑）——已实测 round2 模型经该 decoder onnx≡tflite（bird 0.9582 逐值一致）。**ncnn 是 round2 独有的 C++ 全套**（自带 `ncnn/feeder_detect_ref.py` + C++，因 ncnn 是独立 runtime 无法与 onnx/tflite 共用 Python runner）；round3 精简掉了 ncnn，只出 onnx/tflite。

> 检测 5 类：`bird / squirrel / cat / person / other_animal`。命门 = bird（喂食台观鸟），固定 held-out test **bird AP50 ≈ 85.0**（大目标 90.1）。

---

## 用哪个？（先看这里）

| 你的场景 | 用哪个 | 进哪个目录 |
|---|---|---|
| Android / iOS，想省掉 ncnn 的 OpenMP 编译 | **tflite**（推荐）| [`tflite/`](tflite/) |
| 已有 ncnn 工具链 / 要 C++ 源码 | **ncnn** | [`ncnn/`](ncnn/) |
| 板端 onnxruntime / 想自己转别的格式 | **onnx** | [`onnx/`](onnx/) |

**三者同一模型、接口一致**（喂 0-255 BGR resize 416 → 输出概率 `[3598,37]` → 同一 decode/NMS），换格式只换 runtime、decode 代码一行不改。选一个即可，不用三个都拿。

---

## 目录结构

```
mobile_handoff_nanodet/           ← 包根（onnx/tflite decode 参考在此）
├── nanodet_detect.py             ← ★ onnx/tflite 通用 decode+NMS（round2/round3 共用）
├── labels.txt                    ← 5 类名（行号 = 类别 id，各格式共用）
└── round2/                       ← 本目录
    ├── README.md                 ← 本文件
    ├── labels.txt                ← 5 类名（同上序）
    ├── tflite/                   ← ① TFLite（移动端推荐；decode 用 ../nanodet_detect.py）
    │   ├── feeder_416.fp16.tflite    ·  2.4M，移动端部署用（体积/内存减半、更快）
    │   └── feeder_416.fp32.tflite    ·  4.6M，最高精度、与 onnx 逐字一致
    ├── ncnn/                     ← ② NCNN（C++ 全套，round2 独有）
    │   ├── nanodet_feeder5_mobile_416.param / .bin
    │   ├── feeder_detector.h / .cpp  ·  C++ 核心：detect() 含 decode+NMS
    │   ├── demo_main.cpp         ·  独立 demo（读图→detect→打印→画框）
    │   ├── CMakeLists.txt
    │   └── feeder_detect_ref.py  ·  Python 参考（ncnn runtime 专用）
    ├── onnx/
    │   └── feeder_416.onnx       ·  4.7M，onnxruntime / 板端 / 转格式源
    └── demos/                    ← 样例图 + tflite 画框结果（tflite_out/）
```

---

## I/O 契约（三格式共用，最重要）

**输入**：**任意尺寸**的 BGR 像素（0-255）→ 内部 resize **416×416（直接拉伸，非等比）**，box 自动缩放回原图。**sigmoid + 归一化已焊进图，不用自己做 mean/std**。
> 图片文件格式（jpg/png/webp…）由**调用方解码**——detect 吃的是**解码后的 BGR 像素、与文件格式无关**（python 参考用 `cv2.imread` 支持常见格式；app 用 Android Bitmap / iOS CVPixelBuffer 等解码，注意 **RGBA→BGR**）。分辨率任意，已测 1280² / 640² / 500×376。

**输出**：检测列表，每条 `{ "label": "bird", "score": 0.87, "box": [x1,y1,x2,y2] }`
- `label`：5 类之一；`score`：0~1（GFL 类别分即置信度）；`box`：**原图像素坐标**（已缩放回你传入的尺寸）。
- 默认阈值 `conf=0.40, nms=0.50`。
- C++ 对应 `struct Detection { int label; float score; float x1,y1,x2,y2; }`。

---

## 各格式用法

### ① tflite/

```bash
pip install ai-edge-litert opencv-python numpy      # 或 tflite_runtime / tensorflow
# 从包根 mobile_handoff_nanodet/ 跑上层通用 decoder（onnx/tflite 同一份）：
python nanodet_detect.py round2/tflite/feeder_416.fp32.tflite round2/demos/demo_bird.jpg
```
- **fp16 vs fp32**：`fp16`（2.4M）权重半精度、体积/内存减半、ARM 上更快、精度近无损（检测框 ±1px）——**移动端部署用它**；`fp32`（4.6M）最高精度、与 onnx 逐字一致——作精度对照/验收。
- **移动端集成**：Android 用 `org.tensorflow:tensorflow-lite` / LiteRT（可选 NNAPI/GPU delegate），iOS 用 `TensorFlowLiteC` / LiteRT framework，**无 OpenMP 依赖**。decode/NMS 见上层 `../nanodet_detect.py`（onnx/tflite 通用；与 ncnn 版逐行相同），移植到 Kotlin/Swift 即可。

### ② ncnn/

C++ 三行接入：
```cpp
FeederDetector det;
det.load("nanodet_feeder5_mobile_416.param", "nanodet_feeder5_mobile_416.bin", /*use_fp16=*/true);
std::vector<Detection> results = det.detect(bgr_bytes, width, height);   // 每帧调
```
- **依赖 ncnn**：官方预编译库 <https://github.com/Tencent/ncnn/releases>（iOS `*-ios.zip` / Android `*-android-vulkan.zip`），BSD 许可、商用无碍。
- **编译 demo**：`cd ncnn && mkdir build && cd build && cmake .. && make && cd .. && ./build/feeder_demo ../demos/demo_bird.jpg 0.4 0.5 out.jpg`
- **Python 参考**：`python ncnn/feeder_detect_ref.py demos/demo_bird.jpg 0.4 0.5 out.jpg`
- iOS/Android 集成细节见 `ncnn/feeder_detector.cpp` 注释；NanoDet 官方 Android demo 结构可对照（把 `nanodet.cpp` 换成本 `feeder_detector.cpp`）。

### ③ onnx/

`feeder_416.onnx`：onnxruntime 直接跑（输入 `[1,3,416,416]` NCHW 0-255 BGR，输出概率 `[1,3598,37]`），或作为转其它格式的源。decode 与 ncnn/tflite 相同。

---

## decode 内部格式（自己写 decode / 排查时看）

输出 `[3598,37]`：**3598** = 4 个 stride 特征图 anchor 数 52²(s8)+26²(s16)+13²(s32)+7²(s64)；**37** = `[5 类概率] + [32 框分布]`；**32 = DFL**：4 条边(l,t,r,b)×8 bin。
```
中心 = (x*stride, y*stride)                    # 注意无 +0.5 偏移
每条边: dist = Σ softmax(8 bin)·[0..7]·stride
框 = [cx-l, cy-t, cx+r, cy+b] (416 空间) → ×(原图/416) 缩放回原图 + 裁剪
score,label = max/argmax(5 类)  → 阈值过滤 → 逐类 NMS(stable sort)
```
三份参考（`ncnn/feeder_detector.cpp` C++、`ncnn/feeder_detect_ref.py` ncnn 版、上层 `../nanodet_detect.py` onnx/tflite 版）decode **逐行相同**。

---

## 格式对齐验证（9 张图、同一 decode、都喂 0-255）

- **tflite-fp32 ≡ onnx**：原始张量 `max|Δ|≈1e-5`，检测框 **100% 一致**；
- **ncnn ≡ onnx/tflite**：检测框一致（个别框 ±1px 舍入）；
- **tflite-fp16 ≈ onnx**：`max|Δ|≈2e-2`（fp16），检测框一致（极少 ±1px）。
- 诚实说明：跨 runtime **非严格 bit-exact**（各自浮点/SIMD 实现不同），但 fp32 对齐到 `1e-5`、三方检测框全一致——产品级完全等价。
- `demos/tflite_out/` 是 tflite 跑 6 张 demo 的画框结果，可直接看（与 ncnn 逐字一致）。

---

## 性能 / 边界（诚实交代）

| 场景 | bird AP50 | 说明 |
|---|---|---|
| 固定 held-out test（泛化真值）| **85.0**（大目标 90.1）| 整体 AP50 0.655 |
| 观鸟器工作场景集（部署同款，乐观）| **94.9**（大目标 98.5）| 含泄漏，乐观上界 |
| 320 版（省 ~1.7× 算力）| 83.1 | -1.9pt，命门仍守（需要可另出 320 格式）|

- **NMS 和画框不在模型里**：NMS 在 `detect()`，画框在你的 app（demo 里有示例）。
- **空帧误报**：conf 0.4 时约 7.5% 帧有误报 → 部署阈值 ≥0.4 + 时序滤波（连续 N 帧才告警）。
- **非鸟类弱于 bird**（训练数据 bird 最富）：产品命门是 bird，其余"检到即可"。`cat` 训练含 camera-trap 山猫；`person` 数据稀缺偏弱。
- 模型对**中大目标**最好（喂食台工作点）；很小/很远目标不是设计目标。
