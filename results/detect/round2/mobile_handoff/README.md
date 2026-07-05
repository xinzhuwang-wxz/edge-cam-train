# Feeder 检测器 · 移动端交接包（NCNN）

喂食台粗检测模型（NanoDet-Plus-m, 5 类）的移动端（iOS + Android）部署包。
**给移动同事：你只需要一个函数 `detect(图) → [{label, score, box}]`**，本包已把「预处理 + 推理 + decode + NMS」封装好，照 `feeder_detector.{h,cpp}` 集成即可。

> 检测 5 类：`bird / squirrel / cat / person / other_animal`。命门是 bird（喂食台观鸟），固定 test 上 bird AP50 ≈ 85（大目标 90）。

---

## 1. 包内文件

| 文件 | 说明 |
|---|---|
| `nanodet_feeder5_mobile_416.param` / `.bin` | **ncnn 模型**（2.3MB）。sigmoid + 归一化已焊进图 |
| `feeder_detector.h` / `.cpp` | **C++ 核心**（iOS/Android 共用）。含 decode + NMS，对外 `detect()` |
| `feeder_detect_ref.py` | **Python 参考实现**（可跑，输出 JSONL + 画框）。对照移植 / 验收用 |
| `labels.txt` | 类别名（行号=类别 id）|
| `demo_bird.jpg` | 跑通样例（画框图）|

---

## 2. I/O 契约（最重要）

**输入**：任意尺寸 BGR 图（0-255）。内部会 resize 到 **416×416（直接拉伸，非等比）**，归一化模型已焊，**不用自己做 mean/std**。

**输出**：检测列表，每条：
```json
{ "label": "bird", "score": 0.87, "box": [x1, y1, x2, y2] }
```
- `label`：5 类之一。
- `score`：置信度 0~1（GFL 设计下类别分即置信度，无独立 objectness）。
- `box`：**原图像素坐标**（左上 x1,y1 / 右下 x2,y2），已缩放回你传入的图尺寸。

C++ 里对应 `struct Detection { int label; float score; float x1,y1,x2,y2; }`。
默认阈值 `conf=0.40, nms=0.50`，可调。

---

## 3. C++ API（三行接入）

```cpp
FeederDetector det;
det.load("nanodet_feeder5_mobile_416.param", "nanodet_feeder5_mobile_416.bin", /*use_fp16=*/true);
std::vector<Detection> results = det.detect(bgr_bytes, width, height);   // 每帧调
for (auto& d : results)
    printf("%s %.2f [%.0f,%.0f,%.0f,%.0f]\n",
           FeederDetector::label_name(d.label), d.score, d.x1, d.y1, d.x2, d.y2);
```

---

## 4. 集成

### 依赖 ncnn
- 拿官方预编译库：<https://github.com/Tencent/ncnn/releases>（iOS 用 `*-ios.zip`，Android 用 `*-android-vulkan.zip`）。
- ncnn 是 BSD 许可，商用无碍。

### iOS
1. Xcode 里加入 `ncnn.framework`（+ `openmp.framework`，官方包内有）。
2. `feeder_detector.cpp/.h` 直接加进工程（文件设为 **Objective-C++**，或建一个 `.mm` 桥接）。
3. Swift 侧建一个 Obj-C++ 封装暴露 `-(NSArray*)detectPixelBuffer:...`，内部把 `CVPixelBuffer`/`UIImage` 转成 BGR 连续字节喂 `detect()`。
4. 模型 `.param/.bin` 放进 app bundle，`load()` 用 bundle 路径。

### Android
1. `app/src/main/cpp/` 放入 `feeder_detector.cpp/.h`，`CMakeLists.txt` 链接 ncnn（用官方 `.aar` 或预编译 `.so`）。
2. 写一小段 JNI：`Java_..._detect(env, thiz, byte[] bgr, int w, int h)` → 调 `detect()` → 组装成 Java 对象/JSON 返回。
3. Kotlin 侧把 `Bitmap` 转 BGR 字节（注意 Android 常是 ARGB，需转 BGR）传下去。
4. 模型放 `assets/`，首次拷到内部存储再 `load()`（ncnn 也可直接从 AAsset 读）。

> **参考**：NanoDet 官方 Android demo 结构与此一致（<https://github.com/RangiLyu/nanodet> 的 `demo_android_ncnn`），可直接对照——把它的 `nanodet.cpp` 换成本 `feeder_detector.cpp` 即可。

---

## 5. 性能 / 调优

- **fp16**：`load(..., use_fp16=true)`（默认）→ 内存减半、ARM 上更快、精度近无损。移动端建议开。
- **GPU(Vulkan)**：`feeder_detector.cpp` 里放开 `use_vulkan_compute`（需 ncnn-vulkan 包）。小模型 CPU 通常已够快。
- **线程**：`net_.opt.num_threads` 可设（默认用大小核全部）。
- **输入**：416×416；若要更省算力，边侧另有 320 版在评估（部署默认），需要时可再出一份 320 的 ncnn。

---

## 6. 模型输出内部格式（想自己写 decode / 排查时看）

原始输出张量 `[1, 3598, 37]`：
- **3598** = 4 个 stride 特征图的 anchor 总数：52²(s8)+26²(s16)+13²(s32)+7²(s64)。
- **37** = `[5 类概率] + [32 框分布]`。
- **32 框分布 = DFL**：4 条边(left,top,right,bottom) × 8 bin。每条边不是坐标，而是「到 anchor 中心距离」的离散分布。

decode（已在 `.cpp`/`.py` 实现，此处仅供理解）：
```
中心 = (x*stride, y*stride)          # 注意无 +0.5 偏移
对每条边: dist = Σ softmax(8 bin) · [0..7] · stride
框 = [cx-l, cy-t, cx+r, cy+b]  (416 空间) → ×(原图/416) 缩放回原图
score,label = max/argmax(5 类概率)
→ 阈值过滤 → 逐类 NMS
```
详见根仓 `results/detect/round2/训练监控日志.md` 与《为什么是 32》说明。

---

## 7. 边界 / 已知事项（诚实交代）

- **NMS 和画框不在模型里**，在 `detect()`（NMS）和你的 app（画框）。这是移动端检测的标准做法（ncnn 无法把 NMS 稳定塞进图）。
- **非鸟类（squirrel/cat/person）在 test 上明显弱于 bird**（泛化落差，训练数据 bird 最富）。产品命门是 bird，其余当"检到即可"用，别当高精度依赖。
- 模型对**中大目标**最好（喂食台工作点）；很小/很远的目标不是设计目标。

---

## 8. Python 参考跑法（验收 / 对照）

```bash
pip install ncnn opencv-python numpy
python feeder_detect_ref.py <图片> 0.4 0.5 out.jpg    # 打印 JSONL + 存画框图 out.jpg
```
移动端移植后，用同一批图对比 C++ 与本参考的输出应一致（±像素级）。

---

## 9. 验证证据（已通过）

- **C++ ≡ Python 参考**：多张密集多目标图（GT 15~31 框）× 阈值 {0.15, 0.40}，检测**集合 100% 一致**（含 0 检出空图边界）。对齐要点：decode→缩放到原图+裁剪→原图空间 NMS（stable sort）。
- **decode 正确性**：固定 test 上 pred-vs-GT IoU——满框鸟 0.995 / 局部鸟 0.951 / 中目标 0.89~0.92，框随目标精确定位。
- **ONNX→ncnn 转换**：与 onnxruntime 输出 max|Δ|≈0.03（SIMD 数值噪声，检测无感）。
- **示例**：`demos/`（4 张自然鸟 + squirrel/cat/person/other_animal 各一张，观鸟器视角）。

## 10. 性能参考

| 场景 | bird AP50 | 说明 |
|---|---|---|
| 固定 held-out test（泛化真值）| **85.0**（大目标 90.1）| 整体 AP50 0.655 |
| 观鸟器工作场景集（部署同款，乐观）| **94.9**（大目标 98.5）| 全类 79~95（含泄漏，乐观上界）|
| 320 版（省 ~1.7× 算力）| 83.1 | -1.9pt，命门仍守 |

- **空帧误报**：conf 0.4 时约 7.5% 帧有误报 → 部署阈值≥0.4 + 时序滤波（连续 N 帧才告警）压假警。
- **类别注意**：`cat` 训练含 camera-trap 山猫（非纯家猫）；`person` 数据稀缺、偏弱。
