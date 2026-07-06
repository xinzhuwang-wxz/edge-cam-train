# AGENTS.md — Feeder 检测器移动端集成（机器可读规格）

> 面向 **AI Agent**：本包给 iOS/Android 集成一个 5 类目标检测器（NanoDet-Plus-m，NCNN）。
> 按本规格集成即可，**不要重训、不要改模型**。人读版见 `README.md`。

## 身份
- 任务：粗检测 5 类 `["bird","squirrel","cat","person","other_animal"]`（下标=类别 id）。
- 命门=bird（观鸟器）。其余类仅供"检到即抑制告警"，精度不作强依赖。
- 运行时：NCNN（BSD-3，商用 OK）。模型 2.3MB。
- **三格式接口一致**：本包另有 `onnx/`、`tflite/`（同一模型、同 I/O、同 decode）。本规格是 **ncnn 视角**（C++），下列文件均在 `ncnn/` 目录；tflite 集成见 `README.md`。

## 文件契约
| 文件 | 角色 | 不可变 |
|---|---|---|
| `nanodet_feeder5_mobile_416.param`/`.bin` | NCNN 模型；**sigmoid + 归一化已焊入图** | 是 |
| `feeder_detector.h`/`.cpp` | C++ 核心，iOS/Android 共用；`detect()` 内含 decode+NMS | 移植目标 |
| `feeder_detect_ref.py` | Python 参考（等价实现），用于对拍验收 | 参考 |
| `labels.txt` | 类名，行号=id | 是 |
| `demo_main.cpp`/`CMakeLists.txt` | 桌面独立 demo | 参考 |

## I/O 契约（严格）
- **输入**：BGR、0–255、任意 W×H 的连续像素。
  - ⚠ **BGR 不是 RGB**（Android Bitmap 常是 ARGB/RGBA → 必须转 BGR）。
  - ⚠ **不要自己做 mean/std 归一化**（已焊进模型）。喂原始像素。
- **内部**：resize 到 **416×416，直接拉伸（非等比、非 letterbox、无 padding）**。
- **输出**：检测数组，每条 `{label:str, score:float∈[0,1], box:[x1,y1,x2,y2]}`。
  - `box` = **原图像素坐标**（已用 sx=W/416, sy=H/416 独立缩放回原图 + 裁剪到 [0,W]×[0,H]）。可直接画到原图。
  - 空数组 `[]` = 无检出。按 score 降序。
- 默认阈值 `conf=0.40, nms=0.50`（可调）。

## C++ API
```cpp
FeederDetector det;
det.load("...param","...bin", /*use_fp16=*/true);          // 一次
std::vector<Detection> r = det.detect(bgr, W, H, 0.40f, 0.50f);  // 每帧
// Detection{int label; float score; float x1,y1,x2,y2;}
```

## 模型输出张量（若需自写/排查 decode）
- 原始输出 `out0`：shape `[1, 3598, 37]`（ncnn 取到 `[3598,37]`）。
- `3598` = 4 个 stride 特征图 anchor 数：`52²(s8)+26²(s16)+13²(s32)+7²(s64)`。
- `37` = `[5 类概率(已 sigmoid)] + [32 框分布 DFL]`。
- **anchor 顺序**（必须一致）：level 序 `[8,16,32,64]`，每 level 行主序 `for y: for x`。
- **DFL decode**（每 anchor）：
  ```
  中心 cx=x*stride, cy=y*stride          # 无 +0.5 偏移
  对 4 条边 e∈{l,t,r,b}（各 8 bin，通道序 l,t,r,b）:
      dist_e = Σ_{k=0..7} softmax(bin)[k] * k * stride
  框(416空间) = [cx-l, cy-t, cx+r, cy+b]
  → 乘 (W/416, H/416) 回原图 → 裁剪 [0,W]×[0,H]
  score,label = max/argmax(前5通道)
  → conf 过滤 → 逐类 NMS(stable sort by score desc)
  ```

## 集成步骤
- **依赖**：ncnn 官方预编译库 <https://github.com/Tencent/ncnn/releases>（iOS: `*-ios`；Android: `*-android-vulkan`）。
- **iOS**：`feeder_detector.cpp/.h` 设为 Objective-C++（或 `.mm` 桥接）；Swift 侧封装暴露 `detect(UIImage)->[Detection]`；模型放 bundle。
- **Android**：`ncnn/` 源码放入工程，CMake 链 ncnn；写 JNI `detect(byte[] bgr,int w,int h)`；Kotlin 侧 Bitmap→BGR 字节。
- 参考官方 `demo_android_ncnn`（把其 nanodet.cpp 换成本 feeder_detector.cpp）。

## 不变量 / 禁忌（Agent 必读）
1. **不要重复归一化**（模型已焊 (x-mean)/std，mean=[103.53,116.28,123.675] BGR, std=[57.375,57.12,58.395]）。
2. **不要重复 sigmoid**（前 5 通道已是概率）。
3. **输入必须 BGR**（RGB 会导致颜色相关误检）。
4. **resize 是拉伸不是 letterbox**——回映射只需乘 W/416、H/416，无 padding 偏移。
5. **NMS 与画框在 app**，不在模型（ncnn 无法稳定内置 NMS）。
6. **fp16**：`use_fp16=true` 推荐（体积/内存半、近无损）。

## 验证状态（已通过）
- C++ ≡ Python 参考：多张密集多目标图 × 多阈值，检测集合 **100% 一致**（含空图边界）。
- decode 正确性：固定 test 上 pred-vs-GT IoU 满框 0.995 / 局部 0.951 / 中目标 0.89~0.92。
- 三格式对齐（9 张图、同 decode、都喂 0-255）：tflite-fp32 ≡ onnx（raw max|Δ|≈1e-5，检测框 100% 一致）；ncnn ≡ onnx/tflite（检测框一致，±1px 舍入）。

## 性能参考（供预期，非承诺）
- 固定 held-out test（泛化真值）：bird AP50 **85.0**、大目标 **90.1**；整体 AP50 0.655。
- 观鸟器工作场景集（部署同款场景，乐观）：bird AP50 **94.9**、大目标 98.5；全类 79~95。
- 空帧误报：conf 0.4 时约 **7.5%** 帧有误报 → 建议阈值≥0.4 + 时序滤波（连续 N 帧才告警）。
- 分辨率：416（命门优先）vs 320（省 ~1.7× 算力，bird 83.1，-1.9pt）。
- 类别注意：`cat` 训练含 camera-trap 山猫（非纯家猫）；`person` 数据稀缺，弱。
