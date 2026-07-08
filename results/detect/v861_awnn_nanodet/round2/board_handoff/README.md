# Feeder 检测器 · V861 板端交接包（AWNN / INT8）

喂食台粗检测（NanoDet-Plus-m 416，5 类）的 **Allwinner V861 NPU 部署包**。板端由 **AWNN Runtime** 加载 `_ipu.param/.bin`。
> 检测 5 类：`bird / squirrel / cat / person / other_animal`。命门 = bird。

---

## 这是哪个模型（provenance）

- **源** = `results/detect/round2/exports/main_416_fp32_logits.onnx`
  = round2 **main 模型**（NanoDet-Plus-m-416，1.0x，100% 数据 —— round2 定案部署点，与 `mobile_handoff` **同一个模型**）
  的 **FP32 logits 导出**（按 ADR-0007 剥 sigmoid、剥归一化）。
- **量化** = AWNN 工具链（`awnntools build`），**VS861 策略**（见下），产出板端 INT8 `_ipu`。
- 全链实操与决策：`docs/detect/04-V861-AWNN部署转换.md`；工具链更正见 ADR-0009。

## VS861 量化策略（为什么这么量化 = 决定板上精度）

1. **混合精度——检测头 `gfl_cls` 卷积保 fp32**。AWNN 默认全 INT8+percentile 会**裁掉 cls 头稀有高峰值** → 丢强检出（实测清晰松鼠 0.91→0.33 漏检）。头保 fp32 修复 → **框召回 4/4 / IoU 0.945**，`_ipu` 仍 **1.3MB**。合 ADR-0007「保护头」。
2. **`use_npu_preprocess`**：归一化（BGR mean=[103.53,116.28,123.675] / norm=1/std）**折进 NPU** → 板端**直接喂 0-255 BGR**，不用自己做 mean/std。
3. 骨干 `symmetric_i8` / `per-channel` / `percentile`；266 算子零回退。

## I/O 契约

- **输入**：**0-255 BGR、HWC**、resize **416×416（直接拉伸）**。⚠ BGR 不是 RGB；⚠ 不用自己归一化（已焊进 NPU）。blob 名 `data`。
- **输出**：blob `output`，**logits** `[1, 3598, 37]`（**sigmoid 前**）。
- **CPU 后处理**（玄铁 RISC-V，不在 NPU 图）：`sigmoid(前5类) + DFL(reg_max=7) + 逐类 NMS`。
  - 参考实现 `src/edge_cam/cascade/adapters.py:decode_nanodet(out, orig_wh, num_classes=5, strides=(8,16,32,64), reg_max=7, conf_thr=0.4, nms_iou=0.5)`。
  - `3598` = 52²(s8)+26²(s16)+13²(s32)+7²(s64)；`37` = 5类 + 4×8 DFL。

## 文件

```
board_handoff/
├── model/nanodet_feeder5_v861_416_ipu.param / .bin   板端 INT8 模型（AWNN Runtime 加载, 1.3MB）
├── config.txt                                        awnn_verify 验证配置
├── labels.txt                                        5 类名（行号=id）
└── ref/  demo_bird.jpg + _input_int8.bin + _output_fp32.bin   板端对拍参考(input + 仿真参考输出)
```

## 板端用法（awnn_verify）

1. **编译工具**：V861 Tina SDK `make menuconfig` → Vision → 选 `awnn_runtime` + `awnn_verify`，重编生成 `awnn_verify`。
2. **推包上板**：整个 `board_handoff/` 推到板上。
3. **跑**：`./awnn_verify config.txt` → 看 `test success ^_^`（板端输出 ≡ 仿真参考）+ NPU 层耗时 / 带宽 / 内存统计 / 单次延迟。
4. 集成推理：AWNN Runtime API 加载 `_ipu`，喂 0-255 BGR → 取 `output` logits → CPU 跑 `decode_nanodet`。

## 验证状态（诚实）

- **离线已验证**（无板，AWNN simulate vs onnxruntime FP32，同预处理）：清晰检出图 **框召回 4/4 / 精度 4/4 / IoU 0.945 / 零误报**；命门 bird IoU 0.95-0.99。整网 logit cos-sim 0.994。
- **待板端**（本包目标）：真 INT8 延迟 / fps / 实测内存 / board-vs-sim 位匹配 —— 需 V861 板子跑 `awnn_verify`。**这是唯一未闭合的一步**（项目已知盲点）。
- 边界：很小/很远目标（观鸟器工作点外）本就弱；`person`/`cat` 数据弱。
