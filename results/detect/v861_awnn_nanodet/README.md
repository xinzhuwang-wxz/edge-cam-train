# NanoDet · V861 板端交接包（AWNN / INT8）· 按轮组织

NanoDet-Plus-m（ShuffleNetV2 1.0x, 416, 5 类）的 **Allwinner V861 NPU 部署包**。板端由 **AWNN Runtime** 加载 `_ipu.param/.bin`。**同类型 = 同接口**：I/O 契约与 `awnn_verify` 板端用法对所有轮通用；**按 round 分层**，每个 `roundX/` 是可直接推板的**自包含交付物**。
> 检测 5 类：`bird / squirrel / cat / person / other_animal`。命门 = bird。
> **自包含**：本包不依赖仓库其它文件——`git pull` 后把整个 `v861_awnn_nanodet/` 拷走即可用（decode 参考 `decode_ref.py` 只需 numpy，不 import `edge_cam`）。与移动端 `mobile_handoff_*` 一致。`_build/` 里的转换脚本 import `edge_cam`，但那是我们重转模型时用、**不随交接**。

```
v861_awnn_nanodet/
├── README.md          ← 本文件（通用：I/O 契约 + awnn_verify 用法 + 目录规范）
├── decode_ref.py      ← ★CPU 后处理参考 sigmoid+DFL+NMS（自包含，只依赖 numpy，跨轮通用）
├── round2/            ← ★可上板交付物（自包含，整个目录推到板上）
│   ├── model/nanodet_feeder5_v861_416_ipu.param / .bin   板端 INT8 模型（1.3MB）
│   ├── config.txt     awnn_verify 配置（路径相对本目录，铺平后无需改）
│   ├── labels.txt     5 类名（行号=id）
│   ├── ref/           对拍参考（int8 输入 + fp32 仿真参考输出 + 原图）
│   └── README.md      本轮 provenance / 量化策略 / 验证状态
└── _build/round2/     ← 转换工程（板端同事不用看）：awnn.sh · calib/ · configs/ · decode_repro.py · test_board_handoff.py · reports/
```

> **以后转新轮就照 round2 抄**：`mkdir roundN/` → 丢 `model/` + `config.txt` + `labels.txt` + `ref/` + `README.md`；转换脚本/校准图/报告进 `_build/roundN/`。**结构与移动端 `mobile_handoff_{nanodet,ppyoloe}` 同构**（顶层放通用说明，`roundX/` 只放那一轮的产物）。

---

## I/O 契约（所有轮通用 · NanoDet 架构级）

- **输入**：**0-255 BGR、HWC**、resize **416×416（直接拉伸）**。⚠ BGR 不是 RGB；⚠ **不用自己归一化**（mean/std 已焊进 NPU，`use_npu_preprocess`）。blob 名 `data`。
- **输出**：blob `output`，**logits** `[1, 3598, 37]`（**sigmoid 前**）。
- **CPU 后处理**（玄铁 RISC-V，不在 NPU 图）：`sigmoid(前5类) + DFL(reg_max=7) + 逐类 NMS`。
  - **权威参考实现（自包含）**：本包 `decode_ref.py`（只依赖 numpy，不 import `edge_cam`）——`python decode_ref.py round2/ref/demo_bird_output_fp32.bin round2/ref/demo_bird.jpg` 复现 bird 0.850。移植 C/RISC-V 照它 ~40 行翻译。
  - 仓库内同款（全链复用，非本包必需）：`src/edge_cam/cascade/adapters.py:decode_nanodet`。
  - `3598` = 52²(s8)+26²(s16)+13²(s32)+7²(s64)；`37` = 5类 + 4×8 DFL。

## 板端用法（awnn_verify · 通用）

1. **编译工具**：V861 Tina SDK `make menuconfig` → Vision → 选 `awnn_runtime` + `awnn_verify`，重编。
2. **推包上板**：把某轮的 `roundX/`（整个目录）推到板上。
3. **跑**：`./awnn_verify config.txt` → 看 `test success ^_^`（板端输出 ≡ 仿真参考）+ NPU 层耗时 / 带宽 / 内存 / 单次延迟。
4. **集成推理**：AWNN Runtime API 加载 `_ipu`，喂 0-255 BGR → 取 `output` logits → CPU 跑 `decode_nanodet`。

## 各轮索引

| 轮 | 模型 | 状态 |
|---|---|---|
| **round2** | NanoDet-Plus-m 416（round2 main，与 `mobile_handoff` 同一模型）| 离线闭环（框召回 4/4、IoU 0.945、整网 logit cos 0.994）；**真·上板待 V861 板子** |

> 板端与移动端的区别：本包走 **AWNN `_ipu` + 板上 AWNN Runtime**（不是 onnx/tflite→App）。移动端包见 `../mobile_handoff_nanodet/`。
