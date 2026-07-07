# ADR-0009：V861 部署路径按 AWNN 工具链重写（更正 ACUITY/.nb/A7 假设）

- 状态：Accepted（2026-07-07，实物 SDK 实测 + 用户对齐）
- 关联：更正 [[ADR-0008]]（芯片段「工具链不变」句）、`docs/classify/04-V861选型调研.md` §一；不动 [[ADR-0007]]（logits/后处理留 CPU）、CLAUDE.md §4 四条硬规则；落地见 `docs/detect/04-V861-AWNN部署转换.md`、`results/detect/round2/v861_awnn/`
- 性质：**首份基于板端 SDK 实物（`vs861/` AWNN 镜像+文档）验证的平台 ADR**——此前 plan/ADR-0008/classify-04 的部署段是**网络调研 + V853 类比**，本 ADR 用实测覆盖其中被证伪的部分。

## 背景

拿到 V861 的 AWNN SDK 实物（`vs861/AWNN-image.zip` = Docker 镜像 `awnn:1.0.2`；`vs861/NPU/*.pdf` = 工具链/Runtime/SDK/快速入门四份指南）。此前所有部署假设（plan-v2 §附录C、ADR-0008、classify-04）都建立在**没有板子型号**的调研上，`src/edge_cam/deploy/packager/acuity_packager.py`、`edge/viplite_runner/` 脚手架**从未接通**。本轮首次用实物验证，核出几处调研期事实错误。

## 决策：按 AWNN 实况重写部署路径，更正三处平台事实

| 维度 | 旧假设（被证伪）| **V861 实况（本轮实测）** |
|---|---|---|
| 上游→板端工具链 | ACUITY/pegasus PTQ | **AWNN 工具链** `awnntools {build,profile,simulate,encrypt,generate_config_file}`（Docker 镜像，内含 PPQ 量化器）|
| 板端模型格式 | `.nb`（VIPLite/awnn） | **`_ipu.param` + `_ipu.bin`**（ncnn 衍生容器），由 **AWNN Runtime** 加载 |
| 应用核（后处理落点）| ARM Cortex-**A7** | **玄铁 Xuantie RISC-V**（镜像交叉链 `Xuantie-900-gcc-linux-6.6.0-musl32-x86_64`）|
| 板端验证工具 | 无现成 | **`awnn_verify`**（Tina SDK `make menuconfig` 选 awnn_runtime+awnn_verify 编译）|

**不变量（仍成立，本轮实测印证）**：
1. 上游只产 **FP32 ONNX**，INT8 交 AWNN PTQ（AWNN=Vivante 私有量化的官方封装）——铁律不变。
2. **后处理留 CPU**（sigmoid/DFL/NMS）——ADR-0007 不动；只是「A7」措辞更正为「玄铁 RISC-V」。实测 profile 里 Reshape/Transpose/Permute 确实落 CPU、Conv 落 NPU。
3. **纯 CNN 安全**——实测 NanoDet-Plus-m 全 266 算子 NPU 化**零回退**（LeakyRelu/Concat/MaxPool/depthwise/channel-shuffle 全绿）；§4/§6 算子红线不变。

**部署路径重写为**（检测先做样板，分类/级联照套）：
```
FP32 ONNX ──awnntools build──▶ _ipu.param/.bin(INT8) ──▶ registry/ModelCard ──▶ 板端 awnn_verify/AWNN Runtime
```

## 证据（本轮实测，2026-07-07）

- arm64 Mac 上 `docker run --platform linux/amd64 awnn:1.0.2 awnntools -h` 完整运行（qemu 模拟）→ 离线全链本机可执行。
- logits ONNX（`main_416_fp32_logits.onnx`）`awnntools build` 成功：opset11 直吃、266 op 全量化零回退、产出 `_ipu.param`(63KB)+`_ipu.bin`(1.35MB)；profile 整网输出 **cos-sim 0.9949**（9 图冒烟校准）。

## 后果

- **正面**：部署路径首次接通真硅工具链；检测样板跑通后，分类/级联部署照 `_ipu` 模子套；INT8 权重 1.35MB 远在内存预算内。
- **工程债（转 issue，`/to-issue`）**：
  - `deploy/packager/acuity_packager.py` → 改写为 `awnn_packager`（subprocess 调 `awnntools build`，非 pegasus）。
  - `edge/viplite_runner/` → 对齐 AWNN Runtime API（`.nb`/VIPLite 假设作废）。
  - `registry/promotion` → ModelCard 记录 `_ipu.param/.bin` 产物 + AWNN profile 掉点列。
  - 分类段/级联段部署套用本样板。
- **需同步文档（先过本 ADR 再改）**：CLAUDE.md §1 承重规格（V85x/0.5T/ACUITY/.nb → V861/1T/AWNN/_ipu、A7→RISC-V）；ADR-0008 §背景「工具链不变」句；classify-04 §一工具链行。
- **仍是盲点**：真·上板（真 INT8 延迟/内存/fps/board-vs-sim 位匹配）需 V861 板子——交接包备好，做到「需要板前的最后一步」。
