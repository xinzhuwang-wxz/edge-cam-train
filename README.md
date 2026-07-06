# edge-cam-train

> 边侧 AI 相机（智能观鸟器）鸟类识别的 **数据 → 训练 → 评测/消融 → 量化 → 部署** 全链路框架。

## 场景

**智能观鸟器 / 喂鸟器（bird-feeder camera）的端侧鸟类识别。**

- **目标硬件**：Allwinner **V861**（VeriSilicon Vivante NPU，**1 TOPS INT8-only**，128MB 内存，太阳能/低功耗）。
- **差异化**：市面观鸟器（Bird Buddy / Birdfy 等）清一色"端侧只拍照上传、识别放云端"；本项目把**细分类真正做在端侧芯片上**——离线可用、隐私（图不离场）、低延迟、不依赖订阅。
- **两段级联**：通用**粗检测器（NanoDet-Plus，固定）** → 取 bird crop → **细分类器（EfficientNet-Lite 级，可 OTA 换/扩）**。
  - 高置信 → 出片上**种级**标签；
  - 低置信 / 不在地域清单 / 稀有种 → **层级回退（属 / 科 / bird）+ 地域·月份先验重排 + 云 API 锚点**。
- **平台现实**：上游只产 **FP32 ONNX**；INT8 量化交板端 **ACUITY/pegasus PTQ → .nb → VIPLite**（薄自研胶水）。检测后处理（NMS/decode）留 A7 CPU，不进 NPU 图。

## 训练评测框架

检测、分类两段共用一条链路，每段可独立跑：

```
数据准备 → 训练 → 评测 / 消融 → gate 门控 → registry 注册 / OTA → 量化 / 部署
```

### 检测段（`train/detect` + `data/adapters/detect` + `eval/detect_metrics`）
- **数据**：5 类（bird / squirrel / cat / person / other_animal），`DatasetAdapter` 统一多源 + MegaDetector 出框伪标 + 逐图 provenance。
- **训练**：NanoDet-Plus（Apache，fork 锁版），导 FP32 ONNX（剥 sigmoid、后处理留 CPU）。
- **评测**：`class_precision/recall` + 零样本同尺（`zeroshot_eval`）。
- **产物**：移动端 **NCNN 交接包**（`results/detect/round2/mobile_handoff/`，C++≡Python 对齐验证）。
- **现状**：round1/round2 收官，固定 held-out test **bird AP50 85.0**（工作场景 94.9）。

### 分类段（`train/classify` + `data/adapters/classify` + `eval/`）
- **数据**：多源逐图 CC0/CC-BY 过滤 → 检测器 **crop** 裁框 → **taxonomy**（eBird 规范键，接姊妹 registry）→ 防泄漏 split（按 observer 分组）→ 校准集。
- **训练**：timm + PyTorch Lightning + Hydra；backbone 消融（EfficientNet-Lite / RepViT / MobileNetV4，纯 CNN、INT8 友好）。
- **评测**：四级包络（**fp32 / int8_sim / field / regional**）+ **命门 = 层级可用率**（`eval/hierarchical`，种→属→科→bird 上滚+置信门，自信错种重罚）+ 校准。
- **现状**：v1/v2 crop 消融（Lite0 val top-1 0.748）；选型 + 螺旋 harness roadmap 见 `docs/classify/04–07`。

### 贯穿层
- `eval/gates`：fp32 + int8 多维阈值门（promote 前把关）。
- `eval/ablation`：Hydra multirun 消融 harness。
- `registry`：ModelCard + git-yaml store + promotion（gate 通过才 register/promote），配 `deploy/manifest_api` 出 OTA。
- `deploy/packager`（ACUITY 量化胶水）、`edge/viplite_runner`（端侧推理）：接口就绪，**待 V861 板子填充**（上板段是最大工程盲点）。

## 快速上手

```bash
# 环境（项目专用 conda env，含 torch/timm/lightning/hydra/onnx 全栈）
conda env create -f environment.yml && conda activate edge-cam-train

# 测试：pytest 快（pre-commit 同款，跳 slow）；全量含 torch 端到端：
pytest -m "slow or not slow"

# 分类训练 smoke（CPU 验证框架；真训改 accelerator=gpu + pretrained=true）
python -m edge_cam.train.classify.train \
  data.manifest=data/processed/<dataset>/manifest.json \
  trainer.fast_dev_run=true trainer.accelerator=cpu model.pretrained=false hydra.job.chdir=False
```

## 文档地图

| 文档 | 讲什么 |
|---|---|
| `CLAUDE.md` | 协作指引 + 文档地图（接手先读）|
| `docs/plan-v2.md` · `docs/engineering.md` | 总体方案（what/why）· 工程实现（how）|
| `docs/detect/` · `docs/classify/` | 检测 / 分类全过程（数据·训练·评测）|
| `docs/decisions/` | ADR（不可逆决策，如 ADR-0008 分类模型架构选型）|
| `results/{detect,classify}/` | 实验报告 + 产物 |
