# edge-cam-train

边侧 AI 相机训练→消融→量化→部署框架：在 **Allwinner V85x**（Vivante VIP NPU 0.5–1 TOPS, INT8-only）上做**动物粗检测 + 鸟类细分类**（分级、可选音频）。商用（海外）、PyTorch 系。

> **Status: W0 scaffold** — 目录骨架就绪，模型/部署叶子待实现（见 `docs/engineering.md` §7 里程碑）。

## 文档（先读这两份）
- **`docs/plan-v2.md`** — 总体方案（what/why）：级联架构、分级置信门控、模型/数据/评估、可选音频、附录 B 实验清单、附录 C 承重规格。
- **`docs/engineering.md`** — 工程实现（how）：PyTorch 分层选型、数据准备(§5.5)、仓库结构、W1 步骤、风险、ADR。

## 架构一句话
通用**粗检测器(NanoDet-Plus, 固定)** → 取 bird crop → **细分类器(timm EfficientNet-Lite0, 可 OTA)**；高置信用片上细标签，低置信→层级回退 + 云 API 锚点。上游只产 **FP32 ONNX** → 交 **ACUITY/pegasus 自做 INT8 PTQ → .nb → VIPLite/awnn**。

## 技术栈
- 检测 `train/detect`：NanoDet-Plus (Apache, fork 锁版)
- 分类 `train/classify`：timm + PyTorch Lightning + Hydra
- 数据 `data`：FiftyOne(拉 COCO/OIV7) + MegaDetector(出框) + CVAT/Label Studio + DVC
- 消融 `eval/ablation`：Hydra multirun + aim；门 `eval/gates`
- 部署 `deploy/packager/acuity_packager.py` + `edge/viplite_runner`（自研薄胶水）
- registry/OTA `registry` + `deploy/manifest_api`

## Quick start（W1 spike 先行）
```bash
# 1) 环境
uv sync            # 或 pip install -e .
# 2) 端到端打通 spike（最高优先）：一个检测器 → ONNX → pegasus PTQ → .nb → 上板跑通一帧
#    （摸清 ACUITY 算子兼容 + INT8 掉点；详见 docs/engineering.md §7）
```

## 许可与红线
全栈 Apache/MIT；**避 AGPL（Ultralytics）/ GPL（MMYOLO）/ CC-BY-NC（iNat、BirdNET 权重）**。数据只用 CC0/CC-BY + 自采，维护逐项署名清单。详见 `docs/engineering.md` §5、§8。
