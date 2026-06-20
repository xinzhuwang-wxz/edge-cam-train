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

## Quick start
```bash
# 1) 环境（项目专用 conda env，含 torch/timm/lightning/hydra/onnx 全栈）
conda env create -f environment.yml          # 建 edge-cam-train + 装 .[train,dev]
conda activate edge-cam-train

# 2) 数据准备（CPU，本地）：ImageFolder → 固定 seed 分层 split → manifest
python -m edge_cam.data.prep --config configs/data/birds525.yaml

# 3) 训练 smoke（CPU 本地验证框架；真训上 GPU）
python -m edge_cam.train.classify.train \
  data.manifest=data/processed/birds525/manifest.json \
  trainer.fast_dev_run=true trainer.accelerator=cpu model.pretrained=false \
  data.num_workers=0 data.input_size=64 hydra.job.chdir=False

# 4) GPU 真训（AutoDL）：去掉 smoke 覆盖即可
python -m edge_cam.train.classify.train data.manifest=... trainer.accelerator=gpu
```
- 测试：`pytest`（快，pre-commit 同款）；`pytest -m "slow or not slow"`（含 torch 端到端）。
- GPU（Linux+CUDA）：pip 默认装 CUDA torch；特定 CUDA 版本见 `environment.yml` 注释。
- 上板 spike（ACUITY/pegasus → .nb → VIPLite）见 `docs/engineering.md §7`，待有板子。

## 许可与红线
全栈 Apache/MIT；**避 AGPL（Ultralytics）/ GPL（MMYOLO）/ CC-BY-NC（iNat、BirdNET 权重）**。数据只用 CC0/CC-BY + 自采，维护逐项署名清单。详见 `docs/engineering.md` §5、§8。
