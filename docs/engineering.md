# 鸟类细粒度识别 + 动物粗检测 · V85x 工程实现与框架选型

> 配套文档：本篇讲 **how（怎么把方案落成可训练/可对比/可部署的工程）**；总体方案（what/why）见《…— V85x 边侧推理方案》v2。
> 定调：**商用（海外）、工程最佳实践、成熟稳定可扩展**。主干生态 = **PyTorch 系**（决策见 §4）。

---

## 0. 结论先行

**新建一个干净的边侧 CV 仓库**，模型层**依赖成熟开源框架**（不 fork 改源码、用配置/扩展点在其上加东西），编排层**自建薄层或借成熟 OSS**（MLflow / Hydra / aim，采用成熟工程模式，**不依赖任何内部仓**），部署层**自写薄胶水**接 Allwinner ACUITY。
- 检测：**NanoDet-Plus**（Apache，anchor-free 无 Focus；fork 锁版）。
- 分类：**timm + PyTorch Lightning + Hydra**（成熟、config 驱动消融、可扩展）。
- 消融追踪：**Hydra multirun + aim**（自托管零成本）+ **DVC**（数据/产物版本）。
- 量化部署：上游只产 **FP32 ONNX**，交 **ACUITY/pegasus 自做 INT8 PTQ → .nb → VIPLite/awnn**。
- 编排层（registry/OTA/消融/gate）**用成熟模式自建薄层或基于 MLflow/Hydra/aim**（详见 §3）。
- 领域基座复用：**MegaDetector(MIT 变体)** 预标注、**Pytorch-Wildlife** 借级联范式、**SpeciesNet** 当 teacher、**BirdNET** 仅借范式（权重 NC 禁商用）。

> 一个绕不开的平台现实：**没有任何框架能一路覆盖到 Vivante NPU 部署端**（ACUITY/pegasus→.nb→VIPLite 是厂商私有）。所以模型/训练层可重度复用 OSS，但**量化落地那段一定是薄自研胶水**——这不是设计缺陷，是平台事实。

---

## 1. 工程目标与约束

| 项 | 要求 |
|---|---|
| 商用许可（海外发行） | 全栈避 AGPL / GPL / CC-BY-NC；优先 Apache/MIT/BSD；逐项目可追溯 |
| 复用姿势 | **依赖上游 + 用扩展点（config/callback/plugin）在其上加**；❌ 不 fork 改框架源码 |
| provenance | 新仓代码来源干净（自写 / 许可明确的 OSS）；不掺入无 LICENSE 的内部代码 |
| 可扩展 | config 驱动；消融矩阵可一键扩列；产物/数据可复现（DVC）；registry + OTA 可灰度 |
| 平台 | V85x（Vivante VIP NPU 0.5–1T，INT8-only，ACUITY/pegasus PTQ，Tina Linux） |
| 偏好 | 在现成模型上**微调**；保留有纪律的消融对比（说服 stakeholder） |

**「复用框架」的两种姿势（务必分清）**：
- ✅ **依赖上游 + 扩展点**：pip/submodule 引入，自己的代码作配置 + 适配器 + 回调"在其上"。可升级、可维护、scalable。
- ❌ **fork 进去改源码**：跟不上游、bug/安全补丁断供。商用别这么干（NanoDet 因半停滞需 fork 锁版属例外，且只锁不大改）。

---

## 2. 分层架构与选型（全 Apache/MIT，商用安全）

| 层 | 选型 | License | 用法 | 理由 |
|---|---|---|---|---|
| **检测训练** | **NanoDet-Plus** (RangiLyu) | Apache-2.0 | 直接用（fork 锁 commit） | anchor-free 极小、无 Focus、贴 0.5–1T；自带 `export_onnx`。风险：release 停在 2023-03，需 fork 锁版 |
| 检测（备选，更活跃） | **PicoDet**（PaddleDetection）/ **RTMDet-tiny**（MMDetection, Apache） | Apache-2.0 | 直接用 | PicoDet 维护活跃但**拉 Paddle 运行时**；RTMDet-tiny 是 PyTorch 框架式、但 mmcv 版本耦合需接受 |
| **分类训练** | **timm + PyTorch Lightning + Hydra** | 均 Apache-2.0 | 直接用 | timm 1.0.x（2026 活跃）含 EfficientNet-Lite0/MobileNetV3/RepVGG + 预训练；Hydra `multirun` 天然跑消融；有 `lightning-hydra-template` 现成脚手架 |
| **蒸馏（可选）** | 自写 KD/DKD/CWD loss | 自有 | 借算法 | loss 代码量小；**别**为它拉停摆且 pin torch 1.13 的 MMRazor |
| **音频（可选）** | timm 频谱图 CNN（EfficientNet-Lite0） | Apache-2.0 | 直接用 | 复用同一条分类训练/导出路（方案 §9） |
| **消融追踪** | **aim**（主）+ **DVC**（数据/产物） | 均 Apache-2.0 | 直接用 | aim 自托管零成本、为上万 run 设计；DVC 管代表性集/.nb 可复现。正式 Model Registry 再加 MLflow；W&B 体验最好但闭源付费 |
| **本地 INT8 掉点预估** | ONNXRuntime `quantize_static`(QDQ) | Apache/MIT | 直接用（仅做消融列） | 最贴近 ACUITY 将消费的 ONNX 形态，用于上游淘汰掉点严重配置。**产物不进部署** |
| **registry / OTA** | **MLflow Registry** 或自建薄层（git-yaml 索引 + channels + manifest API） | Apache / 自有 | 直接用 / 自建 | 模式成熟、格式无关：model card + sha256 + channel(candidate/stable) 承载 .nb 灰度 + cloud-fallback；单人项目可省 MLflow 只留 git-yaml |
| **量化+编译+部署** | ACUITY/pegasus PTQ→.nb→VIPLite/awnn | 厂商 | 厂商工具链 | **决定性事实：ACUITY 吃 float 模型自做 INT8 PTQ，不要喂预量化图** |
| **端侧 glue 蓝本** | `frigate_npu_vivante` 的 ONNX→ACUITY→NBG→VIPLite 链路 | MIT | 借范式 | 全链路 + ctypes 调 VIPLite 可借；但其芯片是 A733/VIP9000，**V85x 需换 Tina SDK 内配套版本并真机重做** |

**两条铁律：**
1. **上游只产 FP32 ONNX，绝不上游 INT8。** PaddleSlim/PT2E/ORT 的 INT8 产物对 Vivante 无用；ACUITY 用代表性集自做 PTQ。上游 INT8 仅作消融矩阵的"预期端侧精度"列。
2. **"框架能导 ONNX" ≠ "ONNX 能被 pegasus 干净解析"** 是两个独立验收项。检测后处理（NMS/grid/decode）最易出问题 → 优先导"裸 backbone+head、后处理留 CPU(A7)"。

---

## 3. 编排层（registry / OTA / 消融 / gate）：用成熟模式，自建薄层

编排层**不依赖任何内部仓**，采用业界成熟模式 + 成熟 OSS，自建一薄层即可：

- **消融对比**：Hydra `multirun`（config 驱动笛卡尔网格：`backbone × 输入分辨率 × 增强 × quant 档`）+ **aim**（追踪）+ **DVC**（数据/产物版本）。
- **registry + 灰度 + OTA**：**MLflow Model Registry**，或自建薄层 = `git-tracked models.yaml 索引 + sha256（大 .nb 流式分块）+ channel(candidate/stable) + FastAPI manifest API`。承载 .nb 灰度推送 + cloud-fallback；**gate_pass=True 才 promote**。OTA bundle 见方案附录 C.5（`net.nb + taxonomy_map(ver) + regional_mask(ver) + min_runtime_abi` 原子下发）。
- **质量门 (gate)**：per-key 阈值 + 全局 AND；INT8-only 下两档（`fp32 baseline` / `int8`），维度 = `mAP_drop / Top1_drop / NPU_latency_p95 / 模型体积`。
- **抽象**：用 Protocol 定义 `TrainerBackend / PackagerBackend / InferBackend`，工厂派发，便于加 `AcuityPackager`。

> **关于本地的 `pet-train-unified`**：它是一个 VLM scaffold（v0.1、叶子多为 stub、无 LICENSE）。**评估后决定：不 copy、不 fork、不依赖其代码——只借上面这些通用工程模式（思想）**。这样新仓 provenance 干净、不耦合无关 VLM 依赖，也避免与"全 Apache/MIT 商用安全"原则冲突。上面的 registry/gate/Protocol 都是通用模式，自建或基于 MLflow/Hydra 即可，无需引入任何内部代码。

---

## 4. 主干生态决策：PyTorch 系（针对本项目）

| 维度 | PyTorch 系（选） | Paddle 全家桶 |
|---|---|---|
| 领域基座复用 | **MegaDetector/Pytorch-Wildlife/SpeciesNet 全是 PyTorch** ✅ | 检测在 Paddle、其余仍 PyTorch → 双生态 ❌ |
| 细分类（本项目最难） | **timm 该领域最强、预训练最全** ✅ | PaddleClas 次之 |
| 音频模块 | timm 频谱图 CNN 顺 ✅ | 需另搭 |
| 商用 scale / 人才（海外） | **PyTorch 全球池最深** ✅ | 偏国内 |
| "单一成熟框架"体感 | 检测是 repo 非 framework（NanoDet 半停滞需 fork） | PaddleDetection 更"全家桶" |

**决策：PyTorch 系**——本项目要复用的领域资产 + 最强细分类库 + 音频都在 PyTorch，且做**海外**、PyTorch 人才池最深，净收益高于 Paddle 的"单框架"体感。
**翻盘条件**：若团队本身是 Paddle shop，则 PaddleDetection+Clas+Slim+2ONNX 单生态亦可（本文档 90% 结构可平移，仅替换 §2 检测/分类两行）。

---

## 5. 可复用/借鉴的领域项目

| 项目 | License | 用法 | 注意 |
|---|---|---|---|
| **timm** | Apache-2.0 | **直接用**（分类核心） | EfficientNet-Lite0 具体变体/ONNX 友好性以实测确认 |
| **NanoDet-Plus** | Apache-2.0 | **直接用**（检测，fork 锁版） | release 停 2023-03 |
| **PicoDet**（PaddleDet） | Apache-2.0 | 备选（拉 Paddle） | NPU 全量化变体 `picodet_s_416_coco_npu.yml`；其"NPU"是 Paddle Lite 通用 NPU，落 V85x 仍走 ONNX→ACUITY |
| **MegaDetector** | code MIT；**权重按变体分** | **直接用：预标注 bootstrap** | **用 `MDV6-mit-yolov9-c`(MIT) 或 `MDV6-apa-rtdetr`(Apache)；避开 yolov10 变体(AGPL)**；核 yolov9 推理依赖不踩 Ultralytics |
| **Pytorch-Wildlife** | MIT | 借级联范式（出框→裁剪→分类） | 上板模型不搬其内置 YOLO |
| **SpeciesNet** | Apache（权重 Kaggle 分发条款需核） | 借 taxonomy/blank+vehicle 过滤；可当**云端/蒸馏 teacher** | EfficientNetV2-M 太重不上板 |
| **BirdNET** | code MIT；**权重 CC-BY-NC-SA** | **仅借范式**（频谱图 CNN） | 权重及衍生**禁商用** |

---

## 5.5 数据准备（data prep pipeline）

> 数据**离线在 dev/GPU 机准备**（不在端侧）；全流程 **config 驱动 + DVC 版本化**，产物 = 标准 manifest（检测 COCO-json / 分类 csv+ImageFolder）+ crop 数据集 + 校准集。模块落在 `src/edge_cam/data/`。

**工具（全开源，商用安全）：**
- **FiftyOne**（Apache-2.0）= 数据准备主力：内置 zoo 一行拉 **COCO / Open Images V7**（**按类 + label 类型过滤**，可按 license 字段筛），并做可视化、去重、合并、导出 COCO/YOLO/ImageFolder。
- **CVAT / Label Studio**（自采数据标注）+ 双标抽检/仲裁（方案 C.9）。
- **MegaDetector（`MDV6-mit-yolov9-c`）** = 给分类/相机陷阱数据**自动出框** bootstrap。
- **DVC** 版本化 `raw → interim → processed` 三段 + 校准集 + .nb 产物，保证可复现。

**四条流水：**
1. **检测数据**：FiftyOne 拉 COCO 10 动物 + OIV7(`Squirrel` 等，**只留商用安全图 + 存逐图署名清单**)[+ LILA CC0/CC-BY 子集] → 统一大类**映射表** → OIV7 非穷尽标注做 ignore-region → 跨集去重、统一框格式 → 固定 seed **分层 split** → 导 COCO-json（NanoDet 吃）。bird 在检测层折叠为单类。
2. **分类数据**：开源 CC0/CC-BY 鸟种集 + 本地 BIRDS-525(补充) + 自采 → **taxonomy 归一**(eBird/Clements key) → 用检测器/MegaDetector 出框 → **统一 crop 规范**(方案 C.6: padding/min-size) → 类平衡 + 每类最小 N 门控 → split → ImageFolder/csv。
3. **校准集（PTQ 关键）**：从训练分布抽代表性图，**优先真机回采**，含**夜视/低照噪声/H.264 压缩**样本；**检测器与分类器各一份**、给昼/夜配比 → 产出 `calib/` + `dataset.txt`（pegasus 量化用，方案 C.7）。
4. **音频（可选）**：Xeno-canto API v3 按 `lic` **只拉 CC0/CC-BY、硬排 -nc/-nd** + 逐录音署名清单 → mel 频谱图 + SpecAugment/mixup/背景噪 → 接同一条分类流水。

> 自采数据的隐私（GDPR：人脸/车牌脱敏、采集告知、保留期）与标注 SOP 见方案附录 C.9。**真机回采是闭合 domain gap 的关键**，越早接入越好。

---

## 6. 推荐仓库目录结构

```
edge-cam-train/
├── LICENSE                          # Apache-2.0
├── pyproject.toml                   # 边侧依赖: pydantic/pyyaml/structlog/fastapi
│                                    #  + torch/timm/lightning/hydra-core/onnx/onnxruntime/onnxsim/aim/dvc
├── src/edge_cam/
│   ├── core/                        # 自建: config(Hydra)/logging/seed/paths/coords
│   ├── contracts/schemas/           # 自建 pydantic: dataset(manifest) / model_card / eval_report
│   │                                #   / channel(OTA 策略) / detection(打标契约, 11 类闭集校验)
│   ├── registry/                    # 自建薄层: store(git-yaml) + promotion(包络+gate→ModelCard→register/promote)
│   ├── deploy/
│   │   ├── manifest_api/            # 自建: FastAPI OTA routes (GET /v1/manifest/{platform}/{channel})
│   │   └── packager/
│   │       ├── base.py              # 自建: PackagerBackend Protocol
│   │       └── acuity_packager.py   # ★ subprocess 调 pegasus PTQ→.nb
│   ├── data/                        # ★ 数据准备 pipeline (见 §5.5, 离线 dev 机)
│   │   ├── ingest.py                # FiftyOne 拉 COCO/OIV7 (按类 + license 过滤)
│   │   ├── merge_map.py             # 跨集→统一大类映射 + 去重 + OIV7 ignore-region
│   │   ├── bootstrap_boxes.py       # MegaDetector 出框 (分类/相机陷阱数据)
│   │   ├── crop.py                  # 统一 crop 规范 (padding/min-size, 方案 C.6)
│   │   ├── taxonomy.py              # eBird/Clements 物种键归一
│   │   ├── split.py                 # 固定 seed 分层 split
│   │   └── calib.py                 # 代表性校准集 (夜视/噪声/压缩; 检测/分类各一份)
│   ├── train/
│   │   ├── detect/                  # ★ NanoDet-Plus fork 包一层 (导 FP32 ONNX)
│   │   └── classify/                # ★ timm + Lightning + Hydra
│   ├── eval/
│   │   ├── envelope.py / run_envelope.py  # 四级可行性包络 + Hydra 入口 (+可选 register)
│   │   ├── full_eval.py             # 完整评估编排 seam (量化+mask+包络; run_envelope/ablation 共用)
│   │   ├── metrics.py               # Top-1/5/per-class + topk_hits (训练侧共用)
│   │   ├── regional.py              # likely-species mask (taxon_key → logit 列)
│   │   ├── detect_metrics.py        # 检测 COCO mAP/bird-recall 结构化 + 汇入 detect 总表
│   │   ├── ablation/                # ★ Hydra multirun harness (grid + runner)
│   │   ├── gates/gate.py            # fp32+int8 两档, 4 维阈值门 (+from_yaml)
│   │   └── quant_estimate.py        # ★ ORT-QDQ 本地掉点预估 (消融列, 不进部署)
│   └── edge/
│       └── viplite_runner/          # ★ ctypes 调 VIPLite (借 frigate 蓝本; 输出 CHW reshape!)
├── scripts/                         # 离线工具: setup_nanodet.sh / build_ebird_mapping.py / build_region_list.py
├── configs/
│   ├── ablation/*.yaml              # Hydra multirun: backbone × 分辨率 × quant 档
│   ├── eval/gates/second_gate.yaml  # fp32 / int8 两档阈值
│   └── channels/*.yaml              # OTA candidates+status+cloud-fallback
├── data/                            # DVC 跟踪: 训练集 / 代表性校准集 / .nb 产物
└── docs/decisions/                  # ADR (见 §9)
```

> 标 ★ = 需新写（模型层委托 OSS、不自造）；其余为按成熟模式自建的薄编排层。

---

## 7. 落地步骤（W1 起，对齐方案 §10 / 附录 B）

- **W1 — 端到端打通 spike（最高优先 / 最大不确定性）**：NanoDet-Plus（或 RTMDet-tiny）→ `export_onnx` → onnxsim 静态 → pegasus PTQ → .nb → VIPLite/awnn **板上跑通一帧**。摸清 ACUITY 对 ONNX 算子的真实兼容清单 + INT8 掉点。借 frigate 蓝本立即落实**输出 buffer 按 CHW reshape**。验收：一帧端到端有合理输出。
- **W1–W2 — 编排骨架接线**：自建 core/config/logging + registry(git-yaml + channels) + manifest_api + contracts schema + Protocol 抽象，补 LICENSE。验收：`GET /v1/manifest/v85x/stable` 返回空 manifest、registry 能登记一个假 .nb。
- **W2–W3 — 训练侧立起**：timm+Lightning+Hydra 跑通分类 baseline；NanoDet-Plus 跑通检测 baseline；两侧均能导 FP32 ONNX。aim+DVC 接入。
- **W3–W4 — 消融 harness**：Hydra multirun + metrics 层（mAP/Top-1/per-class）+ ORT-QDQ 掉点列。跑首张消融矩阵（backbone × 分辨率 × {fp32 / 模拟INT8 / ACUITY实测INT8}）。产出 Markdown 表给 stakeholder。
- **W4–W5 — gate + publish 链路**：second_gate（fp32/int8 两档、4 维）；接 AcuityPackager；`gate_pass→promote→stable channel→OTA manifest` 全链路打通。
- **并行 — 数据准备（§5.5，离线 dev 机）**：FiftyOne 拉 COCO+OIV7（按类/license 过滤）→ 映射/去重/分层 split；MegaDetector 出框 bootstrap + 人工校正；构建检测/分类各自校准集；DVC 版本化。真机回采尽早接入。

---

## 8. 关键风险 / 坑

1. **跨设备 latency/精度回灌（最大工程盲点）**：板上 .nb 的真实 latency/精度回传到 EvalReport，需新增一层（串口/网络/手动录 jsonl），无现成支撑。**high**
2. **ONNX 算子兼容是 spike 必踩项**：head 后处理、hard_swish/SiLU、depthwise、上采样在 ACUITY/VIPLite 的 INT8 支持度未知；NMS/decode 大概率外置 CPU。**high**
3. **INT8-only**：frigate 蓝本"INT16 优于 uint8 保精度"在 V85x 可能不可用；敏感层无 INT16 退路则需换骨干或 fake-quant 微调。**medium**
4. **上游 INT8 ≠ 部署 INT8**：ORT/PT2E 掉点是方向性信号，真实数字必须来自 ACUITY verification。**high**
5. **frigate 蓝本不跨芯**：其 NBG 针对 A733/VIP9000，V85x 需用 Tina SDK 内配套 pegasus/viplite 真机重做，勿假设可复用。**high**
6. **许可暗雷（海外发行从严）**：① MegaDetector 避 yolov10(AGPL) + 核 yolov9 依赖；② BirdNET 权重 CC-BY-NC 禁商用；③ MMYOLO=GPL-3.0 整体排除；④ NanoDet fork 只锁版别大改源码；⑤ 不掺入无 LICENSE 的内部代码。**high**
7. **AcuityPackager 复杂度未知**：pegasus 有无 Python API、还是只能 subprocess 调 CLI → spike 时一并摸清。**medium**
8. **MLflow 对单人项目可能过重**：registry 只用 git-yaml + channels 那条腿即可，省掉 MLflow。**medium**

---

## 9. 决策记录（ADR 摘要）

| ADR | 决策 | 理由 |
|---|---|---|
| 新建边侧仓 | 与现有任何内部仓物理隔离、独立 | 范式/依赖/provenance 都不同 |
| 不复用内部 VLM scaffold 代码 | 评估过 `pet-train-unified`：**不 copy、不 fork、不依赖其代码，只借工程模式（思想）** | 它是 v0.1 stub + 无 LICENSE；保持新仓 provenance 干净、不与"商用安全"冲突 |
| PyTorch 系 | 主干 PyTorch，非 Paddle | 领域基座 + timm + 音频全在 PyTorch；做海外人才池深 |
| 上游只产 FP32 ONNX | INT8 交 ACUITY | Vivante 私有量化，预量化图无用 |
| 后处理留 CPU | NMS/decode 不进 NPU 图 | 量化后后处理偏差大 |
| 复用 = 依赖上游 | 不 fork 改框架源码（NanoDet 锁版除外） | 可升级 / 可维护 / scalable |

---

## 10. 待确认（实测 / stakeholder）
(a) 实验追踪是否强约束全开源（决定 aim vs W&B / 是否引 MLflow）；(b) V85x SDK 的 pegasus/viplite 具体版本与 INT8-only/INT16 约束（从 Tina SDK 内确认，公开文档无版本号）；(c) 检测器：NanoDet-Plus 原版 vs RTMDet-tiny/PicoDet；(d) 目标海外市场区域（决定 eBird 地域包覆盖与物种集）。
