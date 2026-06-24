# 鸟类细粒度识别 + 动物粗检测 · V85x 边侧推理方案

> 自包含设计文档。目标：在 Allwinner V85x 边侧设备上实现「动物粗检测 + 鸟类细分类（分级、可云端兜底）」，可选叠加鸟鸣识别。
> 范围：聚焦**训练/微调 + 端侧部署**；云端细分类 API **仅留接口锚点、不在此实现**；音频为**可选模块**（§9）。
> 定位：这是产品的**一个 feature**，资源占用须克制；芯片档位**尚未定死**，方案按硬件**区间**设计。

---

## 摘要：设计要点速览（TL;DR）

- **两段式级联**：通用**粗检测器（固定）** → 取 bird crop → **细分类器（可 OTA 换/扩）**；非鸟直接出大类。这是 MegaDetector+SpeciesNet / Merlin 的成熟分工。
- **分级 + 置信门控**：bird 命中后，片上细分类给「种 + 置信度 + top-5」；**高置信用片上细标签**，**低置信 / 不在地域清单 / 稀有种 → 层级回退（属/科/bird）+ 留云 API 锚点**（本方案不实现云端）。
- **硬件按区间设计**：NPU 0.5–1 TOPS、DDR ≤256MB；对齐**下限 0.5T / 128MB** 保证全系可跑。真正瓶颈是**内存带宽 + 算子覆盖**，不是 TOPS。
- **商用优先选型**：检测器 **NanoDet-Plus / PicoDet（Apache）**；分类器 **EfficientNet-Lite0 / MobileNetV3-Large（timm, Apache）**；避开 AGPL（Ultralytics YOLO）与 CC-BY-NC（iNaturalist）。
- **量化走 PTQ**（`pegasus … asymmetric_affine uint8` + 校准集）；**检测后处理（解码/NMS）放 CPU**；避开 Focus/slice 与 transformer 等量化坑。
- **数据**：开源 + 自采为主（Open Images V7 补松鼠 / 可选 LILA 的 CC0-CC-BY 子集 / 自采；**COCO 图片非商用安全，仅借类目与作对照**）；**自训分类器**；**最终建自有真机数据集**闭合 domain gap。类别**数据驱动 + 层级头 + 地域过滤**，不预设固定大类数。
- **精度三层口径**（验证集 / 量化后 / **现场**）；**地域过滤是最大精度杠杆**。
- **音频（可选）**：复用同一条 NPU 图像分类通路（频谱图当图像）；**独立并行 + 晚融合**；底座**首选 Perch v2（Apache 可商用）embedding + 自训地区线性头**，避开 BirdNET 非商用许可。

---

## 1. 目标与定位

| 功能 | 说明 | 粒度 |
|---|---|---|
| **动物粗检测** | 定位画面中的动物 + 输出大类 | bird / dog / cat / horse / cow / squirrel …（按项目数据集定） |
| **鸟类细分类** | 检到 bird 后识别具体鸟种 | 数百级起步，随数据扩展；**地域过滤 + 层级回退** |
| **鸟叫识别（可选）** | 识别鸟鸣 → 物种 / 在场佐证 | 探索模块，见 §9 |

**分级输出 + 置信度门控（核心机制）**：
- 粗检测给大类；若为 bird → 片上细分类器给「种 + 置信度 + top-5」。
- **置信度高**：直接采用片上细标签。
- **置信度低 / 不在地域清单 / 稀有种**：输出层级回退标签（属/科/「bird」），并**留锚点**上交云 API（契约见 §3.3，本方案不实现云端）。

---

## 2. 硬件目标与硬约束

**目标区间**：Allwinner V85x 家族，**NPU 0.5–1 TOPS**，**DDR 封装内/外挂 ≤256MB（仅少量给本 feature）**。
**设计基线**：按**下限 0.5 TOPS / ~128MB 设计 → 全系可跑**；1 TOPS / 更大 RAM 作余量。

| 维度 | 事实 | 设计含义 |
|---|---|---|
| NPU | VeriSilicon **Vivante VIP**（Pico 小核），**0.5–1 TOPS**；仅 **UINT8/INT8/INT16，无 FP**；128KB 内部缓存 | 必须 INT8；权重从 DDR 流式读 |
| 瓶颈 | **内存带宽 + 算子覆盖**，不是 TOPS | 模型越小越好；避坑算子 |
| CPU | 单核 Cortex-A7 @900MHz（板级可达 1.2GHz）+ RISC-V E907@600MHz | 前后处理压预算；E907 可分担实时/轻任务 |
| RAM | 0.5T 档封装内 64MB(DDR2) 或 128MB(DDR3L)；1T 档可外挂 | **设计下限锚 ≥128MB；出货机型须保证 ≥128MB（64MB 款不在保证范围）**；串行加载、buffer 复用 |
| ISP/编码 | 单 ISP 最大 V851S ~4MP(2560×1440) / V853 ~5MP；H.264/H.265 编码 | 检测可吃较高分辨率帧；编码占内存要算 |
| 工具链 | ACUITY `pegasus`（PTQ）→ `.nb` → VIPLite/**awnn** @ Tina Linux | 见 §7 |
| OS | A7 跑 Tina Linux（推理在此）；E907 跑 Melis RTOS | NPU 访问在 A7 侧 |

> **性能锚点**（aw-ol 官方 1T 档 bench）：`mobilenet_v2 ≈ 133–145 FPS`、`yolov5@416 = 12.35 FPS`、`yolov5s@640 = 6.34 FPS`、`resnet18 = 96.94 FPS`。**折到 0.5T 约减半**。社区实测 1T 档：MobileNetV2 6.4ms / YOLOv5s 62.3ms。**V851S(0.5T) 上 YOLO/NanoDet 的一手 FPS 暂无公开基准 → 须实机实测（§B.4）。**

---

## 3. 总体架构

### 3.1 推理流水线（级联 + 门控）

```
摄像头帧                      [可选] 麦克风音频(§9)
   │                                  │
   ▼                                  ▼
运动门控 (ISP 帧差 / VPU 运动矢量)   声学门控/分类(可选)
   │  仅事件触发 NPU                   │
   ▼                                  │
粗检测器 (NPU, INT8, 320–416)          │
   │  bbox + 大类 (bird/dog/cat/...)   │
   ├── 非 bird ─────────────► 直接输出大类
   │                                  │
   ▼ bird crop (外扩 padding, 192–224)│
细分类器 (NPU, INT8)                   │
   │  种 + 置信度 + top-5             │
   ▼                                  ▼
跨帧 tracking / 时序多数投票 / 去重 ◄── (可选: 音视频晚融合佐证)
   │
   ▼ 置信度门控 + 地域清单过滤 + 层级回退
〔高置信〕片上细标签   〔低置信/不在清单/稀有〕→ 锚点: 上交云 API (不实现)
   │
   ▼
OSD / RTSP / 事件上报
```

要点：
- **串行加载**：检测完释放、再加载分类器（级联本就串行）→ 省内存。
- **检测器保持「通用粗类」且固定**；**分类器可 OTA 换/扩**（地域包）。
- **运动门控在 NN 之前**：常开→事件触发可省 ~90% 算力/功耗。

### 3.2 多目标与时序
- 一帧多只鸟 → 各自 crop、各自分类、各自 track。
- **clip 级标签 = 阈值以上逐帧预测的时序多数投票**（压制单帧错误、对同一个体去重）。

### 3.3 云 API 锚点（仅契约，不实现）
```
needCloud = (det.cls=='bird') && (cls.top1_conf < TH_species
              || species ∉ regional_checklist
              || species ∈ rare_set)
→ enqueue { crop_jpeg, top5_local, geo, ts } → CloudSpeciesAPI()   # 本方案不展开
```

---

## 4. 模型选型

> 选型三准则：**商用许可干净 + 有预训练可微调 + 量化/NPU 友好**。候选**精选少量**做受控对比（§4.3、附录 B），而非发散并行一堆。

### 4.1 检测器（粗动物）

| 模型 | 参数 | FLOPs@320/416 | COCO mAP | 许可 | 量化友好 | 评价 |
|---|---|---|---|---|---|---|
| **NanoDet-Plus-m** ⭐ | 1.17M | 0.9G / 1.52G | 27.0 / 30.4 | **Apache-2.0** | ✅ anchor-free 无 Focus | **首选** |
| **PicoDet-S** ⭐ | 1.18M | 0.97G / 1.65G | 29.1 / 32.5 | **Apache-2.0** | ⚠️ 含 INT8 变体，但 ESNet 含 SE/h-swish 需逐算子验回退(§7) | **首选** |
| YOLOX-Nano | 0.91M | 1.08G@416 | 25.8 | **Apache-2.0** | ⚠️ **stem 含 Focus（同 YOLOv5 INT8 坑）** | 独立 repo(Megvii)；除非换 conv stem 否则不建议上 NPU |
| RTMDet-tiny ⭐备选 | ~4.8M | — | 41(640) | **Apache-2.0** | ✅ anchor-free 无 Focus | **框架支持的第二选**（MMDetection, 活跃维护；低分辨率用） |
| YOLO-FastestV2 | 0.25M | 0.21G | 24.1(mAP@0.5) | ⚠️ **无明确 LICENSE** | ✅ ShuffleNetV2+解耦头, Focus-free | 极轻但**商用前须确认/授权**，否则仅研究用 |
| YOLOv8n/11n | 2.6–3.2M | 6.5–8.7G@640 | 37–39.5 | **AGPL-3.0 ⚠️** | 无 Focus | **仅研究对照**；商用需付费授权 |

- **推荐**：`NanoDet-Plus-m` 或 `PicoDet-S`（anchor-free、无 Focus、Apache、有预训练可微调）。
- **避坑**：**不用带 Focus/slice 的检测器**（YOLOv5、**YOLOX-Nano 的 stem 也是 Focus**）→ INT8 不友好；**不用 Ultralytics 系做商用出货**（AGPL）。
- **框架现实**：上述检测器**分属不同 repo**（NanoDet repo 只训 NanoDet）。无单一 Apache 框架同时覆盖；要"框架式多检测器 + Apache"用 **MMDetection（含 RTMDet/YOLOX）**，但 **MMYOLO 是 GPL 不能用**。跨 repo 对比靠统一"导 ONNX + eval 指标"层完成。
- **检测分辨率**：小/远鸟**优先抬分辨率**（分辨率比模型大小更影响小目标）→ 320 起，延迟允许上 384/416，letterbox，保留 stride-8/P2 高分辨率特征。

### 4.2 鸟类细分类器

| 模型 | 参数 | ImageNet top-1 | 许可 | NPU 友好 | 评价 |
|---|---|---|---|---|---|
| **EfficientNet-Lite0** ⭐ | 4.7M | 75.1% | Apache(timm) | ✅ 去 SE/swish，专为 INT8 | **首选** |
| **MobileNetV3-Large** ⭐ | 5.4M | 75.2% | Apache(timm) | ✅（h-swish/SE 实测） | **首选** |
| RepVGG-A0 | 8.3M | 72.4% | **MIT** | ✅✅ 纯 3×3 conv | **最 NPU 安全**对照 |
| ShuffleNetV2 1.0× | 2.3M | 69.4% | 宽松 | ✅ | 轻量对照 |

- **推荐主选**：`EfficientNet-Lite0` / `MobileNetV3-Large`；`RepVGG-A0` 作「最稳」对照。
- **不选 transformer 类（如 MobileViT）**：attention/LayerNorm 在 VIP-Pico 上多落慢速软算子、中间张量大吃 DDR 带宽。
- **crop 输入 192–224**：细粒度需要足够细节（喙形/眼纹/羽色）。
- **细分类头**：用**层级头（科/属 → 种）**，输出 **top-5 + 置信度**；低置信回退到属/科。

### 4.3 实验对比矩阵（给 stakeholder 的「证据」）
统一数据/口径下的受控网格（详见**附录 B 可执行清单**）：

```
检测器 ∈ {NanoDet-Plus-m, PicoDet-S}   (+ YOLOv8n 仅研究对照)
分类器 ∈ {EfficientNet-Lite0, MobileNetV3-L, RepVGG-A0}
档位   ∈ {0.5T/128MB, 1T/≤256MB}
量化   ∈ {FP32, INT8-PTQ}
记录: mAP/recall · top-1/top-5(地域过滤前后) · 延迟 · 内存峰值 · 量化掉点 · 算子是否回退CPU · 许可
```

---

## 5. 数据方案

> 原则：**开源商用安全数据为主；本地补充集仅作补充；最终建项目自有数据集（贴近真机域）**。

### 5.1 检测数据（粗动物，需 bbox）
> ⚠️ **商用许可**（详见附录 C.1 许可传染表）：**COCO 标注 = CC BY 4.0，但图片受 Flickr ToU、许可不统一、含 NC/ND/全保留图 → 不可视为商用安全**；**Open Images V7 图片为逐图 CC-BY（强制署名 + 链接义务，且 Google 不担保 license 正确性，须抽样/逐图复核）**；**LILA 各子集许可不一（含 NC/研究限定），须逐子集核实，仅纳入确认 CC0/CC-BY 的子集（如 NACTI）**。

- **COCO**（10 动物：bird/cat/dog/horse/sheep/cow/elephant/bear/zebra/giraffe）—— 主要借其**类目/schema 与作研究对照**；商用出货权重慎用其图片。
- **Open Images V7**（含 `Squirrel` 及更多野生动物框）—— **补齐松鼠等 COCO 缺类**；逐图 CC-BY，须维护**署名清册**。
- **可选 LILA BC 相机陷阱集**（仅取确认 CC0/CC-BY 子集如 NACTI）—— 加真实摄像头视角；分类标签用 **MegaDetector 自动出框**，**重定位为 crop/召回来源而非大类监督**（其 animal 单类不供 bird/dog/cat 标签）。
- **跨集合并须有显式映射表**（COCO/OIV7 → 统一大类）+ 处理 OIV7 **非穷尽标注**（漏标当背景会污染负样本）；规范见附录 C.9。
- **粗类优先级**（随部署场景可裁剪）：核心 = **bird / squirrel / cat / dog**；可选/场景相关 = horse / cow / sheep 等。
- **bird 在检测层统一为单一 `bird` 类**（细分交 §5.2）。
- **自有数据（关键）**：部署相机回采帧 → 标注（标注 / 隐私 SOP 见附录 C.9），逐步替换「网图域」。

### 5.2 鸟类细分类数据（图像级，可无 bbox）
- **开源商用安全为主**：优先 **CC0 / CC-BY** 鸟种图像集；本地补充集（如已有的 BIRDS-525 类集，注：已裁剪 ~224²、每类样本偏少、全球种、**商用前须核实图片版权**）。
- **谨慎使用非商用集**：iNaturalist（CC BY-NC）、NABirds（研究授权）——**部署权重不得含其数据**；至多用于**内部蒸馏 teacher / 预训练探索**（不进商用产物）。
- **crop 域统一**：用检测器/MegaDetector 自动裁剪到 crop 域训练 → **缩小 domain gap**。
- **taxonomy 统一**：以 **学名 + eBird/Clements**（带版本）为规范键 → 跨集可合并、支持地域过滤与层级回退。

### 5.3 类别规模策略 —— 数据驱动 + 分级
1. 片上分类器训练成「**当前干净数据能覆盖的最大类集**」，**类数由数据驱动、不预设固定大数**（从数百级起步，随自有数据增长再扩）。
2. 用 **层级头 + top-5 + 地域清单过滤** 让「类多 ≠ 现场差」。
3. 全局长尾 / 稀有种 → 锚点交云（不实现）。
> 不采用「单一巨型扁平分类头硬扛上千种 + 多套数据集手工 taxonomy 对齐」的路线：那既增加训练复杂度，又因长尾噪声拖垮现场精度。数据驱动 + 层级/地域过滤是 Merlin / Bird Buddy 的工业做法。

### 5.4 地域清单（推理期过滤，低成本大收益）
**不改训练、不绑死区域**：训练全局头；推理时按设备地区加载一张 **「likely species」mask（eBird 频率）**，把候选从全局缩到几十~几百。可选、可 OTA。这是把「难的全局分类」变「易的地域分类」的**最大杠杆**。

---

## 6. 训练 / 微调策略（fine-tune 优先）

- **检测器**：加载预训练 → 在合并动物集（COCO+OIV7[+LILA]）上**微调**；bird 统一为单类；后处理（解码+NMS）切出图放 CPU。
- **分类器**：加载 `timm` ImageNet 预训练 → **微调**（优先在现成模型上微调，控成本）。
  - 小样本/长尾：强增强（**模拟夜视灰度 / 低照噪声 / H.264 压缩伪影 / 运动模糊 / 远距离降采样**）+ label smoothing + 类平衡采样。
  - 可选**知识蒸馏**：大 teacher（EfficientNet-B 系 / ViT）→ 移动 student。
- **域适配**：尽早混入真实摄像头帧；**最小 crop 尺寸门控**（太小太远 → 宁可只报「bird」不强报种）。
- **量化**：**PTQ**（`pegasus asymmetric_affine uint8` + 代表性校准集，含夜视/噪声/压缩样本）。无 QAT 官方路径；若 PTQ 掉点过大，再在 PyTorch 端做 fake-quant 微调后导 ONNX 作补救。

---

## 7. 量化与端侧部署

**链路**：`PyTorch → ONNX → onnxsim(静态 shape) → pegasus import → pegasus quantize(PTQ) → .nb → VIPLite/awnn @ Tina Linux`

**必做 / 避坑**：
- 检测**后处理（sigmoid/anchor decode/NMS）放 A7 CPU**（OpenCV）；`pegasus --outputs` 切到卷积输出节点（量化后输出经后处理偏差大，必须切出）。
- **避开 Focus/slice**；SE / h-swish 实测是否回退软算子；**不用 transformer**；**动态 shape 先消除**。
- **内存**：串行加载、frame buffer 复用、单模型单位数 MB；扣系统/ISP/编码后**实测可用内存**再定。
- **工程坑**：Tina-SDK 须 Ubuntu 编译；**VIPLite `.so` 版本必须与 pegasus 对齐**（否则 `VIP_ERROR_NETWORK_INCOMPATIBLE`）；产物是 `.nb`（NBG）。
- **W1 先打通最小通路**：拿一个 MobileNet 跑通 import→quantize→on-device，再上业务模型（该 NPU 工具链在同价位里最不成熟，时间风险大）。

---

## 8. 评估与现实目标

**三层口径**：验证集 → INT8 量化后 → **现场**（现场会显著掉，预留 domain-gap；曾有「99.5% 验证 → ~88% 现场」案例）。

| 指标 | 口径 | 目标（待实机校准，区间） |
|---|---|---|
| bird recall / animal mAP@0.5 | 验证集 | 检测召回优先（漏检比误检更伤级联） |
| 端到端误检(非鸟判 bird) / 漏检 | 现场 | 误检 < 2% / 漏检 < 5%（待复核） |
| 细分类 top-1 / top-5 | **地域过滤前/后对比** | 地域过滤后 top-1 显著优于全局；以实验定 |
| 延迟 | 实机 INT8（分检测/分类/前后处理） | 锚点：MobileNet 类 ~13ms@0.5T、yolov5@416 ~6FPS@0.5T |
| 内存峰值 / 24h 稳定性 | 实机 | 无 OOM / crash |
| **端到端节拍 / 帧率** | 实机（事件触发→出标签） | **≤200ms / ≥5fps@0.5T（待校准，B.7 硬门）**；常态 vs 事件触发分列 |
| 含运动门控的端到端 recall | 现场 | 对比「纯 NN recall」（门控会漏静止/慢动鸟） |
| 平均功耗 / 占空比 | 实机 | 按供电模式定（见附录 C.11） |

> 不承诺脱离实测的硬精度数字；以**实验矩阵（附录 B）+ 地域过滤后的现实目标**为准。端到端延迟预算分解见**附录 C.2**；运动门控对静止/慢目标的漏检需**周期性全帧扫描**兜底。

---

## 9. 鸟叫识别（可选模块）

> 结论：**硬件天然支持、模型可复用同一条 NPU 图像分类通路、架构走「独立并行 + 晚融合」**。**选型已更新**：**Google Perch v2（Apache 2.0，2025-08）**给出了可商用的强音频 embedding（近 15,000 物种 / ~10,000 鸟，EfficientNet-B3，embedding ~12M），**首选「Perch embedding + 自训地区线性头」**——既绕开 BirdNET（CC-BY-NC-SA，降为 benchmark），又比从零自训更强。原「真正的红线是没有可商用音频模型」的前提已被 Perch 解除；剩下的红线只是 BirdNET 衍生与音频**训练数据**许可。建议先做 PoC + 留干净锚点，不深做。

### 9.1 硬件：V85x 自带音频输入（零额外 codec）
- 片上**模拟音频 codec/ADC**：V851S/SE = 1× ADC(16/20-bit, 8–48kHz) + 1× 差分麦(`MICIN1P/N`)，SNR~95dB；V853/S = 2× ADC + 2 麦。四款均支持最多 **8 路 PDM 数字麦(DMIC)**。I2S：V851x 1 路 / V853x 2 路（仅接外部 codec / 麦阵时才需要）。
- 采样率上限 **48kHz** → 够鸟鸣（BirdNET 即 48kHz），但**抓不了超声/蝙蝠**。
- 软件：Tina Linux 标准 ALSA（`arecord -Dhw:audiocodec -f S16_LE -r 48000 -c 1`）。
- **mel/FFT 前端只能放 A7 CPU**（NPU 无 FFT 算子；E907 是 RV32IMAC 无 DSP 扩展）。成本小（A7+NEON，约单核个位数 %），但须连同视频编码负载一起实测。
- **A/V 并发**：音频 IP 与 CSI/ISP/编码物理独立、不争用（工程推断，**上线前实测**；厂商「全双工」指音频播放+采集，非 A/V 并发）。
- 选型：单麦够用 → V851S/SE 更省；要双麦/测向或两路 I2S → V853/S。

### 9.2 模型与通路：复用 NPU 图像分类路
- 范式 = **音频 → mel 频谱图 → 当图像 → CNN 分类**（Merlin Sound ID / BirdNET 均如此）→ 算子家族就是图像分类 CNN，**直接走相机现有 INT8 conv 通路，无需新 runtime**。
- BirdNET v2.4 参考：EfficientNetB0-like、~6500 类、~0.83 GFLOPs（算力毫无压力；约束是**模型尺寸/内存 + 算子覆盖**）。输入 3s@48kHz → mel 频谱图（**具体 bins×frames / 通道数以 BirdNET-Analyzer 官方 config 为准，§9.6 待核实**）→ NPU 输入 shape 与 RGB 不同，pegasus/awnn 输入层须按实测 config 配。
- **商用推荐路（更新）**：**首选 Perch v2（Apache 2.0）作 embedding 底座**——近 15,000 物种 / ~10,000 鸟、EfficientNet-B3（embedding ~12M / 分类头 ~91M）、社区已有 ONNX/TFLite。官方最佳实践 = **取它的强 embedding、在上面训一个轻量线性头**适配精选地域物种（它的 logits **未校准、稀有种不可靠**，须用自己的数据调阈值）。
- **from-scratch 退路**：仍可在**精选地域物种表（数十~数百种）**上**自训 EfficientNet-Lite0 / MobileNetV3-Large**（去 swish/SE，最 NPU 安全；与视觉路复用同骨干）。两条都**绕开 BirdNET 的非商用许可**。
- **上端注意**：Perch backbone 是 **EfficientNet-B3（带 SE/swish）**，直接上 V85x INT8 NPU 有回退软算子风险 → 端侧要么 op 实测，要么把 Perch 当 teacher / embedding 源**蒸馏到 NPU 安全的 Lite0 频谱图小头**再落端；音频独立晚融合、不争主 NPU 路。
- 可行性背书：BirdNET-Go/Pi 在 Pi 级 CPU FP32 50–550ms/3s（<1500ms 实时预算）；BirdNET-STM32 已 INT8 PTQ 部署到 NPU。
- 精度现实：精选数百种、干净叫声 top-1 ~80–90%；**真实重叠/噪声/远场会大幅崩**（BirdCLEF 现场 ROC-AUC ~0.52–0.56）→ 必加「非鸟/背景」拒识类 + 置信阈值 + 3s 窗时序投票。

### 9.3 数据与许可（承重红线）
- **Perch v2（Apache 2.0）= 现在有可商用的音频底座了**：权重 Apache、明确允许商用 → 解除了「没有可商用音频模型、只能从零自训」这个前提。⚠️ 训练数据含 Xeno-Canto/iNat 等，weights 授权为操作性 grant，**与 SpeciesNet 同属「输出授权灰区」**——蒸馏进出货权重时按下方铁律（teacher 含 NC → student 不可商用）谨慎核。
- **BirdNET 模型 = CC BY-NC-SA 4.0（非商用 + 传染性 ShareAlike）→ 不能出货，连微调衍生权重都被传染**。源码 MIT 但权重不是。商用唯一途径是找 Cornell 单独授权。→ **BirdNET 仅限内部原型 / benchmark**（Perch 已可替代其商用角色）。
- **Xeno-canto**：>100万录音 / >12900 种，但 **~99% 是 NC 和/或 ND，仅 ~0.7% CC0/CC-BY**。商用须用 API v3 按 `lic` **只留 CC0/CC-BY、硬排 -nc/-nd**，并维护逐录音署名清单。
- **BirdCLEF(Kaggle) = 研究专用**；**Macaulay = 授权付费**（仅补缺）。
- **taxonomy** 用 **eBird/Clements（带版本）**作规范键、与视觉物种集对齐；但 **eBird 数据/API/checklist 的商用性须法务确认**。
- 攒商用安全集：eBird 定种 → XC 只拉 CC0/CC-BY → 因池子 <1% 覆盖薄 → 重增广(SpecAugment/mixup/背景噪) + GBIF/自录补缺 → PTQ INT8。

### 9.4 多模态架构：独立并行 + 晚融合（不要紧耦合）
- **结论**：音频做**自包含管线**（mic→spectrogram→INT8 CNN→{species, conf, ts}），与视觉路解耦、可开关；只在上层加一薄层「**时间戳重叠**」晚融合。Merlin 的 Sound ID / Photo ID 就是两个独立模型，只共享 **位置+日期+eBird 先验**。
- **是否需要对齐**：**时间**只需粗粒度共现窗（不需信号级对齐）；**空间**单麦无方向 → 只能在物种/事件级融合。
- **三种输出诚实呈现，绝不把音频标签静默绑到画面里那只鸟**：①同窗同种既见又闻→升置信；②闻而不见（离框/遮挡/夜间）；③见而不鸣。
- **声学唤醒门控（强力省电杠杆，~100×）**：E907/A7 上跑 tiny 常开活动检测 → 有声学活动才唤醒 NPU 贵路径。对电池相机价值大。
- 音频是**补充非替代**（audio-only 在 SSW60 仅 ~33% top-1）。

### 9.5 落地建议
1. **可选模块、先 PoC 不深做**；晚融合只留薄薄一层时间戳重叠规则。
2. **先落「共享锚点」**：位置 + 日期 + 地域物种 allowlist + 统一事件时间戳——**对纯视觉的地域过滤(§5.4)同样立即有用**，并为音频零成本接入。
3. feature-level 紧融合**推迟**，且仅当干净精度成唯一瓶颈才做（明知其在缺模态/噪声下更脆）。
4. **商用许可硬门**：不出货 BirdNET 及衍生权重；训练数据只 CC0/CC-BY；eBird 商用性法务确认；音频底座**首选 Perch v2（Apache）embedding + 自训地区线性头**，自训 EfficientNet-Lite0 为 from-scratch 退路。
5. **早做工具链尽调**：pegasus 跑通 EfficientNet-Lite0 INT8 端到端、确认无算子 fallback、测掉点、profiling A7 上 mel 前端 + 视频负载合并成本。

### 9.6 关键待解（音频）
E907 是否真有 FPU（文档自相矛盾，决定能否 float-FFT）；V85x A7 上双 96×511 mel/3s 的实测成本；pegasus 对 EfficientNet-Lite0 全算子映射实证；Cornell BirdNET/Macaulay 商用授权条款；目标物种逐种的 CC0/CC-BY 现存量；出货麦配置（单麦无向 vs 麦阵测向）与续航预算；**Perch v2 的 EfficientNet-B3 在 Vivante NPU 的 SE/swish 算子覆盖**（端侧直跑 vs 蒸馏到 Lite0 小头）；**Perch 权重「输出授权灰区」对蒸馏出货权重的影响**（同 SpeciesNet）；Perch 的 mel / 输入 config 与端侧前端对齐。

---

## 10. 里程碑（建议）

| 阶段 | 内容 |
|---|---|
| W1 | **工具链通路打通**（最小模型 import→quantize→on-device）+ 数据/许可清册 |
| W2–3 | 检测器微调（COCO+OIV7[+LILA]）+ 量化 + 实机延迟 |
| W3–5 | 分类器微调（开源+本地+自采）+ 层级头 + 地域过滤 |
| W5–6 | 端到端级联 + 内存/稳定性 + 实验对比矩阵出报告（附录 B） |
| 探索 | 音频模块 PoC（视 §9）；自有数据集持续回采 |

---

## 11. 风险与缓解

| 风险 | 等级 | 缓解 |
|---|---|---|
| **NPU 工具链不成熟**（文档零散/版本易错配/Ubuntu-only） | **高** | W1 通路先行；锁定 pegasus↔VIPLite 版本；必要时绕开 Tina-SDK 自建 C app |
| 算子回退软算子/CPU（Focus/SE/transformer） | 高 | 选 anchor-free + 纯 conv backbone；逐算子实测；后处理放 CPU |
| **商用许可**（iNat 非商用 / Ultralytics AGPL / BirdNET 非商用） | **高** | 检测器用 Apache(NanoDet/PicoDet)；数据用 CC0/CC-BY+自采；音频用 **Perch v2(Apache)** / 自训 |
| **domain gap**（网图→真机，99.5%→88%） | 高 | 尽早自采真机帧；crop 域统一；强增强；最小尺寸门控 |
| 芯片未定（0.5T–1T / RAM） | 中 | 按 0.5T/128MB 下限设计；实验矩阵覆盖两档 |
| 小样本/长尾 | 中 | 微调+蒸馏+类平衡+地域过滤+层级回退 |
| **内存峰值 / 并发 OOM** | **高** | 串行加载 + buffer 复用 + 保守预算表(C.8) + 实测可用内存（拆静态驻留 / 并发瞬时峰值两条评） |
| **端到端延迟 / 帧率超预算** | **高** | 预算分解(C.2)；降帧/降分辨率/更小检测器；模型 load 开销单列 |
| **INT8 PTQ 掉点（细分类敏感，无官方 QAT）** | **高** | 校准集质量(C.7)；关键层 INT16 混合精度兜底；掉点直接影响片上/云分流比例 |
| 小/远鸟召回低 | 中 | 抬检测分辨率 / ROI / coarse-then-zoom |
| 回采数据隐私合规 | 中 | 人脸/车牌脱敏、采集告知、保留期(C.9) |
| 地域过滤收益不及预期 / eBird 商用受限 | 中 | 收益按区间(C.4)；GBIF 替代；法务前置 |
| OTA/版本契约缺失致静默错标 | 中 | index↔mask/taxonomy 同包版本契约(C.5) |
| 功耗/续航无目标致无法权衡 | 中 | 一级约束化(C.11) |

---

## 附录 A. 决策点（含默认推荐）

| 决策点 | 选项 | 默认推荐 |
|---|---|---|
| 芯片档位 | 0.5T/128MB ↔ 1T/≤256MB | 按 0.5T 设计，1T 留余量 |
| 检测器 | NanoDet-Plus-m / PicoDet-S / (YOLOv8n 研究) | **PicoDet-S 或 NanoDet-Plus-m** |
| 分类器 | EfficientNet-Lite0 / MobileNetV3-L / RepVGG-A0 | **EfficientNet-Lite0**，RepVGG 兜底 |
| 地域过滤 | 启用 / 不启用 | **启用**（推理期，低成本大收益） |
| 片上类规模 | 数据驱动数百级 + 层级/地域 | **数据驱动，不预设固定大数** |
| 音频模块 | 做 / 不做 / PoC | 硬件已确认支持(§9.1)；建议 **先 PoC**：**Perch v2(Apache) embedding + 自训地区线性头**（自训 Lite0 为退路）+ 晚融合 + 声学唤醒；**不出货 BirdNET** |
| 供电模式 | 电池/太阳能 ↔ 常供电 | **须显式声明**（决定功耗/分辨率/常开策略，见 C.11） |
| 帧率/节拍目标 | 区间(待校准) | ≤200ms / ≥5fps@0.5T 作硬门(C.2) |

---

## 附录 B. 实验计划与可执行清单

> 目的：用**受控对比**产出能说服 stakeholder 的证据。**所有实验共享同一份数据划分、同一评测脚本、同一量化流程，每次只变单一变量**，结果才可比。

### B.0 总原则
- 单变量受控：一次只改架构 / 输入 / 量化 / 档位 之一。
- 数据划分固定随机种子；**test 集仅在最终各跑一次**（避免过拟合到测试集）。
- 每个实验登记同一组列（见各表）；统一写入一张实验总表（CSV/飞书表）。

### B.1 统一基线（所有实验固定项）
| 项 | 基线设定 |
|---|---|
| 数据划分 | train/val/test = 固定 split + seed；类映射表版本化（eBird taxonomy v20xx） |
| 检测评测 | mAP@0.5、mAP@0.5:0.95、**bird-recall@IoU0.5**（最关键）、误检率、漏检率 |
| 分类评测 | top-1、top-5、**地域过滤后 top-1/top-5**、混淆对（易混种） |
| 系统评测 | 端到端延迟（拆：前处理/检测/crop/分类/后处理，ms）、内存峰值(MB)、并发稳定性、24h 无 OOM |
| 硬件档位 | {0.5T-128MB, 1T-≤256MB} 两档都测 |
| 量化 | PTQ，校准集 ≥500 张代表图（**须含夜视灰度/低照噪声/压缩伪影/运动模糊样本**） |
| 量化评测 | 每模型记录 FP32 vs INT8 掉点 + 逐算子是否 fallback（软算子/CPU） |

### B.2 实验一：检测器选型（变量 = 架构 / 输入分辨率）
候选 × 配置：`{NanoDet-Plus-m, PicoDet-S} × {320, 416}` + `YOLOv8n@416`(研究对照)
微调基线（示例，按各 repo 默认微调）：官方预训练加载 → 全量/分层微调，epochs 100、cosine + warmup、§6 增强、bird 统一单类、后处理切出图。

| exp-id | 架构 | 输入 | mAP@.5 | bird-recall | 延迟@0.5T | 延迟@1T | 内存 | INT8掉点 | 算子fallback | 许可 |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | NanoDet-Plus-m | 320 | | | | | | | | Apache |
| D2 | NanoDet-Plus-m | 416 | | | | | | | | Apache |
| D3 | PicoDet-S | 320 | | | | | | | | Apache |
| D4 | PicoDet-S | 416 | | | | | | | | Apache |
| D5 | YOLOv8n(对照) | 416 | | | | | | | | AGPL |

### B.3 实验二：分类器选型（变量 = backbone / 输入 / 蒸馏）
候选 × 配置：`{EfficientNet-Lite0, MobileNetV3-L, RepVGG-A0} × {192, 224} × {蒸馏 off/on}`
微调基线（示例）：timm 预训练 → 全量微调，AdamW lr 1e-3、weight_decay 1e-4、label smoothing 0.1、类平衡采样、§6 强增强、crop 外扩 10–15% padding、epochs 50(筛)/80(终)。

| exp-id | backbone | 输入 | 蒸馏 | top-1 | top-5 | 过滤后top-1 | 延迟@0.5T/1T | 内存 | INT8掉点 | 许可 |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | Eff-Lite0 | 224 | off | | | | | | | Apache |
| C2 | Eff-Lite0 | 224 | on | | | | | | | Apache |
| C3 | Eff-Lite0 | 192 | off | | | | | | | Apache |
| C4 | MobileNetV3-L | 224 | off | | | | | | | Apache |
| C5 | RepVGG-A0 | 224 | off | | | | | | | MIT |

### B.4 实验三：量化与端侧落地（变量 = FP32 vs INT8-PTQ；档位）
对每个入选模型执行 §7 链路并实机测：
- 记录：FP32→INT8 精度跌幅、逐算子 fallback 清单、实机延迟分解、内存峰值、(检测+分类+H.265 编码)并发是否稳定。
- **这是补「V851S 无公开 FPS 基准」缺口的关键实验。**

### B.5 实验四：地域过滤消融（变量 = 过滤 on/off）
同一分类器，推理期加/不加 eBird likely-species mask：记录 top-1/top-5 提升 + 候选集缩小倍数（全局 N → 区域 n）。

### B.6 结果汇总模板（交付 stakeholder 的一页纸）
| 推荐组合 | 检测器 | 分类器 | 档位 | 端到端延迟 | 内存峰值 | bird mAP | 种 top-1(过滤后) | 商用许可 | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| 方案A(均衡) | | | 0.5T | | | | | ✅ | |
| 方案B(高精) | | | 1T | | | | | ✅ | |

### B.7 选型决策规则（怎么读结果）
1. **硬门（先过）**：端到端延迟 ≤ 预算（如 100ms@10fps）、内存峰值 ≤ 实测可用、**许可=商用安全**、**无关键算子 fallback**。
2. **软优（再排）**：过滤后 top-1 最高 → 同档延迟最低 → 内存最小。
3. **降级路径**：0.5T 不达标 → 升 1T 档；仍不行 → 降检测分辨率 / 降帧率 / 缩类（地域包）。

### B.8 命令骨架（占位，按 repo/SDK 实际版本核对）
```bash
# 1) 训练分类器（示例: timm）
python train.py --model efficientnet_lite0 --pretrained --num-classes K \
  --input-size 3 224 224 --epochs 80 --opt adamw --lr 1e-3 --smoothing 0.1 \
  --aug-... (灰度/噪声/压缩/模糊/降采样)

# 2) 导出 ONNX + 静态化简化
python export_onnx.py --weights best.pt --imgsz 224 --opset 11
python -m onnxsim model.onnx model-sim.onnx --input-shape 1,3,224,224

# 3) pegasus 转换 + PTQ（检测器额外用 --outputs 切掉 decode/NMS）
pegasus import onnx   --model model-sim.onnx --output-model net.json --output-data net.data
#   检测器: pegasus import ... --outputs "<conv_out1> <conv_out2> <conv_out3>"
pegasus quantize --model net.json --batch-size 1 \
  --quantizer asymmetric_affine --qtype uint8 --dataset dataset.txt
#   dataset.txt: 每行 "./calib/xxx.jpg 0"; inputmeta.yml 设 scale=1/255 等
pegasus export ovxlib ... --pack-nbg-unify        # 产出 .nb (NBG)

# 4) 上板: VIPLite/awnn 加载 .nb; 检测后处理(decode/NMS) 在 A7 用 OpenCV
#   注意: VIPLite .so 必须与 pegasus 版本对齐 (否则 VIP_ERROR_NETWORK_INCOMPATIBLE)
```
> pegasus 子命令/flag 随 ACUITY 版本变动；以实际 SDK 文档为准。检测器评测延迟 = NPU 卷积部分 + CPU 后处理，须分别计时。

---

## 附录 C. 关键规格补全（承重缺口）

> 集中放置审稿发现的承重缺口，保持正文精简。标「待校准」者为需实机/法务确认的占位值。

### C.1 许可传染决策表（商用 + 微调场景）
| 上游要素 | 许可 | 商用出货 | 触发义务 / 备注 |
|---|---|---|---|
| Ultralytics YOLOv5/v8/11 权重 | AGPL-3.0 | ❌（除非购授权） | **仅微调、不分发代码也需授权**；衍生权重受限 |
| NanoDet / YOLOX | Apache-2.0 | ✅ | 保留版权声明 |
| PicoDet (PaddleDetection) | Apache-2.0 | ✅ | 同上 |
| timm 权重(MobileNetV3 / Eff-Lite / ShuffleNet) | Apache-2.0 | ✅ | 同上 |
| RepVGG | MIT | ✅ | 同上 |
| iNaturalist 数据 | CC BY-NC | ❌ 进出货权重 | **NC 传染**：用其训练/蒸馏的权重不可商用 |
| NABirds 数据 | 研究授权 | ❌ 进出货权重 | 仅研究 |
| BirdNET 权重 | CC BY-NC-SA | ❌ | NC + **SA 传染**，微调衍生也被锁 |
| **Perch v2 权重**（google-research）| **Apache-2.0** | ✅（带义务）| 可商用音频底座（替 BirdNET 商用角色）；保留版权声明；训练数据 provenance 属「输出授权灰区」，蒸馏出货权重按下方铁律谨慎 |
| COCO 图片 | Flickr ToU（杂） | ⚠️ 风险 | 标注 CC BY，但图片版权不统一 |
| Open Images V7 图片 | 逐图 CC-BY | ✅（带义务） | 强制署名+链接；Google 不担保，须复核 |
| Xeno-canto / LILA | 逐条/逐子集混 | 仅 CC0/CC-BY | 硬排 -nc/-nd；逐项登记署名 |

> 铁律：**换数据 / 换 head 不能洗白上游权重 license**；**teacher 含 NC 数据 → student 同样不可商用**。维护一张**署名与许可清册**（逐图 / 逐录音）随产物披露。

### C.2 端到端延迟预算与帧率（一级指标）
| 阶段 | 上限(ms, 0.5T, 待校准) | 备注 |
|---|---|---|
| 运动门控 | <2 | ISP 帧差 / VPU 运动矢量 |
| 模型 load/init（切换时） | 单列计 | 串行加载有 `.nb` load 开销，仅首次/切换发生 |
| 粗检测（NPU + CPU 后处理） | ~40–80 | 0.5T 上 yolov5@416≈166ms → 用更小检测器/降分辨率 |
| crop + 预处理 | <5 | |
| 细分类（NPU, 每只鸟） | ~13–25 | 多目标 ×N |
| tracking / 投票 / 门控 | <5（A7） | |
| **端到端节拍目标** | **≤200ms / ≥5fps@0.5T（事件触发，待校准）** | 升为 **B.7 显式硬门**；常态 vs 事件触发分列 |

### C.3 tracking & 时序聚合规格
- tracker：轻量 **SORT（IoU 关联 + 卡尔曼）跑 A7**，**仅 bbox 关联、不引 ReID**，给算力预算。
- 状态机：`max_age / min_hits` 控制 track 生死。
- 聚合：**per-track 累加分类 softmax 概率 → 归一取 top-5**；**置信门控作用于聚合后**结果。
- 去重：**同一 track id 只上报一次**；多目标各自独立 track。
- 验证：附录 B 增「单帧 vs track 投票」消融。

### C.4 地域 mask 生成 pipeline
- 数据源：eBird Status&Trends 栅格 **或商用更安全的 GBIF 替代**（**eBird 商用性须法务确认，未过则用 GBIF**）。
- 分桶：~50km hex 或 省/州级；阈值：累计频率覆盖 ~99% 的种入清单。
- 产物 schema：`{region_id, [species_taxon_key], version}`，OTA 下发。
- 无 GPS 设备：出厂 / 配网时指定地区入口。
- **`taxon_key → 分类头 index` 映射表随权重版本化**（防 index 漂移静默错标）。
- 收益限定：**取决于部署地物种密度与迁徙季，高密度地区增益有限**（待 B.5 实测）。

### C.5 OTA 升级与版本契约
- bundle = `{net.nb, taxonomy_map.json(ver), regional_mask(ver), min_runtime_abi}`，**整包原子下发**。
- **分类头 index 变更 → major 版本**；mask / taxonomy 必须同包、单调校验（防错标）。
- A/B 双分区 + 失败回滚 + 签名校验 + flash 预算 + 灰度。
- **VIPLite `.so` ↔ pegasus ABI 强绑定**：明确 runtime 是否随包升级及 ABI 判定（否则 `VIP_ERROR_NETWORK_INCOMPATIBLE`）。

### C.6 统一 crop 规范（训推共用同一函数）
- padding：各边外扩 **15%** → 取**最小外接正方形** → resize（或明确 letterbox）。
- 最小尺寸门控：**crop 短边 <32px 或框面积占比 <0.3% → 只报 `bird` 不报种**（阈值待校准）。
- 训推一致：训练统一用**同档检测器出框**；MegaDetector 仅初标，再过同一 crop 规范化。
- 多目标 crop 逐个串行。

### C.7 校准集与归一化规范（PTQ 静默掉点首因）
- 校准集**优先真机 / 现场回采**；早期不足用**真实退化**（实拍夜视、真实 H.264）而非纯合成，标合成比例上限。
- **检测器与分类器各建校准集**，给昼 / 夜配比。
- **归一化分写**：检测器多为 0–1（scale=1/255）；timm 分类器需 **mean/std**（或导出前 fold 进首层）。**训 / 推 / 校准三处一致性在 B.4 校验**。
- per-channel 量化是否被 pegasus 支持须实测；敏感层可 **INT16 混合精度兜底**；fake-quant 补救须用**真实 NPU 后端验证一致性**。

### C.8 内存预算（保守）与驻留策略
- 给保守预算表：系统(Tina) / ISP / 编码 / 推流 / A7 buffer / NPU 工作集 / **余量（标「待实测、可能逼近 0」）**。
- **串行加载 vs 双模型常驻做 A/B**；多目标会放大分类侧瞬时峰值。
- 风险：§11「内存峰值/并发 OOM」拆 (a) 静态单模型驻留可行性、(b) 并发瞬时峰值，均按 high 评。

### C.9 检测集合并 & 数据标注 / 隐私 SOP
- 合并：COCO/OIV7 → 统一大类**显式映射表**；OIV7 仅取目标类**穷尽标注子集**或 ignore-region 过滤；统一框格式 / 坐标系、去跨集重复图。
- 标注 SOP：检测 / 分类各自标注规范（框口径 / 遮挡截断 / 最小可标尺寸）；CVAT/Label Studio + 双标抽检 + 仲裁 + 一致性门槛；进训练集人工核验抽样比例与拒收标准。
- **隐私（商用硬风险）**：回采数据**人脸 / 车牌脱敏**、采集告知、保留期限（GDPR / 当地法规）。

### C.10 云锚点端侧契约（端侧现在就要写）
- 入队：**有界队列**（上限 + 丢弃策略）+ 断网落盘缓冲；`crop_jpeg` 规格（如 224² q80）。
- 触发：`TH_species` 先用全局常数（待 B 校准）；`rare_set` = 地域 mask 中频率低于阈值的种（同源生成）。
- 成本控制：**送云比例 < X%**、限频 / 批处理。
- 闭环：云返回结果作**弱标签反哺再训练**（留锚点、不实现云端）。

### C.11 功耗 / 续航（一级产品约束）
- 写明**供电假设**（电池 / 太阳能 vs 常供电）与目标：平均功耗(mW)、续航(天)、可接受占空比。
- §8 增「平均功耗 / 占空比」口径；声学唤醒(§9.4) 省电收益以此为基线量化。
- 若仅常供电，也需一句**显式声明**（否则「抬分辨率 / 双模型常驻」无法权衡）。

### C.12 实验可复现补全（接附录 B）
- 数据 split：给比例（如 70/15/15）+ **按种分层** + 每类最小 N 兜底。
- B.8 补**检测器训练栈**：NanoDet(`train.py` + `export_onnx`) 与 PicoDet(PaddleDetection `tools/train.py` + `paddle2onnx`) 各自命令与 batch/lr/输入尺寸。
- 蒸馏补温度 `T` 与权重 `α`。
