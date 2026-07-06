# ADR-0008：分类模型架构与选型（V861 端侧优先）

状态：Accepted（2026-07-06，AI 自主拍板 + 用户对齐）。关联 [[ADR-0001]]（可行性优先）、[[ADR-0002]]（eBird taxonomy）、[[ADR-0003]]（模型族 seam）、[[ADR-0005]]（数据许可+teacher+区域）；**本 ADR 更新 ADR-0005 决策 5（backbone 档）、补充决策 3（teacher）、细化决策 6（区域）**。调研落地见 `docs/classify/04`（芯片+选型）、`05`（系统架构）、`07`（竞品）、`06`（螺旋 harness roadmap）。

## 背景

分类段像检测段一样重做，且**目标芯片从 V85x（0.5–1T）升级到 Allwinner V861（1 TOPS INT8）**，本轮**性能优先、内存/量化难度降级**（但"性能相近看次级变量"、"一个算法不独占内存"）。做了五路+竞品调研，核验出若干反直觉事实：

- **V861 = V853 同一 Vivante VIP9000PICO 血脉**：硬件 UINT8/INT8/INT16、**无 FP16/BF16**；官方仅保证"common CNN operators"、**未承诺 transformer/attention**；工具链 ACUITY/pegasus→.nb→VIPLite 不变、胶水可复用。算力天花板 ResNet-50 级（分类偶发触发，25fps 够）；内存是墙（INT8 权重宜 ~10–15MB）。**§4 四条硬规则一条不改。**
- **全行业观鸟器竞品清一色"端侧只动检+上传、识别放云"**（Bird Buddy/Birdfy/白牌），**无一款在片上做鸟种分类**。V861 硬件是行业标配（同族 V831/V853），但**片上做分类是差异化**。
- **iNat 权重（NC 训练数据传染）/ Merlin（专有闭源）/ BioCLIP（NC）都不能直接用**——本项目须自研训练。
- **超多类分水岭不是类数、是"类集是否开放"**：物种表是封闭策划集 → 分类头 + 地理先验 + 层级回退（iNat 8.5万种、Merlin 2000种验证）就是成熟解，端侧扛得住。
- **ArcFace 分类头权重 = 类原型**：一套嵌入可端侧 argmax 分类、云端 kNN 检索两用，化解"embedding vs softmax 岔路"。
- **teacher 判红看训练数据不看权重 license**：DINOv2（通用自监督、LVD-142M 非 NC、Apache）可当 teacher；BioCLIP（NC 鸟类数据）不可。

## 决策（模型部分）

1. **端侧优先做分类**：分类落 V861 片上 NPU（INT8），不上云；当地几百种**分类头**（非端侧检索）；云端只做**最小兜底**（稀有种/低置信/扩种）。这是相对全行业全云的差异化（离线/隐私/零云费/低延迟）。用户反馈回传闭环该建，但与分类在端/云无关。

2. **端侧 backbone = 纯 CNN，bake-off 定档**（**更新 ADR-0005 决策 5：Lite0 → Lite4 档**）。候选：**EfficientNet-Lite4@256（主推/保守，同族 pipeline+ACUITY 复用、INT8 掉 0.2%）** / MobileNetV4-Conv-M@320（体积优 9.7MB）/ RepViT-M1.5@224（激进 82.5%）。命门精度接近时按**四维加权裁决**（NPU 算子成熟度 > INT8 掉点 > 体积/克制 > 工具链成熟度）。**避 transformer/attention/SE慎用/h-swish实测**（V861 只保证 CNN 算子）；INT8-only，权重压 ~10–15MB。排除 FastViT/MobileOne（Apple 非商用）、ConvNeXt-V2（CC-BY-NC）。

3. **类数 / 地域策略：路线 A 优先、B 兜底**（**细化 ADR-0005 决策 6**）。
   - **A（首选）= 一个"发行区并集"大模型 + 推理期 geo/月份 mask**：训一个覆盖所有发行区物种并集的模型（首发欧洲+北美并集 ~1300–1500 种，端侧扛得住），推理按用户位置+月份 mask 收窄到当地当季几十种。维护一套、OTA 简单、跨区可用。
   - **B（兜底）= 区域包**（Merlin 式，按大区切不同类集）：仅当发行区扩到并集几千上万、端侧一个模型装不下/精度掉太多时启用。
   - **两者都不是冗余多塔**（多塔 = 同一物种集训多个地域专用版，否决）。类集由**发行地域清单**（taxonomy registry + GBIF occurrence）定义，非"数据源恰好有的种"。全球 11,167 种是**云端兜底天花板**，不进端侧模型。

4. **teacher / 蒸馏：不必需，按序上**（**补充 ADR-0005 决策 3**）。先直接训 baseline（timm ImageNet 预训练 Lite4 + CC0/CC-BY 鸟种数据，无 teacher）。提分优先级：**干净大模型打标（Noisy-Student）+ Cleanlab QC > 层级 aux loss > 在线 Decoupled-KD**。teacher **只能** DINOv2（Apache，通用自监督非 NC）或自训 clean 集成；**禁** BioCLIP/iNat/BirdNET 权重（NC 传染）。DINOv3-ConvNeXt 架构诱人但自定义许可，待法务。

5. **地理/月份先验：独立模型 × 视觉晚融合乘法，推理期，留 A7 CPU**（落实 ADR-0005 决策 2）。用 SINR/`geo_prior`（MIT 代码，geo_prior 原生带 day-of-year 月份维）**自训** on GBIF CC0/CC-BY occurrence（含 eBird EOD CC-BY）。`P(种|图,位置,月) ∝ P(图|种)·P(种|位置,月)`，**不进 NPU 主干**、可 OTA。月份 = 同一时空先验的输入维（非独立系统）。地域清单外种给小先验 k（非清零）。**禁 eBird Status&Trends 丰度**（商用）。

6. **层级回退：分数上滚（roll-up）+ 置信门**。种→属→科→bird，低置信回退到敢确定的最细层级；训练期可加 genus/family aux loss。**命门 = 加权层级可用率（种 1.0 > 属 > 科 > bird）+ 区域内 top-1 + 校准（per-class ECE/Platt）**；critical_error（自信报错种）最重罚。回退卡片内容 **roll-up Wikipedia 属/科条目**（CC-BY-SA），代表图指种级旗舰种，不为属/科手写百科。

7. **embedding/检索（演进路，非 v1 必做）**：训练用 **ArcFace/margin-softmax → 一套嵌入两用**（端侧 argmax 分类主力 / 云端 kNN 检索兜底扩种，加新种只加向量不重训）。**ONNX 保留 penultimate FP32 embedding 出口**（时间敏感 seam，事后补贵）。embedding 对 INT8 敏感 → 须专列"INT8 检索召回"量化评估。

8. **taxonomy 骨架接姊妹 registry**（落实 ADR-0002）：接 `bird-tagger/taxonomy`（11,167 种 + rollup + genus/family/order），`ebird_code` 为规范键；填 EbirdTaxonomy 占位、供层级回退与跨源合并。

9. **不可直接复用外部模型**：iNat 权重（NC）、Merlin（专有）、BioCLIP（NC）均不可 → **自研训练**（timm 预训练 backbone + CC0/CC-BY 数据 + 可选 DINOv2 蒸馏 + 自训 geo 先验）。iNat/Merlin 仅作产品范式参照。

## 结果与影响

- **上游只产 FP32 ONNX**（含 softmax + embedding 双出口），INT8 交板端 PTQ（§4.1 不变）；后处理（mask/回退/geo）留 A7 CPU（§4.2 不变）。
- **roadmap（06）**：R1 = backbone bake-off + 建命门尺子（层级可用率/校准/AURC）+ 数据现状诊断；R2 = 富训 + 方法杠杆（打标/aux/DKD）+ geo 先验 + 量化包络 + registry 收口 + 上板 spike。
- **待上板 spike 定案**（板子到手）：真实 INT8 掉点 / SE·h-swish 算子回退 / 真机 fps / embedding INT8 召回——这些是当前离线阶段的最大盲点。
- **仍开放（open-item）**：iNat 商用许可（ADR-0005 决策 1 未解）；DINOv3/BioCLIP 若动用须法务；首发发行地域清单待定（影响类集）。
