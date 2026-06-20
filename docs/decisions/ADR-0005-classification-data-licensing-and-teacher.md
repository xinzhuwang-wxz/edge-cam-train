# ADR-0005：细分类数据许可红线 + teacher + 区域策略

状态：Accepted（2026-06-21）。关联 [[ADR-0001]]（可行性优先）、[[ADR-0002]]（eBird taxonomy）、
[[ADR-0003]]（模型族 seam）、CLAUDE §4（许可红线）。落地见 `docs/classify/01·02`。

## 背景

为细分类（鸟种）做数据/训练方案前，对初稿做了**联网调研核验**（每条有出处）。发现初稿多处与"海外发行、
商用从严"的 §4 红线冲突，且部分是**反直觉**的（"开放许可"≠可商用、MIT 权重≠数据干净）。这些决策**不可逆**
（数据/teacher 选择会传染到 shipped 权重），故立此 ADR，避免后续会话重蹈。

核验出的关键事实：
- **iNaturalist Open Data 默认 CC-BY-NC**；"开放"只表示可再分发，**可商用是 CC0/CC-BY 的少数**，逐图 license 在 `photos.license`。
- **eBird 频率/丰度（Status&Trends）商用禁止**（Cornell 自定条款，需书面授权），**不能进发行产品**。
- **BioCLIP/BioCLIP2 权重 MIT 但训练数据含 CC-BY-NC 未过滤**（iNat ~86% NC、GBIF/FathomNet NC）→ 蒸馏污染。
- **NABirds（康奈尔）/ CUB-200（加州理工）非商用**，明确"不得做产品"/版权未清 → 连"为商用产品评测"都踩线。
- **SpeciesNet** 是 Apache、但 EfficientNetV2-M（~54M，服务器级），且无逐图 license 清单、输出授权未明（法律灰）。
- 端侧 backbone：实验1 实测 EfficientNet-Lite0 INT8 掉 0.19pt vs MobileNetV3 3.71pt（SE/h-swish INT8 不友好）。

## 决策

1. **训练数据只用可商用、逐图过滤**：iNat `photos.license ∈ {CC0,CC-BY}` + GBIF 逐 `media.license` 过滤 + 自建 feeder。
   权重天生可发行、无传染。逐图署名清册随产物披露。
2. **区域/季节先验从 GBIF CC0/CC-BY occurrence 自建**，**不用 eBird 频率数据**（商用禁止）。eBird 物种**代码**
   仅作对外稳定 ID 串（不涉频率数据）。区域为推理期 mask（OTA、不重训），评估必报 with/without。
3. **第一版不蒸馏**：BioCLIP/BioCLIP2 含 NC 数据 → **不当 shippable teacher**；SpeciesNet 待法务且服务器级 → 不用。
   若做蒸馏，teacher **只能自训于同一批 CC0/CC-BY+自建 clean 数据**。`soft_label` hook 保留（#7 未来）。
4. **NABirds/CUB 排除出商用线（含评测）**：用**自建 CC0/CC-BY holdout** 做主指标 + 跨源泛化；不引入这两者做任何
   产品相关评测。
5. **端侧 backbone = EfficientNet-Lite0**（INT8-only 平台实测最优）；Lite1/2 扩容；避 SE/h-swish/transformer/vanilla-RepVGG。
6. **区域落端：一个 pipeline 两用**——默认全局头 + 推理 mask（[[ADR-0001]]）；当某区域物种过多致 head 超端侧预算
   /精度不足时，同一 pipeline 训 region-filtered 专用模型（config 切换）。不做"每区域各自独立维护一套"的初稿方案。

## 结果

- 训练数据规模会**远小于** iNat 的 4 亿（可商用 Aves 子集，需自己数）——用质量/多样性 + 自建 feeder 补，不靠堆量。
- 失去 eBird 丰度的精度 → 用 GBIF occurrence 频率近似（略糙但合规）。
- 失去现成 SOTA teacher（BioCLIP/SpeciesNet）→ 第一版无蒸馏，靠 clean 数据直训；蒸馏待自训 teacher。
- 失去 NABirds/CUB 学术对标 → 自建 holdout，数字不与论文直接可比（可接受，产品看自有分布）。
- 两项需用户直接确认（联网查不到）：iNat **元数据本身 license**（邮件 iNat）；CC0+CC-BY 在 Aves 的**确切占比**（自数）。

## 备选（已否决）

- 直接用 iNat 全量 / eBird 频率 / BioCLIP teacher / NABirds 评测：均触 §4 红线或传染，发行不可用。
- 每区域独立模型（初稿）：N 套训练/维护/OTA 更重，与 [[ADR-0001]] 全局头+mask 冲突；改为"一个 pipeline 两用"。
