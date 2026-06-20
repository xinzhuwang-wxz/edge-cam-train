# ADR-0005：细分类数据许可红线 + teacher + 区域策略

状态：Accepted（2026-06-21）。关联 [[ADR-0001]]（可行性优先）、[[ADR-0002]]（eBird taxonomy）、
[[ADR-0003]]（模型族 seam）、CLAUDE §4（许可红线）。落地见 `docs/classify/01·02`。

## 背景

为细分类（鸟种）做数据/训练方案前，对初稿做了**联网调研核验**（每条有出处）。发现初稿多处与"海外发行、
商用从严"的 §4 红线冲突，且部分是**反直觉**的（"开放许可"≠可商用、MIT 权重≠数据干净）。这些决策**不可逆**
（数据/teacher 选择会传染到 shipped 权重），故立此 ADR，避免后续会话重蹈。

核验出的关键事实（数字 = GBIF API 实测，occurrence-license proxy）：
- **iNaturalist Open Data 默认 CC-BY-NC**；可商用是 CC0/CC-BY 的少数（鸟类 ~16%：CC0 4% + CC-BY 12%，余 ~84% NC）。
- ⚠️ **iNat ToS 另禁"用任何 iNat 数据训练商用 AI"——连 CC0 都禁**；且 AWS Open Data 的**元数据表本身无 license 声明**
  （registry License 字段只覆盖图片）；ToS 与逐图 CC 许可的边界**官方含糊**（员工未回应）。→ 对商用发行是**红线级风险**。
- **可商用鸟图实测**：Aves 有图 42.6M → CC0/CC-BY ~737 万 → iNat ~482 万、**非 iNat ~254 万**；非 iNat 可用**实拍**
  ≈ naturgucker(德)671k + 挪 572k + 丹 138k ≈ **~1.4M（偏欧洲）**，CC0 大头是博物馆标本（域不符，弃）。**去 iNat 则北美薄**。
- **eBird 频率/丰度（Status&Trends）商用禁止**；但 eBird **原始观测（GBIF EOD 数据集）= CC-BY，可用**。
  GBIF Aves 带坐标 occurrence **~21 亿 CC0/CC-BY**（建区域/月先验池，只要种+经纬+日期、不要图）。
- **BioCLIP/BioCLIP2 权重 MIT 但训练数据含 CC-BY-NC 未过滤**（iNat ~86% NC、GBIF/FathomNet NC）→ 蒸馏污染。
- **NABirds（康奈尔）/ CUB-200（加州理工）非商用**，明确"不得做产品"/版权未清 → 连"为商用产品评测"都踩线。
- **SpeciesNet** 是 Apache、但 EfficientNetV2-M（~54M，服务器级），且无逐图 license 清单、输出授权未明（法律灰）。
- 端侧 backbone：实验1 实测 EfficientNet-Lite0 INT8 掉 0.19pt vs MobileNetV3 3.71pt（SE/h-swish INT8 不友好）。

## 决策

1. **iNat：R&D 先用，商用发行前必须解决许可（open-item）**。逐图仍过滤 `CC0/CC-BY` + research + Aves + species，
   用它把模型/流程做出来；**但上市前必须**（a）拿到 iNat 书面澄清 或（b）替换为非 iNat 源。**commercial-clean** 源 =
   naturgucker/挪/丹（CC-BY）+ GBIF 非 iNat CC0/CC-BY + 自建 feeder（这些无 iNat ToS 风险、权重无传染）。
   逐图署名清册随产物披露（GBIF 引 DOI）。
2. **区域/月份先验从 GBIF CC0/CC-BY occurrence 自建**（~21 亿池，**含 eBird EOD CC-BY 原始观测**）；**不用 eBird
   Status&Trends 丰度**（商用禁）。eBird 物种**代码**作对外稳定 ID。**不存原始 21 亿**——按目标种×区域×月聚合计数
   或 SINR 式采样训小模型 → 输出几 MB 表/网络。推理期软先验重排（OTA），评估必报 with/without + 不加过滤基线。
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
- **iNat 是 R&D 先用、上市前的 open-item**（书面澄清或替换）；占比已实测（鸟类 ~16% 可商用）；元数据 license 仍未声明。
- 去 iNat 则公开可商用实拍偏欧洲（~1.4M）→ 北美/长尾更依赖**自建 feeder**（权重更高）。

## 备选（已否决）

- 直接用 iNat 全量 / eBird 频率 / BioCLIP teacher / NABirds 评测：均触 §4 红线或传染，发行不可用。
- 每区域独立模型（初稿）：N 套训练/维护/OTA 更重，与 [[ADR-0001]] 全局头+mask 冲突；改为"一个 pipeline 两用"。
