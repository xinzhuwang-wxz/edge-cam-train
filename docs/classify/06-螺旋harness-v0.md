# 螺旋 harness v0 + 分类活 roadmap

> 这是 harness 的**活文件**（会随每圈螺旋更新，非一次写定）。承接 [00-螺旋harness起始](00-螺旋harness起始.md) 的概念、[04](04-V861选型调研.md)[05](05-系统架构最佳实践.md) 的调研现实。v0 阶段先用本文档形态跑，跑通后再 skillify 固化。
> 定案人：AI 自主拍板（用户 2026-07-06 授权"你来推进"），冻结层改动才需回头开 ADR。

---

## A. 命门指标（harness 的锚 · 冻结）

**主命门 = 层级可用率（Hierarchical Usability）@ 校准可靠**
- 每个测试样本，模型输出它敢确定的**最细层级**（种 / 属 / 科 / bird）。
- **可用** = 输出层级正确（报种则种对；置信不足回退到属，则属对即可用）。
- **重罚"自信报错种"**：高置信却报错种 = 不可用 + 计入错误率（这是产品最伤的失败模式）。
- 可用率 = 可用样本 / 总样本。
- 依据：与 iNat roll-up / Seek rank-ladder 产品形态、plan-v2 分级门控、层级 aux loss 三者同构（05 模式 2）。
- **回退价值分层（2026-07-06 调研收口）**："回退算可用"**站得住**——产品上回退到属/科能给诚实有用的结果（iNat "we're pretty sure this is genus X" 范式，内容 roll-up Wikipedia，见 05 §11）。但**可用价值随层级递减**：命门宜从二元可用升级为**加权可用率**（种 1.0 > 属 ~0.7 > 科 ~0.4 > bird ~0.1，权重待标定），`hierarchical.py` 已分层计数（report_genus/family/bird），加权在其上算。**critical_error（自信报错种）恒为最重罚**——比"老实回退"差得多。

**配套尺子（并读，不单独定案）**：
- **区域内 top-1**（regional-masked，只在 in-region 子集）——地域场景真实精度。
- **校准 ECE + Reliability Diagram**（per-class Platt）——门控地基，稀有种过自信会让"高置信片上标签"悄悄出错。
- **Coverage@Precision=0.9 / AURC**——"保 90% 精度能覆盖多少查询"，拒识/回退的产品口径。

**过程代理**：种级 top-1/top-5 只作训练过程代理，**不定案**（同检测 round"val 峰作代理、held-out test 定案"）。

---

## B. 审查闸门（贯穿全程 · 每个"知道了新东西"的时刻过一遍）

四个 lens：
| Lens | 问 | 触发级别 |
|---|---|---|
| ① 服务命门 | 对**层级可用率/区域内精度**是正还是负？量级多大？值不值下一步成本？ | 反向且超阈 → 硬触发 |
| ② 合现实 | 数据/评测/裁框是否匹配观鸟器部署场景？（检测段痛点正源于漏此审） | 确证不匹配 → 硬触发返工 |
| ③ 守红线 | 触 §4 硬规则 / ADR-0001·0002·0005 / V861 包络 / 许可(NC传染) 了吗？ | 命中 → 立即停 |
| ④ 撬路线 | 这发现让 roadmap 某步失效/该前置/该加校验？发现更大杠杆？ | 是 → 生成 roadmap 增量 |

**触发规则**：
- **硬触发**（必须改）：红线命中 / 命门反向掉超阈 / 现实不匹配被确证 → 停当前 → 写审查发现落盘 → 改活 roadmap（返工/改序/开 ADR）。红线+改 ADR 走人确认（不可逆）。
- **软触发**（记录评估）：发现更优杠杆、假设被数据推翻 → 写 roadmap 增量候选，下一圈开头决策。
- 全程 harness 内驱动，产物落盘可追溯。

**已发生的闸门实践（示范）**：本轮调研已触发多次——芯片包络收敛"性能优先能上大模型"预期（①）、BioCLIP/eBird S&T 许可判红（③）、发现姊妹 taxonomy registry 改写 seam 债与检索库图景（④）、embedding FP32 出口是时间敏感 seam（④）。
- **2026-07-06 用户反馈"数据是大问题，检测那里花了很久，数据需要打磨"** → lens ①④ 触发 → **数据打磨从 R1.2 一个步骤，升级为贯穿 Round1/Round2 的主线**（见 D）。判据：分类数据比检测更难（细粒度标错毒性大 / 长尾结构性 / crop 质量耦合检测器 / 许可逐图+iNat NC 未决 / 域 gap 对细节更敏感 / 360→数千种规模），data-centric ROI 已被调研证 > 换 backbone（05 §六）。
- **2026-07-06 数据源选型 dogfood（harness 第一圈真实 dogfood）**：用审查闸门三 lens 评估所有可商用鸟类图源（见 [08-数据源审查](08-数据源审查.md)）→ 审出 iNat ToS 契约挡商用（连 CC0）、"好图 vs 现场"域 gap 本质、北美商用干净图缺口 → **回改路线：定 v1 首发欧洲、北美 fast-follow+gate、新增 Macaulay CC-BY 量化 / FeederWatch 北美先验 / 自采 feeder 三动作**。示范"审查判据从 did-it-work 升级为 does-it-serve-the-goal + 直接改活 roadmap"。
- **2026-07-07 用户定调「用螺旋 harness 重新认真做一遍分类，之前 360 类不纠结」+ 两配方决策落定** → lens ①④：
  - **前一版（feeder 360 / 0.748）降级为「可行性信号」**（证明 taxonomy 桥能通、0.748 可达），**不再当 bake-off 锚**（§E step3 从"基线锚"降为一次性 sanity）；新建以**发行地域（欧洲）干净数据**为地基从头做。
  - **决策：genus/family aux loss 进训练配方**（从 baseline 起，非 R2 晚期杠杆）——零推理开销、直接抬层级可用率（逼混淆收在正确属内→roll-up 更干净）；用 registry 属/科树算 aux 项，类集里未映射类的 aux 项 mask；**留一条 no-aux 消融行量化增益**（证据非信仰）。
  - **决策：roll-up 在端上做**（A7 CPU 后处理，阈值/策略当 OTA config）——保离线自足；确认 ADR-0008 §6 / ADR-0005 §2（后处理留 CPU）。
  - **澄清（写给后人，防误读）**：模型输出是**粒度一致的种级分布、永不降级**；层级回退 = 后处理 roll-up（端上/产品策略）+ **校准**（模型层，命门地基）+ aux loss（可选便宜杠杆）。命门评的是**整条栈给不给得出诚实答案**，非"模型多头降级"。
- **2026-07-07 执行落地（R1.1 桥 + 欧洲类集 universe）** → lens ②④：
  - **路线 A 用最新实践复核确认**（不推翻 ADR-0008 决策3）：SpeciesNet（相机陷阱同域生产：MegaDetector→全球分类器 + geofence + rollup）、iNat（55k + SINR）均为"一个并集模型 + 推理 mask"，非区域包；区域包是 Merlin 的手机+旅行场景选择，不适配我们**固定 GPS 单设备**。增量：K=0.02 小先验（+9% vs 硬 mask +6.6%）、固定 GPS 可"训并集/部署裁剪"后路。
  - **lens ② 发现**：bird-tagger registry **只有身份、无区域** → 区域必须另接 occurrence 源（eBird checklist / GBIF）；印证 05 §模式1"geo 独立一层"。
  - **registry vendored + 桥落地**：vendor registry 进 `data/taxonomy/ebird_clements_2025`（版本 pin eBird/Clements 2025.0）；建 `data/ebird_registry.py` + `Hierarchy.from_registry`（TDD 绿）——**填上命门 metric 缺的 registry→Hierarchy 桥**。
  - **欧洲类集 universe 落仓**：`registry × eBird 45 国 checklist 并集` → `data/region/europe/europe_species.jsonl` = **1,211 种 / 486 属 / 109 科**（ever-recorded 上界，含迷鸟/外来如鸵鸟）；命门 metric 已能在真实欧洲层级树上 roll-up；全在端侧 ~1,500 包络内。
  - **下一步**：逐种商用干净图覆盖审计（数据打磨主线）。
- **2026-07-07 R1.2 图像覆盖审计落地 + 用户纠序** → lens ②：
  - **用户纠序（关键）**：粒度主信号 = **逐种商用干净图量**，不是 occurrence 频次——迷鸟在原产地可能有大把图（鸵鸟 clean=438 实证），按 occurrence 先砍会误折叠有图的种。**「常规 vs 迷鸟/外来」划分放最后**当补充过滤，不当第一刀。
  - **图覆盖审计**（`scripts/audit_europe_image_coverage.py` → `data/region/europe/europe_image_coverage.jsonl`）：GBIF 全局 CC0/CC-BY StillImage **扣 iNat**（iNat 占 ~55–76%、ToS 禁商用，扣除是③红线实影响）。1,211 种 0 错 0 未匹配。
  - **粒度线画出**：≥500 图 468 / 100–499 358 / 30–99 211 / 1–29 155 / 0 图 19 → **826 种（≥100 图）够种级叶子**；174 尾部（<30 图）折叠，其中 105 种的属有「富兄弟」（折叠有意义）、69 种属也贫（→科/cloud）。
  - **caveat**：GBIF crosswalk 1190 EXACT / 17 HIGHERRANK / 4 FUZZY（21 非精确，多为近期 split，待清）；图级 license 下载时逐条再核（03 坑）；图量是"可得"非"已下"，实际训练量再打折。

---

## C. roadmap 冻结层 / 可变层

- **冻结（改需 ADR）**：goal、命门定义、§4 六条硬规则、ADR-0001/0002/0003/0005、V861 包络（INT8/纯CNN/内存克制/避transformer/后处理留CPU）、许可红线（CC0/CC-BY、避NC传染）。
- **可变（harness 每圈可改）**：backbone 档与分辨率、方法杠杆入队时机、命门阈值数值、数据源取舍（许可内）、置信门控阈值、**embedding-vs-softmax 大岔路**（可变但需 ADR 记 ONNX 双出口决策）。

---

## D. 活 roadmap（round1 定靶探路 → round2 富训收口）

沿用检测 round 的两段式。**当前 = Round1 第 1 圈。**

### 🅳 数据打磨（贯穿主线 · 分类段最大难题，用户 07-06 定调）
分类数据比检测更难，且 data-centric ROI > 换 backbone（05 §六）。贯穿始终、非一次性步骤：
- **数据源 + 首发地域（[08-数据源审查](08-数据源审查.md) 已定，2026-07-06）**：**v1 首发欧洲**（naturgucker/挪/arter CC-BY ~1.4M 支撑充分）；北美 **fast-follow + gate**（Macaulay CC-BY 量化 / 自采北美 feeder / iNat 澄清，三选一达标才上）。可商用图源 = 欧洲三源 + Macaulay CC-BY 子集 + Flickr/Wikimedia CC-BY + 自采 feeder；**iNat 仅 R&D**（ToS 连 CC0 都禁商用）。先验 = GBIF occurrence（含 eBird EOD CC-BY）+ **FeederWatch（北美喂食器专属）**。
- **现状诊断（R1 早期起）**：类分布/长尾曲线、Cleanlab 标签噪声率、**crop 质量审计**（检测器裁框好坏直接毒害分类输入）、许可覆盖（CC0/CC-BY 逐图 + iNat NC 未决）、**域 gap**（网络好图 vs 观鸟器现场：遮挡/背身/夜视/糊）。
- **持续打磨**：洗标（Cleanlab）→ 长尾策略（LA loss/解耦重训）→ 干净打标扩稀有种（Noisy-Student+QC）→ 真实 feeder 域补齐。
- **数据质量指标进命门配套**（数据是因、命门是果）。

### Round1 · 定靶探路（建尺子 + 数据诊断 + 选型）
| 圈 | 做什么 | 命门相关 | 状态 |
|---|---|---|---|
| **R1.1** | **接 taxonomy registry**（bird-tagger species.jsonl+rollup）填 ADR-0002 seam 债 + **建命门尺子**（层级可用率/校准/区域内/AURC）| 建"层级可用率"必需 taxonomy 树 | ✅ registry vendored + **registry→Hierarchy 桥 TDD 绿**（`data/ebird_registry.py`）+ **欧洲类集 universe 1,211 种落仓**（`data/region/europe/`）；命门 metric 已跑真实层级树。**待**：校准(ECE/Platt)/AURC 尺子 + occurrence 频次分层 |
| **R1.2** | **数据现状诊断**（🅳 现状诊断全项）——摸清**欧洲 1,211 种**数据脏在哪（类分布/长尾/许可/**crop 域 gap**）| 数据质量=命门的因 | ▶ **图覆盖审计已落**（826 种≥100图够种级叶子 / 174 折叠 / 扣 iNat）；**待**：occurrence 常规过滤（最后）+ 长尾类分布 + crop 域 gap |
| R1.3 | **backbone bake-off**（Lite4@256 / MNV4-Conv-M@320 / RepViT-M1.5@224，各训+INT8消融+算子回退+内存实测）按四维加权定档 | 命门天花板 | 待（尺子+数据就绪后）|
| R1.4 | **定 ONNX 双出口 seam**（softmax + penultimate FP32 embedding）——时间敏感，事后补贵 | 检索路线前置 | 待 |

### Round2 · 富训收口（甜点定了再富训 + scaling + 出货）
- 补数据（feeder 真实域）+ 方法杠杆按序：干净大模型打标(Noisy-Student)+QC → 在线 DKD。（**层级 aux loss 已前置进 baseline 训练配方**，2026-07-07，见 §B；不再作 R2 晚期单列杠杆。）
- geo_prior 自训"物种×地区×月份"时空先验（GBIF CC0/CC-BY，推理期软重排）。
- 量化包络 + gate 阈值定数 + 真实模型进 registry + 上板 spike（ACUITY/.nb/延迟）。
- embedding 检索兜底（云端 kNN + 与 KB embedding 表对齐，DINOv2 encoder）——按需。

---

## E. 第一圈（R1.1）任务分解

**目标**：没有对的尺子，后面 bake-off 没法比（同检测 round1"先建评测尺子"）。而"层级可用率"必须先有 taxonomy 树。

1. ~~**接入 taxonomy registry**~~ ✅（2026-07-07）：registry vendored 进 `data/taxonomy/ebird_clements_2025`（版本 pin）+ `data/ebird_registry.py` 的 `Hierarchy.from_registry` 桥（TDD 绿）。**类集不再是旧 360，而是 registry × eBird 欧洲 checklist 筛出的 1,211 欧洲种**（`data/region/europe/`，`scripts/build_europe_species_list.py` 可重建）。
2. **eval 建命门尺子**（TDD，本地）：`eval/metrics` 加层级可用率（沿 rollup 上滚 + 置信门）、`eval/calibration` 加 ECE/Reliability/per-class Platt、`eval/risk` 加 AURC/Coverage@Precision；区域内 vs 全局双报。
3. **拿现有 V2 Lite0 模型（0.748）过新尺子**：得到 baseline 的层级可用率/校准/区域内基线数——这既验证尺子、又是 bake-off 的对照锚。

**闸门自审**：每步问"这尺子真衡量'服务目标'吗"（如：层级可用率的回退层级判定口径对不对？校准在长尾尾端可信吗？）。

---

## F. v0 运行方式

- **本文件 = 活 roadmap 本体**：每圈结束更新 D 表状态 + 把审查发现写进 B 的"已发生闸门实践"。
- **基座循环**：{定圈目标 → 做 → 四 lens 审现实+命门 → 改本表 → 下一圈}。
- **何时 skillify**：R1 跑通 2–3 圈、闸门/触发口径稳定后，用 skillify 把循环固化为 skill、结合进 autoresearch（§五节奏，别先造完再用）。

---

## 七、新 session 起点（专注做分类 · 可直接接手）

> 分类段的调研 + 定靶已收官，进入执行。新 session 读本节即可上手，不必翻长对话。

**一句话现状**：命门 = 层级可用率；**v1 首发欧洲**；backbone bake-off = Lite4@256 / MNV4-Conv-M@320 / RepViT-M1.5@224；数据源已审（08）；用螺旋 harness（审查闸门 + 活 roadmap）驱动，每步过三 lens。

**先读（文档地图）**：本文（06 harness+命门+roadmap）→ [08 数据源审查](08-数据源审查.md) → [04 芯片+选型](04-V861选型调研.md) → [05 系统架构](05-系统架构最佳实践.md) → [07 竞品](07-竞品与端侧差异化.md) → ADR-0008（模型）/ ADR-0005（数据许可）/ ADR-0002（taxonomy）。审计基线见 01/02/03。

**已定**：命门指标（§A）· backbone 选型（04/ADR-0008）· 系统架构（05：端云 / teacher=DINOv2-Apache / geo 先验 / 层级回退 / ArcFace 一套两用）· 数据许可+源+**首发欧洲**（ADR-0005/08）· taxonomy 接 bird-tagger registry（ADR-0002）· 命门 metric `eval/hierarchical.py`（TDD 绿）。

**未定（open items）**：iNat 商用许可（红线，未解）· v1 精确类集（欧洲种清单，待 taxonomy+首发地域定）· 自采 feeder（未启动，域真实性唯一来源）· Macaulay CC-BY 子集量化（北美 fast-follow gate）。

**下一步（Round1，按 §D/§E）**：
1. **接 taxonomy registry**（bird-tagger `species.jsonl`+`rollup`）填 EbirdTaxonomy 占位 + 让命门尺子在真数据上跑（层级可用率需真实属/科树）。
2. **数据现状诊断**（现有 360 类：类分布 / 长尾曲线 / Cleanlab 标签噪声 / crop 质量 / 许可覆盖 / **域 gap** —— 给 ②合现实 lens 补量化指标）。
3. **backbone bake-off**（三候选各训 + INT8 消融 + 算子回退 + 内存，四维加权定档）。

**起始 prompt（可直接粘贴到新 session）**：
> 继续【分类】段执行。读 `docs/classify/06`（harness+命门+roadmap）+ `08`（数据源）+ `04/05/07` + ADR-0005/0008/0002 接手现状——命门 = 层级可用率（`eval/hierarchical.py`），v1 首发欧洲，用螺旋 harness 驱动（每步过审查闸门：①服务命门 ②合现实/域 ③守红线）。下一步 Round1：接 bird-tagger/taxonomy registry 填 EbirdTaxonomy + 现有 360 类数据诊断（脏度/许可/域 gap）+ backbone bake-off（Lite4@256 / MNV4-Conv-M@320 / RepViT-M1.5@224）。先做哪个你定，别先造完再用、每圈审"这真服务命门吗"、审出偏离就改活 roadmap（06）。
