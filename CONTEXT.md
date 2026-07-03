# CONTEXT.md — 领域模型与当前焦点

> Stage 2「定靶」产物：沉淀领域语言 + 已锁定决策。配合 `docs/decisions/`（ADR）一起读。
> 文档背景见 `docs/plan-v2.md`（what/why）、`docs/engineering.md`（how）。

## 当前焦点（2026-06）

**问题**：在没有 V85x 板子的前提下，回答「**微调后的模型能不能达成目标**」。
**策略**：精度问题可在 dev/GPU（AutoDL 租卡）离线闭环；上板只影响延迟/内存/INT8 实测三列。
**进度**：检测 + 分类**双侧均已租卡首训**——检测 NanoDet-Plus 320/416 已训（bird 召回 fp32 87.6%↑ / int8 84.1%，命门守住）；
分类 v1/v2 crop 消融 + 区域评估已跑（field top-1 0.37→0.63）；可行性包络 ①→④、消融 harness、发布链（promotion）全接通；
地域过滤（US eBird 包）就绪；三份实验报告成文（`results/{detect,classify,实验1}`）。
下一步 = 补检测量化包络/gate 收口（缺端到端入口，见 `results/detect/feeder_320/README` 流程位置）+ 租卡跑余下
B.2 PicoDet 对照(#8) / B.3 distill(#7) / B.5 季节 mask(#10)。

## 可行性包络（「能否达标」的判定链）

逐级掉点，每一级都可在无板子下模拟：

| 级 | 量 | 怎么得到 | 可信度 |
|---|---|---|---|
| ① | FP32 验证集 top-1/top-5 | 真训真测 | 硬数 |
| ② | − INT8 PTQ 掉点 | ORT-QDQ static 模拟（ACUITY 同形态 ONNX） | 方向性可信，真数需板子 |
| ③ | − domain gap | 退化增强当「类现场」测试集（夜视灰度/低照噪声/H.264/运动模糊/远距降采样） | **代理估计，非真现场**（plan §8：99.5%→88% 教训） |
| ④ | + 地域过滤增益 | eBird likely-species mask，全局 N→区域 n | 真增益，区域相关 |

**交付形态**：逐级掉点表 + 诚实的现场区间估计 → 判断达标有没有戏、瓶颈卡在哪级。

## 已锁定决策（详见 ADR）

- **类规模**：训**全局头**（数据覆盖的最大类集），**保留地域清单接口做消融对比**（plan §5.4：训全局、推理期加 mask）。全局/地域两个数都拿得到。→ ADR-0001
- **达标口径**：**先建可行性包络看区间**，再定硬数；走一步看一步，不预设硬指标。→ ADR-0001
- **开发 vs 实施顺序**：开发按 plan 先搭框架（数据准备是其中一个*模块*）；「数据 ready 第一步」指**运行时**顺序，非「先下完所有数据再写码」。→ ADR-0001
- **第一份数据集**：本机 `~/Downloads/bird_species(DST1192)`（BIRDS-525 同源，525 类/16796 样本；taxon_key 已映射 408/525 eBird code）当开发夹具 + 首次可行性数据；**仅作可行性，不进商用权重**（版权未核，plan §5.2）。

## 关键术语（领域键）

**级联结构**
- **级联 (cascade)**：粗检测器（固定）→ bird crop → 细分类器（可 OTA 换/扩）。非鸟直接出大类。
- **统一大类 (unified coarse class)**：检测层 **5 个** feeder-cam 大类（`bird / squirrel / cat / person / other_animal`，[[ADR-0004]]）；跨集（COCO/OIV7/Caltech CT/NACTI）经各源映射表归一到此闭集；bird 在检测层为单一类，细分交分类器；person=补粮的人(检到→抑制告警)；other_animal=长尾哺乳动物合并(raccoon/rabbit/fox/dog/deer…)，避免稀有类样本饿死（实验1 skunk AP 1.3 教训）。旧 11 类口径已废（[[ADR-0004]]）。
- **crop 域 (crop domain)**：检测框按统一规范（padding 外扩 + 最小尺寸门控）裁出的图像域；训练裁剪与端侧推理裁剪共用同一函数。BIRDS-525 近似 crop 域。

**评估与门控**
- **可行性包络 (feasibility envelope)**：上面 ①→④ 的逐级估计链。
- **模拟 INT8 (simulated INT8)**：ORT-QDQ 在 FP32 ONNX 上的掉点估计，方向性信号；**真实 INT8** 只能来自板上 ACUITY，二者不等同。
- **likely-species mask (地域物种先验)**：推理期按地区(+季节)加载的「此地此时合理可能出现的物种」集合，作用于分类 logits。**软先验、非硬真值**：候鸟**含在内**（eBird「曾记录」清单或更紧的季节频率）；迷鸟/反季靠**置信门控 + 层级回退 + 云锚点**兜底；**训练不用**（全局头），仅推理注入，可 OTA 换。plan 称「最大精度杠杆」。
- **类现场 (field-like)**：退化增强模拟的测试分布，代理真机域；**不是**真现场。
- **质量门 (gate)**：对包络的 4 维阈值判定；`gate_pass` 是 candidate→stable 的硬闸；默认不设阈值（先看包络再定数）。
- **置信门控 (confidence gating) / 层级回退 (hierarchical fallback)**：按 crop 尺寸+置信度决定报到哪级；不确定沿 种→属→科→`bird` 回退，宁粗不错。

**规范键与产物**
- **taxon_key（规范键）**：物种规范键 = eBird/Clements 物种 code（带版本，[[ADR-0002]]）；跨集合并 + 地域 mask + 层级回退共用。未映射样本为空、不进地域先验。
- **manifest**：分类数据准备产物 + 训练/评估输入（固定 split + 类索引 + 逐样本溯源）；可移植（相对路径 + data_root 覆盖）。
- **打标契约 (labeling contract)**：LLM/人工批量打标的合法标签靶子 + 硬闸（检测锁 11 类闭集+bbox 合法性，分类锁类集）；越界/幻觉当场拒。
- **发布 (promotion)**：包络+门结论 → 固化 ModelCard（gate_pass/指标/溯源）→ registry →（过门）promote 到 stable channel → OTA。连接「评估」与「部署」。
- **FP32 ONNX**：上游唯一交付物（铁律）；INT8/.nb 交板上 ACUITY。

**框架架构（模型族扩展，[[ADR-0003]]）**
- **模型族 (model family)**：一类走全流程 `train→export→eval→publish→deploy` 的模型。当前两族对等：**检测**（粗分类+框）与**细分类**。加新族（YOLO 系等）= 加一个 adapter,不动 caller。
- **TrainerBackend**：train+export 的 seam（Protocol）。`train(cfg)->TrainedRef`（不透明句柄：分类=内存模型,subprocess=checkpoint 路径）+ `export_fp32_onnx(ref,...)`。每族一个 adapter（ClassifyBackend/NanodetBackend/YoloBackend），工厂派发。
- **共享脊柱 (shared spine)**：族无关、两族共用的下游链 `EvalReport→ModelCard(task)→registry→PackagerBackend→ChannelPolicy/OTA`。加族不重新发明它。
- **EvalReport（统一）**：`levels[].metrics: dict[str,float]` + `primary` 指标名;分类级带 top1/top5、检测级带 map/ap50/bird_recall。不分裂子类,指标进 dict。gate 泛化为「命名指标阈值集」。
- **CascadePipeline**：脊柱之上的组合层（不属任何单族）。`infer`（贯穿置信门控/层级回退）+ `evaluate`（检出率/级联 top-1/量化组合）,复用 crop 域。产品本体的代码归宿。
- **onnx_artifact 契约**：FP32 ONNX 铁律的唯一归宿。`export_fp32_onnx` 内总跑 `check_onnx_contract`（FP32+静态 shape,上 ACUITY 硬契约）+ 有 torch 模型时 `check_onnx_matches_torch`（数值对齐）。
- **数据脊柱 (dataset spine)**：共享 `Provenance/DatasetSpec` 基（`name/version/splits` + 逐样本 `source/license/taxon_key`），两族 manifest 继承：分类用 `DatasetManifest`,检测新增 `DetectionManifest`（包 COCO bbox + 同款溯源,替代裸 COCO labels.json）。provenance 一路流到 ModelCard（许可红线全链路可追溯）。检测 `bird` 粗类 ↔ 分类 species 经 [[ADR-0002]] eBird 键对接 → 级联在数据层即接得上。

## 当前不做（避免范围蔓延）

- 不碰上板/ACUITY 真量化（无板子；INT8 用 ORT-QDQ 模拟代替）→ issue #4（待板子）。
- 不做音频（plan §9 可选模块）。
- 不追商用数据合规闭环（当前数据仅作可行性）。
- 已 defer 的增量：蒸馏（#7）、PicoDet 对照（#8）、地域 mask 季节频率版（#10）。

> 注：检测器**已不在「不做」之列**——数据/config/评测/搭建脚本备齐（issue #2 已闭），且 320/416 两档**已首训**（见下 Changelog）。

## 变更记录（Changelog）

**2026-07-04 · 检测数据集管线完全重写**（[[ADR-0006]]：标准化获取 + 透明化溯源 + JSONL）
- **原则 D0**：不新旧并存——新管线上线即替换旧路径（经 critic 独立 verify：REVISE→并入 3 MAJOR + 遗漏项）。
- **透明化**：逐图 `author/original_url/source_media_id/asset_sha256` + 逐框 `label_provenance` 流；
  `license_manifest.csv` 扩 7 列 → **真兑现 CC-BY 逐图署名**（§4，此前只 source+license 不足）。
- **清旧路径（关 #18）**：打标契约 `detection.py` 迁 5 类；删 `detection_classes`(11类)/`merge_map`/
  `detection_ingest`/`detection_feeder.yaml`；`FEEDER5_CATEGORIES` 提为 contracts 层规范。
- **JSONL 唯一格式**：`DetectionManifest` → JSONL + `.meta.json` sidecar（移除旧单文件 JSON）。
- **acquire seam**：`AcquireSpec` 进 `DatasetSpec`（数据来源单一事实源）+ `adapter.acquire()`（manual
  校验/幂等/收据）+ CLI `acquire --list`（全源来源清单）；OIV7 折进 `_fetch` **删** `fetch_oiv7_direct.py`，
  4 源全声明来源。232 测试绿。
- **待续（需 box：网络/GPU）**：iNat S3 采集器(CC0/CC-BY) + MD 伪标注独立阶段 + Label Studio 人审、
  Roboflow bird-feeder（补 feeder 域）。

**2026-06-21~22 · 检测 5 类首训（feeder_320 / feeder_416）**（[[ADR-0004]]）
- **数据**：4 源经 DatasetAdapter 归一到 5 类闭集 → train **74,978** / test 24,791 / eval_feasibility 3,035（COCO，仅评估）。
  ENA24 + Caltech CT(ECCV18 sm) + OIV7 商用可训（OIV7 补 bird/person/平衡）；COCO `eval_only` 防许可传染。
  分布 bird≈other_animal 37~38k、person 30k、cat 8.7k、**squirrel 3.9k 最低**（数据可得上限）。逐图 `license_manifest.csv`。
- **训练**（3090 box · NanoDet-Plus/ShuffleNetV2 1.0x · fp32 · COCO 权重 warm start · batch96/AdamW 余弦）：
  feeder_320@30ep **mAP 0.459 / AP50 0.679**；feeder_416@70ep **mAP 0.498 / AP50 0.716**。
- **命门反转**：bird 召回 fp32 **87.56%**（实验1 64.5% → **+23pt**），int8(ORT-QDQ 模拟) 84.08%（−3.5pt，守住）；
  AP 偏低 = 320 定位不精 + 误报（召回涨、框准欠），喂鸟器「宁多框勿漏、后接分类器」可接受。FP32 ONNX 导出零损耗。
- **待收口**：量化包络/gate/检测总表缺端到端入口（类比分类 `run_envelope`，见 `results/detect/feeder_320/README` 流程位置）；
  PicoDet 对照(#8) 未跑。产物：`results/detect/{feeder_320,feeder_416}/` + `docs/detect/03-实操日志.md`。

**2026-06-20 · 可行性实验1 + 架构成熟化**（[[ADR-0003]]）
- **实验1**（双卡 GPU 全链路可行性）：分类 eff_lite0@224 test 0.921 / int8 −0.19pt（已发布 stable）；
  检测 NanoDet-Plus mAP 0.591 / bird 检出 64.5%；级联 fp32 0.858（int8 检测 −8.3pt，框质量经裁剪放大）；
  优化 A1/A2/A2v2/B 均未超基线（掉点=裁框信息损失+评估假象）。详 `results/实验1/实验报告.html`。
- **架构重构 P1-P5**：立 `TrainerBackend`/`Quantizer`/`Detector`/`Classifier`/ingest 五类 seam（Protocol+工厂+注册，
  config 切换）；`EvalReport` dict 化（检测进发布链）；`Provenanced`+`DetectionManifest` 数据脊柱（含蒸馏 soft_label hook）；
  `CascadePipeline` 升一等模块。零回归（旧产物兼容已验）。
- **架构审查 issue 收口**（#11-16，均 closed）：#12 级联真 OnnxClassifier/OnnxDetector(numpy NanoDet 解码)；
  #13 DetectionManifest 承重(from_coco/write_nanodet_labels)；#14 检测训练经 NanodetBackend；#15 ingest 注册表；
  #16 soft_label 决策保留为 hook；#11 regional 改 in-region on/off 口径(修 41.7% artifact → 真实 +1.3pt)。
- 至此：检测/分类两族对等、可串联(级联)、易扩展(加 YOLO=注册一 adapter)、蒸馏友好；177 测试绿。
