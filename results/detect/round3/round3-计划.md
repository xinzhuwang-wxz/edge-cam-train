# 检测 Round3 · 针对性改进轮 · 计划（定稿 v1，含 2023-2026 调研）

> **定位**：round3 ≠ 重新选型。它是在 round2 部署点 **main（NanoDet-Plus-m 416, 1.0x, test bird 85.0）**
> 上**热启动续训**的**缺陷驱动**改进轮——修 round2 暴露的具体病，尤其**类间混淆（站立松鼠→鸟）**、
> **远景/中鸟覆盖**、**次要类泛化**，并**首次把"检测混淆矩阵 + 类判别"做成一等评估指标**。
> 架构冻结（1.0x / 416 / GhostPAN 96-neck / 5 类）→ round2 权重全量加载。
> 相关：`results/detect/round2/训练监控日志.md`（基线）· [[round2-数据计划]] · [[feeder-scale-target-scope]] · [[ADR-0007]]。
> 调研原始件：本会话后台 agent（21 次检索、带引用），要点已折入下文各节。

---

## 0. TL;DR

**主线**：热启保命门（bird 不退），**数据中心**三管齐下砸"松鼠判别 + 类间分离 + 鸟尺度覆盖"，评估层第一次把"对角线大不大"量出来（且 fp32 / int8 都量）。

### 只做 3 件事（最高 ROI，全约束干净，全 data-centric）
1. **回捞 round2 假阳性 + 加跨类硬负样本图**，热启重训。用 round2 权重捞"松鼠/猫→bird 且过阈"的帧核对后按正确类喂回；再加"只有松鼠 / 只有猫 / 空喂食器"的**负样本图**（图里无 bird 框，占比 ~10%+）。→ 直接把混淆矩阵那一格变小、把置信度拉开（利用 sigmoid-per-class 事实，见 §2）。
2. **喂食器域"中大化"所有类**（Bird Buddy 洞察）：找 feeder 域图 + **crop-to-feeder-framing**（把相机陷阱小松鼠/猫 zoom-crop 成中大近景，§A6）+ copy-paste。**主攻各生物在喂食器场景的中大样子**；远景鸟仅受控次要补充 + 化解"小 blob→鸟"先验（§3）。
3. **搭可回归的判别评估（fp32 + int8 都算）**：逐类归一化混淆矩阵 + TIDE `Cls` 误差 + top1-top2 置信度 margin 直方图。→ 让前两件事可量化、可逐 round 回归，并第一时间抓量化期类判别塌缩（§6）。

### 明确不做（调研反证）
- **mixup** —— GhostNetV3(2024) 实证对 **ShuffleNetV2 级紧凑模型有害**（NanoDet 骨干正是 ShuffleNetV2 1.0x），直接命中，避开。
- **换 Seesaw / EQLv2 损失** —— 为 softmax 互斥类设计，与 NanoDet 的 **sigmoid+QFL 联合表示冲突**，且破坏 QFL↔DFL 耦合、威胁 warm-start。**保 QFL 原样**。
- **SAHI 推理切图** —— 破坏单发 416 + Vivante NPU 部署（只能当训练期数据手段）。
- **大幅 rotation / 大幅 LSJ** —— 破坏喂食器域姿态/尺度先验、伤中大目标工作点。
- **上游 QAT** —— 与硬规则"上游只产 FP32 ONNX、INT8 交 ACUITY/pegasus"冲突（§7）。

---

## 1. 诊断靶单（round3 要打的病）

**用户明确提的：**
1. **远景鸟太少**（round2 用 `min_box_area_frac=0.005` 滤掉 OIV7/iNat 的 <0.5% 远景鸟）。
2. **观鸟器"中鸟"不足**。3. **其它类别 feeder 场景样本不足**。
4. **★站立松鼠→鸟**（头号靶）。5. 增强确认（见 §4）。
6. **★松鼠识别到位 + 类间置信度拉开**（混淆矩阵对角线大、松鼠 vs 鸟敏感）。

**我从 round2 记录 + 本轮实测补挖的：**
7. **次要类 val→test 崩**：squirrel 90.7→51.4、cat 95.2→54.5。同根：量少 + 域窄。
8. **squirrel 尺度畸形**（train 实测）：**46% 挤在 0.5-2% 小/中远**、大目标（>6%）仅 30%、总量最少 3105 框——**最像栖枝鸟的"大尺度直立姿态"最稀缺**。
9. **零 hard negative**：负样本只有 Caltech 空帧，没有"易误报成鸟"的干扰物（round2 conf0.4 时 7.5% 帧误报）。
10. **从没算过混淆矩阵**（`analyze.py` 只是分类版），用户诉求当前无法量化/回归。
11. **可复用未用源**：卡上 `roboflow_squirrelgarden`（1176 图）round2 没用。

**尺度基线（train 实测，供定量）**：bird 大52.7/中24.4/中远20.2/远2.7% · squirrel 大30.1/中17.7/中远46.2/远5.9% · cat 大17.3/中63.2。

### 1.1 ★M1 实测修正（2026-07-07 无卡跑出 round2 混淆基线，`round2_confusion_baseline.md`）
**跑出真实混淆矩阵后，靶单重排**（这是"看实际数据调整计划"的落地）：
- **"松鼠→鸟"argmax 混淆量级小但非零、且未被低估**（critic Finding 1 已用数据复核）：worksite 0、test 24/1123（2.1%）；
  **共激活复核** squirrel 24 = 矩阵值 → 贪心匹配没系统性低估；合计非鸟区域点亮 bird 仅 104。**但只覆盖 NMS 存活的独立 bird 框，
  "同框 bird 紧随第二"的 margin 现象看不到（须 GPU）**。用户 dogfood 的"站立松鼠→鸟"最可能在**未标注野图**或是 **margin**（§2）。
- **真正可测短板 = ① 次要类 recall 低**（test squirrel 44%/cat 40%，大量漏检→bg）**② other_animal 黑洞**
  （cat→other **271**、squirrel→other **164**，catch-all 吸收次要类，**新发现**）。
- **⇒ round3 重心微调**：从"别把松鼠叫成鸟"改为**"把松鼠/猫检出来（recall）+ 与 other_animal 判别分离"**；
  **top1-top2 margin 指标升为高优**（唯一能量化用户"置信度都差不多"的工具，需 GPU dump 全类分）；
  **补部署域带标 eval**（量野图里的真实混淆）。数据动作（§3 加源/crop/hard-neg/RFS）方向不变，**验收指标改以 recall + margin + 对 other 分离为主**。

---

## 2. ★根因洞察：为什么"触发后 label 置信度都差不多"

调研核对官方 config 确认：**NanoDet-Plus 分类分支 = per-class sigmoid + QualityFocalLoss（联合表示），不是 softmax**。含义：**bird 头与 squirrel 头之间没有 softmax 那种天然互相压制**——一团"站立松鼠" blob 可以同时点亮 squirrel 头和 bird 头且互不扣分。这就是"几个 label 置信度都差不多"的**结构性根因**。

**推论（决定 round3 战术）**：拉开类间置信度**必须靠数据显式提供跨类负例**（让 bird 头在"松鼠区域"吃到明确负梯度），**不能靠换损失**。这条贯穿 §3 全部数据动作，也解释了为什么 §5 保 QFL 不动。

---

## 3. 数据层（A，本轮主战场，data-centric）

> 机制复用 ADR-0006 管线（`acquire → build → gate`），加源=改 config、零改代码。build 尾照跑 `data/gate.py` 6 项硬门。

> **★域锚点（Bird Buddy 洞察，2026-07-07 用户）**：喂食器/观鸟器是"生物凑到镜头前吃东西"的场景——
> **鸟、松鼠、猫爬上喂食台时，在画面里全是中大近景目标**。所以 round3 数据的**主力 = "各生物在喂食器
> 场景的中大样子"**（bird 中大 + squirrel/cat 爬上喂食器的中大 + 其它），远景只作**受控次要补充**（§A3）。
> 这把 [[feeder-scale-target-scope]] 与"补更全鸟模式"统一了：**域=中大近景，主攻中大，远景少量提鲁棒**。
> 落地两条腿：**① 找 feeder 域图**（scout + acquire）+ **② 从现有图 crop 出中大近景**（§A6，把相机陷阱的小
> 松鼠/猫转成喂食器域中大）。参照物：Bird Buddy / Netvue Birdfy 等智能喂食器的真实场景图。

### A1. 回捞假阳性 + 跨类硬负样本（★最高 ROI，直击混淆）
- **FP-mining**：拿 round2 main 权重在训练/未标注池上推理，专捞"squirrel/cat 被判 bird 且置信度过阈"的帧 → 核对标签 → 按正确类喂回。把混淆矩阵最亮的非对角格直接变成负梯度。（anchor-free 无 RoI 阶段，经典 OHEM 不适用，但**数据级 hard-example mining 完全适用且更可控**。）
- **跨类负样本图**：加"只有松鼠 / 只有猫 / 空喂食器"的图，各自类标注、**图里无 bird 框**。因 bird 头独立 sigmoid，这是喂给 bird 头的明确负例（背景负样本图实证可降 FP 80%+）。**红线：负样本图里绝不能有未标注的鸟**（否则变漏标假阴）。建议负样本占比 ~10%+。
- 引用：OHEM(1604.03540) · Hard FP Suppression(1810.04002) · 背景负样本−80%FP(2604.02282)。

### A2. 松鼠判别（round3 主攻）
- **加 `roboflow_squirrelgarden`**（卡上 1176 图，未用）+ 逐源核许可。
- **定向补"站立/直立姿态大尺度松鼠"**——纠正 A8 的尺度畸形，正打混淆根因。
- squirrel 总量 3105 → **追平次要类（目标 ~6–8k，见 §10 决策）**，且补大尺度尾。

### A3. 鸟的尺度覆盖（**中大近景为主，远景受控次要**）
- **主力（对齐域锚点）**：补**喂食器场景中大近景的鸟**（"鸟落喂食台吃食"），feeder 域源扩量 + §A6 crop。这是部署工作点，权重最大。
- **次要（受控补鲁棒）**：松开 `min_box_area_frac` 地板（0.005→ ~0.002，保留非零地板挡不可标注斑点），把被砍的**远/中鸟收回一部分**（"学更全鸟模式"），**设远景配额上限**，不喧宾夺主。数据已在手、成本近零。
- **⚠️ 硬权衡**：回来的小 blob 正是加剧"松鼠→鸟"的东西 → **必须与 A1 配对**（回捞 FP + 跨类负例）对冲，用尺度分层 AP + 混淆矩阵双指标盯；大目标 test≥90.1 一旦退就回调远景配额。

### A4. copy-paste（小目标召回主力，鸟/松鼠等量）
- 小鸟 crop 多尺度/多位置贴到喂食器背景，精确控制小实例数量/尺度/位置。
- **关键：给松鼠/猫做等量小尺度 copy-paste**——避免只放大"小 blob→鸟"先验，让易混类在小尺度同增。
- 引用：Simple Copy-Paste(2012.07177) · Context-Aware Copy-Paste(2407.08151) · 小目标综述 2023-25(MDPI 15/22/11882)。

### A5. 采样：类平衡（数据层预过采样 = RFS 等效，见 §8）
- squirrel 最少 → 梯度弱 → sigmoid 下最易被 bird 头点亮。**RFS 等效 = build 阶段按类频率物理复制含稀有类的图**进训练 COCO JSON。
- ⚠️ **critic 纠错**：NanoDet 子进程/独立 env 训练，**注入不了 Lightning sampler**（原写法错）→ 只能走数据层预过采样（§8）。别过采样过头致过拟合。
- 引用：Instance-Aware RFS(2305.08069) · 类不平衡实证(2403.07113)。

### A6. ★喂食器域"中大化"——crop-to-feeder-framing（所有类，Bird Buddy 洞察）
把**现有已标注的动物**（相机陷阱小/中松鼠 3105、猫 3674、宽幅鸟）做**动物为中心的 zoom-crop**：以动物框为中心裁一块使其占画面 ~20–60%（中大近景）→ resize 416 → 该实例"表观尺寸"变成喂食器域中大。**用已有数据把"相机陷阱域小松鼠/猫"转成"喂食器域中大松鼠/猫"**，直接对齐部署域，且**爬喂食器/站立的中大松鼠正是判别最要紧的样本**。
- **与 copy-paste(A4) 的分工**：crop=在原图内 zoom（真背景/真光照，最逼真，但受限于已有背景）；copy-paste=贴到新背景（更灵活，但要防不真实合成）。**两者互补，都产中大近景实例**。
- **实现**：优先**离线预裁**（每个 crop=一条派生 record，带源 provenance/许可，进 gate 可审）；框裁后重新校验有效。可作 `data/` 一个纯函数 + 测试（无卡可做）。
- **⚠️ 坑**：**不要把过小的源动物强行放大**（20px 松鼠 upscale 到 400px 会糊+假细节）→ 设**源最小分辨率门**，只裁本就够清晰的；保持长宽比、框裁后仍有效；背景多样性别塌（别全裁同一批相机陷阱）。
- 引用：与 A4 同族（Simple Copy-Paste 2012.07177 的 scale-jitter/crop 思想）；随机缩放裁剪是检测标准增强。

---

## 4. 增强层（B）——回答"有没有 flip/rotation/scale" + 定稿

**round2 现状（实测）**：flip 0.5、scale[0.6,1.4]、stretch[0.8,1.2]、translate 0.2、亮度/对比/饱和**有**；**rotation / shear / perspective 全 0（关）**；**无 mosaic/mixup/copy-paste**。
> NanoDet pipeline 的 `rotation/shear/perspective/scale/multi_scale` 键**原生现成**，开=纯改 config；mosaic/copy-paste 官方不带，需**自实现 transform wrapper**（扩展点，不 fork，也**不引 AGPL 的 Ultralytics 代码**，只借算法思想）。

| 增强 | round2 | round3 定稿 | 依据 |
|---|---|---|---|
| flip / scale / stretch / translate / 色彩 | 有 | **保留** | — |
| **multi_scale / scale jitter** | [0.6,1.4] | **适度加宽 ~[0.5,1.5]**（帮远/中鸟尺度泛化），最后 N ep 收窄 | 中等 SSJ 优于全 LSJ（小模型+warm-start+有限 epoch） |
| **rotation** | 0 | **小角度 ±5~10°**（建模相机微倾，安全）；**禁大旋转**（鸟不会上下颠倒） | 喂食器域姿态先验 |
| **shear / perspective** | 0 | **轻微 ±2~5°**（可选） | 同上 |
| **★copy-paste** | 无 | **加**（小鸟 + 松鼠/猫等量，见 A4） | 本问题最高 ROI 现代增强 |
| **mosaic** | 无 | **第二梯队**：一箭双雕（§2 跨类共现 + 小目标）；**必须 close-mosaic 最后 15ep**；承担小 blob 混淆权衡 | YOLOX close-mosaic |
| **mixup** | 无 | **不做** | GhostNetV3：对 ShuffleNetV2 有害 |

> 落地节奏：copy-paste + rotation/多尺度先上（低风险）；mosaic 作为**消融项**在前 3 件事完成后再引入（§9）。

---

## 5. 损失 / 采样层（C）——保 QFL 不动

- **损失不改**（见 §2/§0）：保 `QualityFocalLoss + DistributionFocalLoss + GIoULoss` 原样。Seesaw/EQLv2/class-margin 与 sigmoid-QFL 冲突、威胁 warm-start，**收益低风险高，不上**。
- 类判别改进**全部放到数据侧**（§3 的 A1/A2/A4/A5）——这是本模型正确的杠杆点。

---

## 6. 评估层（D）——新增一等指标（★可无卡现在就建）

round2 只有逐类 AP，**round3 先把"类判别"变成可量化、可回归**，否则用户诉求无法验收。在**固定 test 集 + 部署置信度阈值**下产三件套，**fp32 与 int8 各一份**：

1. **★检测混淆矩阵（新建）**：pred↔GT 按 IoU(≥0.5)+类匹配；未匹配 pred=background 列(FP)、未匹配 GT=background 行(漏检)；**按行归一化**看"squirrel 行漏到 bird 列多少"。核心 KPI=**对角线占比**。现有 `analyze.py` 分类版不适用 → 在 `eval/detect_metrics.py` 加检测版（**纯函数 + 测试，红先绿后**）。
2. **★TIDE `Cls` 误差**：把"框对类错（与错类 IoU≥阈）"压成单个可回归数——即"松鼠↔鸟混淆"的定量值。`pip install tidecv`，低成本。
3. **置信度分离 + 校准**：**top1-top2 margin 直方图**（最高类分−次高类分，直接量化"不能都差不多"）+ **D-ECE**（`netcal`，防"分拉开了但都过自信"）。

**回归护栏（硬）**：① bird 命门 test AP50 **≥85.0**、大目标 **≥90.1**（不许把命门做退）；② squirrel↔bird 错分对 **↓** + squirrel test AP50 **↑**；③ 混淆矩阵对角线占比 **↑** + TIDE Cls **↓**。
> 保留 round2 口径：尺度分层 AP + 逐类 AP + worksite 集。

---

## 7. 量化友好（E）——per-tensor INT8 会放大类混淆

**核心风险**：Vivante VIP 是 **per-tensor 对称 INT8**（比 per-channel 更狠）——离群 channel 会压低其余 channel 有效比特，**分类头上可混淆类的 logits 量化后更易挤到一起 → "置信度都差不多"被量化进一步放大**。应对（全不违反"上游只产 FP32 ONNX"）：

- **保 LeakyReLU**（有界、量化友好）；**别碰 h-swish/SE**（软算子回退 + 量化敏感）。
- **PTQ 校准集必须含难混淆样本**（站立松鼠 / 小鸟）——校准范围覆盖到判别性激活，否则类 margin 塌。用 §3 的难样本构建校准集。
- **§6 三件套 fp32 + int8 各跑一遍**，直接量出量化带来的类判别塌缩（正好落 gate 的 fp32+int8 两档）。
- **FP32 阶段刻意做大类 margin = 给 per-tensor 量化买 headroom**（§3 数据动作的免费协同）。
- **QAT 不走**（与硬规则冲突）——只靠"好校准集 + FP32 大 margin"防塌，诚实取舍。
- 引用：量化白皮书(1806.08342) · EasyQuant(2006.16669) · SmoothQuant(2211.10438) · NVIDIA QAT vs PTQ。

---

## 8. 训练方法（F）——★新+旧怎么训 + 参数怎么设（用户 2026-07-07 + critic 采纳）

- **★新+旧一起训（并集），不是只训新**：round3 = 从 round2 main **热启** + 在 **round2 全量数据 ∪ round3 新增**
  （新源/crop/hard-neg/受控远景）上重训。**只训新会灾难性遗忘**——bird 命门 85.0 是 round2 富 bird 数据撑起的，丢不得。
  （与分类段 [[ADR-0008]]"扩地域 = warm-start 重训并集"同一范式。）
- **热启**：`schedule.load_model = round2 main ckpt`（`results/detect/round2/checkpoints/main/model_best/model_best.ckpt`，**权重热启非 resume**）。架构冻结 → 全加载。
- **★LR 下调（critic Finding 6）**：round2 是 COCO→feeder **大域迁移**，lr 1e-3 合理；round3 是 **feeder→feeder 小域移**
  （只改数据分布）→ lr 1e-3 会过热冲垮已收敛的 bird 特征。**初始 lr 降到 5e-4**（或 1e-4 + 更长 warmup），命门稳后消融再试回升。
- 其余沿用 round2：batch96 · AdamW wd0.05 · Cosine · EMA0.9998 · 416 · fp32 · SwanLab。**epoch**：小域移或许 <24 够，
  但新数据量增 → 首轮仍 24ep + 早停护栏（下）；可作开放消融。
- **类平衡 = 数据层预过采样（verify G1：不进 P0，作独立消融点）**：NanoDet 训练走**子进程/独立 nanodet env**，
  注入不了自定义 sampler；RFS 等效 = build 阶段**物理复制含稀有类图**进训练 COCO JSON（需实现 `_oversample_rare`，
  max_repeat 上限防过拟合 + 防跨 split 泄漏）。**★P0 不含过采样（纯新数据基线），过采样 = 独立点 P6**（否则 P0 分不清"新数据 vs 过采样"，verify G1）。
- **★灾难性遗忘护栏（critic 补）**：① 每 ep 记 bird val AP，**连续 3ep 跌破 83 → 早停/回退**；② 留 round2 域 **canary 集**
  监测特征漂移；③ SwanLab 实时盯 bird 曲线（Monitor 告警，锚 round3 worktree）。

## 9. 消融矩阵（GPU，★拆点归因——critic Finding 5：不 bundle）

每样东西**单独加一个点**、各产 §6 三件套（fp32+int8），才能干净归因（用户 2026-07-07 认可拆点看归因）：

| 点 | = | 加了什么 | 影响层 | 验收看 |
|---|---|---|---|---|
| **P0** | 基线 | 热启 + **新真实数据**（round2 ∪ wan-firdaus/squirrelgarden/bcc），**保 round2 过滤/增强** | 数据 | squirrel/cat recall↑、对 other 分离↑、bird 命门不退 |
| **P1** | P0 + | **放开 OIV7 远景过滤** 0.005→0.002（补小/中远鸟）。**★OIV7 同时改 split_ratios[0.85,0.15,0.0]（verify G4）** → 新小鸟只进 train/val、**test 不变**，保 bird≥85.0 口径 | 数据 | 小鸟召回↑ vs 大目标 AP/混淆是否受损 |
| **P2** | P0 + | **crop-to-feeder-framing**（**所有类含 bird**，合成中大近景） | 数据(合成) | cat/squirrel/bird 中大近景增益 vs 合成分布副作用 |
| **P3** | P0 + | **copy-paste**（各类等量） | 增强 | 小目标召回 + 类判别 |
| **P4** | P0 + | **mosaic(close，最后15ep关)** | 增强 | 小目标 + 跨类共现（margin） |
| **★P5** | P0 + | **旋转 ±5-10°（+轻微 shear/perspective）** | 增强 | 相机微倾鲁棒（喂食器域安全，禁大旋转） |
| **P6** | P0 + | **稀有类数据层过采样**（物理复制 squirrel/cat 图，max_repeat 上限，verify G1）| 数据 | squirrel/cat recall↑ vs 过拟合 |

- **旋转（P5）你一直提，正式编号**：小角度建模相机微倾，低风险；禁大旋转（鸟不会上下颠倒，破坏姿态先验）。
- **★P1 口径修（verify G4）**：放开 OIV7 过滤必配 OIV7 `split_ratios[0.85,0.15,0.0]`，否则新小鸟进 test→bird 基线变、护栏失效。
- **mixup 不列**（GhostNetV3 证实对 ShuffleNetV2 有害）。
- **务实节奏**：先跑 **P0**（纯新数据基线），据实看结果，再挑 P1–P6 里针对性补（不预先烧 7 次卡）。每点 vs round2 基线回归。

## 10. 决策（已定 2026-07-07）
1. **时机 = 现在就做 round3 检测**——分类段另有专门 session 并行推进，互不阻塞。
2. **远景鸟 = 受控补**：`min_box_area_frac` 0.005→~0.002 + 设远景配额上限；**大目标 test AP50 ≥90.1 不许退**，尺度分层 AP + 混淆矩阵双指标盯（A3）。
3. **数据 = 双管齐下**：① 榨干现有源（`roboflow_squirrelgarden` 未用源 + FP 回捞 + copy-paste）；② **同时 acquire 新 Roboflow feeder 场景源**（用户提供 access/key；本会话已起 scout 在 Roboflow Universe 找"观鸟器场景 + 站立松鼠 + 院子猫"的商用许可数据集）。Roboflow key：`app.roboflow.com/settings/api`。

## 11. 无卡 vs 需 GPU 分工
- **无卡现在能做**：数据装配 config + 加 squirrelgarden/hard-neg 源 + **§6 检测混淆矩阵/TIDE/margin 评估代码 + 测试** + 数据门复跑 + copy-paste transform + FP-mining 脚本。
- **需你开 GPU**：热启训练 + §9 消融。

## 12. 模型族：为什么 round3 不换（YOLOX / RTMDet / Damo-YOLO）— 决策记录

用户问：既然 VS861 上 YOLOX/Damo-YOLO/RTMDet 也能部署，NanoDet 又有"置信度都差不多"的结构性根因，值不值得换？**结论：round3 不换**，理由五条：

1. **"置信度都差不多"不是 NanoDet 的病，是整个现代单阶段检测头的通性。** YOLOX（BCE）/ RTMDet（QFL）/ Damo-YOLO（GFL）分类头**全是 per-class sigmoid，都没有 softmax 跨类竞争**——和 NanoDet 一模一样（§2）。**换过去 = 换一个有同样结构性根因的模型**，治不好；解法仍是数据侧喂跨类负例，与架构无关。
2. **round1 已 bake-off 过、且按命门拍板**：NanoDet vs RTMDet-tiny 24ep 短训 **bird AP50 0.774 ≫ 0.483**，NanoDet 命门完胜且 4× 小（RTMDet 整体 mAP 更高但稀释了 bird）。见 `docs/方法论-检测复盘.md`。无新证据推翻。
3. **round2 证明瓶颈是判别数据、不是容量**（bird 85.0/大目标 90.1）。三个问题全是数据问题，更大模型不填数据缺口、也不改 sigmoid 非竞争。
4. **V861 部署现实压倒性偏 NanoDet**：AWNN 实测 NanoDet **266 算子零回退**（LeakyReLU/channel-shuffle/depthwise 全 NPU 原生）、1.35MB、cos 0.9949。YOLOX 带 **Focus/slice**（INT8 不友好，round1 已排除）；RTMDet/Damo 用 **SiLU/大核/rep-block**——"能部署"≠"零软算子回退"≠"塞得进两段级联内存预算"。V861 **内存是墙**，检测+分类级联要求检测器保持轻量给分类器留内存（RTMDet-tiny 是 4× 参数）。
5. **热启动会被丢掉**：round3 从 round2 NanoDet（feeder 收敛的 bird 85.0）热启；换族=从 COCO 重启=丢 feeder 域收敛。

**诚实 nuance**：更大模型可能让 logits 更锐（容量对判别质量非零帮助），但主杠杆是跨类数据、round2 已证数据受限、换族代价（算子风险+内存+热启丢失+重选型）远大于边际收益 → **容量是现在的错杠杆，不是没用的杠杆。**

**挂起的 fallback**：若 data-centric round3 撞到**容量天花板**（判别喂好数据也上不去），再做**模型族 bake-off v2**（RTMDet-tiny 经 MMDetection Apache、非 mmyolo-GPL、QFL、无 Focus 是最干净候选），但须：重验 V861 零回退（SiLU！）+ **更新 [[ADR-0003]]** + 走独立选型实验（非热启轮）。ADR-0003 的 seam 已让"加 RtmdetBackend"仅 ~60 行，技术随时可试，但决策是 ADR 级。

### 12.1 ★M 线：模型 bake-off v2（用户 2026-07-07 启动，并行 P0–P6，不替换 NanoDet）

用户主动启动"试别的模型"——并行验证"**稍大是否更强**"（研究背书，license 逐个抓官方仓库 LICENSE 核实）：
- **✅ Apache 可用**：PicoDet/PP-YOLOE（PaddleDetection）· DAMO-YOLO · **RTMDet（仅 mmdetection，mmyolo=GPL 禁）** · EfficientDet · NanoDet。
- **❌ 排除（点名）**：YOLOv6 / Gold-YOLO / mmyolo（GPL）· YOLO-NAS（**权重禁商用**）· YOLO-MS（CC-BY-NC）· Ultralytics YOLOv5/8/10/11（AGPL）· YOLOX（**Focus stem** 出局）· RT-DETR/D-FINE（transformer 违算子红线）。

**★前提：只从 V861 支持列表（PDF）里选**——表上=全志验证过可部署（有实测延迟/内存）；表外=部署风险，不选。

**从表上检测模型筛**：ppyoloe_s(7.9M/640²/18MB/Apache) · ppyoloe_m(23M,太大) · ssd300(34M/48MB/BSD,老VGG重) · FasterRCNN(27M/33MB/BSD,两阶段224²低分辨率) · tiny-yolov3(8.9M,2018老架构) · retinanet(96MB,爆表)。

**决策 = `ppyoloe_s` 主实验**（7.9M / 640² / **18MB 内存** / Apache / **V861 已验证可部署**）：
- 表上**唯一现代 anchor-free**、许可 Apache、带 **COCO/Objects365 预训练权重**、参数稍大（NanoDet 1.2M 的 ~6.6×）—— 正合"表上+合适场景+有权重+稍大"。
- **实验**：ppyoloe_s（PaddleDetection 微调 round3 数据）vs NanoDet P0，**同固定 test 比** → 证伪/坐实"容量不是瓶颈、数据才是"（怎样都有信息）。
- ⚠️ 算子：SiLU+ESE 上 AWNN 前逐个验回退；640² 比 416² 更吃内存（18MB 仍在预算内）。
- **待办**：无卡=卡上装 PaddleDetection + round3 COCO→PP-YOLOE 格式；训练需 GPU。
- **表外备注（不选）**：PicoDet-M(3.46M/416²,同 Paddle 框架/同 AWNN 路,边侧专用)理论能部署但**未在表上验证**，不符前提。
- 引用：V861 AWNN 支持列表 PDF · PaddleDetection Apache LICENSE · PP-YOLOE 论文(2203.16250)。

## 13. critic /verify 修订记录（2026-07-07 · VERDICT: REVISE → 已采纳）

critic 对抗审计（读了源码）的硬纠错，已改上文 + 下列补丁：
- **F1 混淆矩阵盲区**：已用数据复核（§1.1/基线）——squirrel→bird **未被低估**（共激活 24=矩阵 24），claim 已弱化；margin 现象仍须 GPU。
- **F2 RFS**：Lightning sampler 不可用（NanoDet 子进程/独立 env）→ 改**数据层预过采样**（§8/A5）。✅
- **F3 squirrelgarden 非"零改代码"**：需新写 ~20 行 adapter（子类化 `RoboflowFeederAdapter`）+ 核 workspace/project/version/label_map + 许可 → 列入 M2 无卡任务。
- **F4 copy-paste/mosaic 工作量**：走**离线数据增强**（预生成增强图 + COCO JSON，不改 NanoDet pipeline），~2-3 天；首轮矩形 crop（无 mask）。mosaic(close) 同须自实现、单列。
- **F5 消融 bundle**：已拆点 P0–P4（§9）。✅  **F6 LR**：已降 5e-4（§8）。✅

补充护栏 / 缺口（critic minor + missing，已纳入）：
- **次要类地板护栏**：squirrel test AP50 **≥60**（现 51.4）、cat **≥60**（现 54.5）——不止"↑"。**★cat≥60 仅 P2+ 适用（含 crop/stray-cats 后）；P0 cat +0 框 → P0 只记录 cat 值、不判 pass/fail（verify G5）。**
- **误报率护栏**：worksite 空帧 FP@0.4 现 ~7.5% → hard-neg 应降；设"**不许升**"护栏（远景鸟/copy-paste 可能推高）。
- **★other_animal 判别（缺口）**：不只"加 squirrel/cat 把它们拉出 other 盆地"——还要**让 other_animal 更判别**（训练 other 里剔像猫/松鼠的、补明确非猫非松鼠动物），并验证"更多 squirrel 数据能否真拉出 other"这个假设。
- **★部署域 eval 规格化（缺口）**：worksite 现集**结构上测不了"喂食器站立松鼠"**（其 squirrel 来自后院/相机陷阱）→ round3 须**采+标真实喂食器野图**：先 ~200–500 帧、LabelStudio、覆盖站立/爬台/多光照/多鸟共现。
- **gate 阈值**：`data/gate.py` `ROUND2_MIN_BOXES{squirrel:2000}` → round3 目标 6–8k **框**（非图；"6–8k"统一指框），gate 同步调。
- **crop 最小分辨率门**：代码已定 `min_box_px=48`（`feeder_crop.py`）。
- **margin 指标非"无卡"**：直方图代码可无卡写，但**产数据须改 NanoDet 推理 dump 预 NMS 全类分**（GPU + nanodet env）。
- **负样本无未标注鸟（缺口）**："只有松鼠"的 Roboflow 图常含背景鸟 → 须**过 bird 检测器/人核**再当 bird 负样本，否则漏标假阴。
- **数据中心天花板 + fallback（skeptic）**：sigmoid 无跨类竞争是结构限，数据有天花板；若 squirrel→bird margin 加 3× 数据仍 <+0.1 平台 → 触发 §12 bake-off v2。每点盯 margin 增量判是否触顶。
- **开放问题**：24ep 对小域移或偏多（可消融）；copy-paste 鸟+松鼠同框在 sigmoid 下是**共现监督(好)还是鼓励共激活(坏)**——P3 消融看 margin 走向；**TIDE 须先验证吃得下 NanoDet 输出格式**再当"低成本"。

---
### 主要引用
GFL/QFL/DFL(2006.04388) · NanoDet(github RangiLyu) · OHEM(1604.03540) · Hard-FP(1810.04002) · 背景负样本(2604.02282) · RFS(2305.08069) · 类不平衡(2403.07113) · Seesaw(2008.10032)/EQL(2003.05176) · Copy-Paste(2012.07177) · Context Copy-Paste(2407.08151) · 小目标综述(MDPI 15/22/11882) · SAHI(2202.06934) · YOLOX(2107.08430) · GhostNetV3(2404.11202) · TIDE(2008.08115) · D-ECE(2004.13546) · 量化白皮书(1806.08342) · EasyQuant(2006.16669) · SmoothQuant(2211.10438)。
