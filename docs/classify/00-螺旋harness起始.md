# 螺旋 harness · 分类段起始

> 下一 session（分类）的起点。检测 round1/round2 已收官，方法论见 `docs/方法论-检测复盘.md`。
> 本文定义驱动分类开发的「螺旋 harness」，并给出第一步审计清单。

---

## 一、螺旋 harness 是什么（loop-harness 的进阶）

**loop-harness（已有）**：goal 驱动 + 约束 + 迭代（改→验→留/弃）。问题：**计划是死的**，按 roadmap 一路执行到底。

**螺旋 harness（要做）**：在此之上多两层，让**计划活起来、随现实螺旋上升**：

1. **可变 roadmap** —— 路线图显式存在，但随调研/反馈**随时可改**（活计划，非陈旧计划盲执行）。
2. **贯穿全程的内容审查闸门（核心新意）** —— 这是一个**普适、贯穿全程的反射**，不是"审这几类产物"的清单。**任何时候产生或接触任何信息/活动**，都自动审查**实际内容**：不是"跑通了吗"，而是"**这东西真的服务目标吗 / 和现实吻合吗 / 它意味着下一步要不要变**"。
   - 覆盖（举例，非穷举）：**调研/investigation 的发现**、**采来的**数据、**生成的**（伪标/裁剪/增强）、**派生的**（split/校准集/taxonomy）、**训练出的**（模型/权重）、**评出的**（指标/消融/报告）、**导出的**（ONNX/交接包）、乃至**一个决策/假设**——**无一例外**。
   - 一句话：**在每一个"知道了新东西"的时刻，都停下来问它对目标和路线的含义。**
3. **审查回改路线** —— 审查发现偏离（如"采的数据和观鸟器场景没关系"、"伪标质量差"、"某类样本其实是别的东西"），**自动触发 roadmap 变更**（返工/改下一步/加校验），全程**在 harness 管理内、不靠人驱动**。

→ **螺旋上升**：每轮按反馈精修路线，系统朝目标爬，而非盲执行。

**与普通 plan 的区别**：普通 plan 执行完就完了；螺旋 harness 的**每一步产物都反哺路线**。

**痛点来源（检测教训）**：数据按计划采了，但**缺一个"自动看一眼产物是否匹配目标"的闸门**——审查靠人晚驱动，于是"数据≠观鸟器场景"这事儿发现晚、大返工。这个反馈机制要做成一等公民，且**贯穿每一样产物**。

---

## 二、外部实践调研（2026）—— 螺旋 harness 不是无先例，是有据的综合

**① 名字有 30 年血统：Boehm 螺旋模型。** 软件工程的**螺旋模型**（Boehm，风险驱动迭代）每圈 4 活动 = 定目标 → 析风险 → 工程 → **评估产出 + 对照成功指标/反馈 再进下一圈**。螺旋 harness ≈ **把螺旋模型套到自主 agent 开发**——"评估产出对照现实"正是每圈的评估闸，"螺旋上升"正是它的原意。这validates了直觉。

**② 领域有专名：harness engineering。** Anthropic/OpenAI 称 **"harness engineering"**，Geoffrey Huntley 称 **"back pressure engineering"**：框架越好、agent 越可靠。参考枢纽 `github.com/ai-boost/awesome-harness-engineering`。

**③ 三个新增各有成熟对应（可直接借鉴/下载）：**

| 螺旋 harness 组件 | 外部对应（可抄）|
|---|---|
| **可变 roadmap** | **Task-Decoupled Planning**（Supervisor 建依赖图 + **Self-Revision 执行后更新图**，局部重规划不级联）；**AdaPlanner**（闭环自适应改计划，环境反馈双向）；**LATS**（MCTS + 失败回溯）；**Plan-and-Execute**（需要时才重规划）|
| **贯穿现实审查闸门** | **Reflexion**（自省存记忆→精修计划）；**Self-Refine**（内部自检反馈）；**PARC**（自省式长程编码 agent）；**SkillOpt**（validation-gated 更新）+ 螺旋模型每圈评估 |
| **长程 + 对项目负责** | **Anthropic《Effective Harnesses for Long-Running Agents》**（feature list + git commit + test gate 当跨 session 状态）；**goal persistence**（目标存活、计划可弃→触发重规划）；**ralph loop**（fresh context + 文件系统当记忆，spec↔code 比对出任务，每圈一任务；避"Dumb Zone" 100–150k token 后掉质）；**statewright**（状态机护栏，按 phase 限工具）；Meta **REA**（hibernate-wake 断点续 6h 任务）|

**④ 真正的空白（值得自己抽成技能）：** 单一现成品都不完全是你要的，但**每块都有强先例**。你的独特打包 = **审查是"贯穿全程/一切信息（含调研发现、决策）"的一等反射 + 直接改活 roadmap + goal 持久 + 螺旋上升**——这个特定组合没现成 skill，值得抽（结合 autoresearch）。

**⑤ 若抽成"结合 autoresearch 的新技能"，建议架构：**
- **基座循环**：`autoresearch`（goal-metric-verify）或 `ralph`（PRD 持久 + fresh context）。
- **活 roadmap**：一个 harness 每圈读+改的路线文件（学 Anthropic feature-list / TDP 依赖图）。
- **现实审查反射闸门（核心新增）**：任何产物/调研/决策后，一个 audit subagent 问"服务目标吗 / 合现实吗 / 下一步要不要变"→ 写发现 + roadmap 增量（Reflexion 式，落盘）。
- **goal 持久**：goal 存活不变，roadmap 可弃可改。
- **循环骨架 = Boehm 螺旋圈**：{定目标 → 做 → **审现实+目标** → 改活 roadmap → 下一圈}，token 到阈值就 fresh context 续（ralph 式）。
- **待定义**：审查闸门**审什么**（每类信息的"服务目标"判据）、**怎么触发**改路线（信号/阈值）、**改什么**（roadmap 哪部分可变、哪部分冻结）。

**关键参考链接**：awesome-harness-engineering（GitHub 枢纽）· ralph（snarktank/ralph）· Anthropic long-running-agent harness · AdaPlanner / Reflexion / PARC（论文）· RoadmapBench（长程 agentic 开发评测）。

---

## 三、第一步 · 审计已有分类工作（先审计，别急着动手）

读并提炼：
- `docs/classify/`（README + 01 数据集 + 02 训练评估 + 03 实操日志）
- ADR-0002（eBird taxonomy seam）、ADR-0005（分类数据许可 + teacher + 区域）
- `results/classify/`（cascade_v2、envelope_v1_vs_v2、crop 消融、区域评估、figures/assets）
- `src/edge_cam/train/classify/`（train/module/data/augment/export）+ `src/edge_cam/data/`（crop/taxonomy/split/calib/pseudolabel）

**产出**：之前分类做到哪、可借鉴的 know-how、踩过的坑、已发布 registry 的模型（fp32 / int8_sim / field 各多少）、还差什么。

---

## 四、方法论底座

沿用 `docs/方法论-检测复盘.md` 的 10 条（命门指标 / 可行性包络 / 过程vs定案分离 / scaling量化 / 证据驱动 / 诚实caveat / 两目标两尺子 / 复用OSS / 许可红线 / 落盘可追溯）。**螺旋 harness 是给这套方法论加一个"贯穿全程的内容审查-回改路线"的执行引擎。** 可考虑将其提炼为方法论第 11 条。

---

## 五、落地节奏（重要：别"先造完再用"，共同螺旋上升）

先造完美技能再用 = 纸上完美计划，违背螺旋精神。正确节奏是**用螺旋 harness 的方法造螺旋 harness**，分类既是被开发的项目、也是 harness 的第一个试验+反馈场：
1. **搭 v0 骨架**（`write-a-skill` 建结构）——最小可跑：基座循环（autoresearch/ralph）+ 一个活 roadmap 文件 + 一个审查闸门。别一次大而全。
2. **在分类里 dogfood**——分类真实反馈（数据审查、消融、区域评估…）既推进分类、也暴露 harness 问题；每圈审"这 harness 真帮到了吗/审查闸门审得对吗"→ 反过来改 harness。
3. **固化+精修**（`skillify` 固化跑通的工作流 / `learner` 从会话提炼）——每圈把 harness 打磨更好，结合进 autoresearch。

## 六、下一 session 起始 prompt（可直接粘贴）

> 背景：检测 round1/round2 已收官，方法论在 `docs/方法论-检测复盘.md`。现在开始【分类】段，并用一个自研的「螺旋 harness」驱动开发（概念+外部实践调研+架构建议在 `docs/classify/00-螺旋harness起始.md`）。
>
> 【第一步·审计已有分类】读 docs/classify/、ADR-0002/0005、results/classify/、src 分类代码，提炼：做到哪、可借鉴 know-how、坑、已发布 registry 模型（fp32/int8_sim/field）、还差什么。
>
> 【第二步·搭螺旋 harness v0 并在分类里 dogfood】按 00-螺旋harness起始.md §二⑤架构 + §五节奏：**先搭最小 v0**（基座 autoresearch/ralph + 活 roadmap 文件 + 一个"贯穿全程现实审查→回改路线"的闸门），**别先造完再用**；拿 v0 跑分类，用真实反馈同时推进分类+精修 harness（三步共演化）。核心=审查判据从「did it work」升级为「does it serve the goal」，对项目负责、螺旋上升。先和我把"审查闸门审什么/怎么触发改路线"定义清楚再开跑。
