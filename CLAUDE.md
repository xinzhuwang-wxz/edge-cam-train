# CLAUDE.md

> 给所有 AI 会话（Claude Code 等）的常驻指引。**接手任何会话先读这份**，再读 §文档地图 指向的两份设计文档。
> 关联远端：<https://github.com/xinzhuwang-wxz/edge-cam-train>

---

## 1. 这个项目是什么

**edge-cam-train** = 边侧 AI 相机的「**数据准备 → 训练/微调 → 消融对比 → 量化 → 部署/OTA**」全链路框架。

- **目标硬件**：Allwinner **V85x**（VeriSilicon Vivante VIP NPU，**0.5–1 TOPS，INT8-only，无 FP**；DDR ≤256MB，按 **0.5T / 128MB 下限**设计）。
- **任务**：两段式级联 —— 通用**粗检测器（NanoDet-Plus，固定）** → 取 bird crop → **细分类器（timm EfficientNet-Lite0，可 OTA 换/扩）**；非鸟直接出大类；bird 命中后给「种 + 置信度 + top-5」，**高置信用片上标签，低置信/不在地域清单/稀有种 → 层级回退(属/科/bird) + 留云 API 锚点（不实现云端）**。
- **定位**：产品的**一个 feature**，资源占用须克制；商用（海外发行）、PyTorch 系、成熟稳定可扩展。
- **当前阶段**：**W0 scaffold** —— 目录骨架就绪，模型/部署叶子（标 ★）待实现。`pyproject.toml` 尚未创建，本地尚未 `git init`。

### 一条绕不开的平台现实
**没有任何框架能一路覆盖到 Vivante NPU 部署端**。所以：模型/训练层**重度复用 OSS**，**上游只产 FP32 ONNX**，量化落地那段（ONNX → ACUITY/pegasus PTQ → `.nb` → VIPLite/awnn）**一定是薄自研胶水**。这不是设计缺陷，是平台事实。

---

## 2. 文档地图（先读这两份）

| 文档 | 讲什么 | 何时读 |
|---|---|---|
| **`docs/plan-v2.md`** | **what/why**：级联架构、分级置信门控、模型/数据/评估口径、可选音频(§9)、**附录 B 实验清单**、**附录 C 承重规格** | 做方案决策、对齐领域模型 |
| **`docs/engineering.md`** | **how**：PyTorch 分层选型、数据准备(§5.5)、**仓库结构(§6)**、W1 落地步骤(§7)、风险(§8)、ADR(§9) | 动工程、写代码、选库 |
| `docs/decisions/` | ADR（架构决策记录），目前为空，后续不可逆决策落这里 | 做/查不可逆决策 |
| `README.md` | 面向外部的项目简介 | — |

> 文档里写过的事实（架构、选型理由、许可红线、历史决策）**不要在 CLAUDE.md 里重复**；这份只放「怎么协作」+ 指针。

---

## 3. 仓库结构速览

完整结构见 `docs/engineering.md §6`。要点：包根是 **`src/edge_cam/`**（不是 `lujiazui`、不是别的）。标 **★** = 需新写的业务叶子；其余为按成熟模式自建的薄编排层。

```
src/edge_cam/
├── core/                 配置(Hydra)/logging/seed/paths/coords
├── contracts/schemas/    pydantic: model_card / eval_report（含边侧 Literal + 版本）
├── registry/             自建薄层 或 MLflow：store/manifest/publish/verify
├── deploy/
│   ├── manifest_api/     FastAPI OTA routes
│   └── packager/acuity_packager.py   ★ subprocess 调 pegasus PTQ→.nb
├── data/                 ★ 数据准备 pipeline（离线 dev/GPU 机；FiftyOne/MegaDetector/crop/taxonomy/split/calib）
├── train/
│   ├── detect/           ★ NanoDet-Plus 包一层（导 FP32 ONNX）
│   └── classify/         ★ timm + Lightning + Hydra
├── eval/
│   ├── ablation/         ★ Hydra multirun harness + metrics
│   ├── gates/            ★ fp32+int8 两档 4 维阈值门
│   └── quant_estimate.py ★ ORT-QDQ 本地掉点预估（消融列，不进部署）
└── edge/viplite_runner/  ★ ctypes 调 VIPLite（借 frigate 蓝本；输出 CHW reshape）
configs/                  Hydra: ablation/ · eval/gates/ · channels/
data/                     DVC 跟踪：训练集 / 校准集 / .nb 产物
```

---

## 4. 不可触碰的硬规则（违反即 PR 必拒）

这些是已落定的 ADR/许可红线，**任何会话默认遵守，改动前必须先开 ADR 讨论**：

1. **上游只产 FP32 ONNX**，绝不上游 INT8 —— Vivante 私有量化，预量化图无用；INT8 一律交 ACUITY/pegasus PTQ。上游 INT8（ORT/PT2E）**仅作消融矩阵「预期端侧精度」列，产物不进部署**。
2. **检测后处理（NMS/decode/grid/sigmoid/anchor）留 A7 CPU**，不进 NPU 图；导 ONNX 时切「裸 backbone+head」。
3. **复用 = 依赖上游 + 扩展点（config/callback/plugin）**，❌ 不 fork 改框架源码（NanoDet 半停滞需 fork 锁版属唯一例外，且**只锁不大改**）。
4. **许可红线（海外发行从严）**：全栈 Apache/MIT/BSD；
   - ❌ 避 **AGPL**（Ultralytics YOLOv5/v8/11）、**GPL**（MMYOLO）、**CC-BY-NC**（iNaturalist 数据、**BirdNET 权重**及衍生）。
   - 数据只用 **CC0/CC-BY + 自采**，维护**逐图/逐录音署名清册**随产物披露。
   - 换数据/换 head **不能洗白上游 license**；teacher 含 NC 数据 → student 同样不可商用。
   - MegaDetector 用 `MDV6-mit-yolov9-c`(MIT) 或 `MDV6-apa-rtdetr`(Apache)，**避开 yolov10 变体(AGPL)**。
5. **不掺入无 LICENSE 的内部代码**（如 `pet-train-unified`）—— 只借工程模式（思想），不 copy/fork/依赖其代码，保新仓 provenance 干净。
6. **避坑算子**：不用带 **Focus/slice** 的检测器（YOLOv5、YOLOX-Nano stem）；SE/h-swish 须实测是否回退软算子；**不用 transformer 类**；动态 shape 先消除。

---

## 5. 固定工作模式（所有 AI 会话默认遵循，无需用户每次声明）

按阶段推进，**每阶段产物落盘**。看任务性质匹配阶段，不必每次从 0 走到 4。

| 阶段 | 目标 | 技能/方法 | 触发 |
|---|---|---|---|
| **0 护栏** | 改坏了立刻知道（省后面无数事故） | issue tracker = **GitHub Issues**（已配 `ready-for-agent`/`hitl` 标签）；**pre-commit 钩子**（ruff + ruff-format + pytest）；**git 危险命令拦截**（防 force-push/误删分支，Claude Code hooks） | 新环境/新会话接手时**核对一遍护栏在位** |
| **1 审计** | 看清架构与问题 | `zoom-out`：给出项目全貌 + 工程债清单 | 接手不熟悉的部分、阶段性回顾 |
| **2 定靶** | 沉淀领域模型 + 决策 | `grill-with-docs`：对着领域模型把决策写进 `CONTEXT.md` 和 **ADR**（`docs/decisions/`） | 出现模糊术语、要做不可逆决策 |
| **3 规划** | 找重构点、拆任务 | `improve-codebase-architecture` → `to-prd` → `to-issues` → `triage` | 启动一轮改造 |
| **4 执行** | 一片一片安全推进 | `tdd` + `diagnose` + `review`；**小步提交** | 拿到一个明确 issue/任务 |

### 执行纪律（硬约束）
- **每阶段产物落盘**：HTML 报告 / `CONTEXT.md` / ADR / PRD / 测试 —— 不落盘等于没做。
- **提交前必跑**：`pytest` + `ruff check src tests`（外加 `ruff format`）。pre-commit 钩子兜底，但本地先跑。
- **安全面的改动必须先写测试**：状态机（如 tracker `max_age/min_hits`、OTA channel 状态）、硬规则（§4 许可门 / gate 阈值 / FP32-only / 后处理留 CPU）、权限隔离 —— **先红后绿**。
- **不可逆决策先开 ADR**：写进 `docs/decisions/`，再动代码。
- **每阶段产物可追溯**：消融结果进实验总表，模型/数据进 DVC + registry，提交信息关联 issue。

> ✅ **护栏在位（核对清单）**：`git`（remote=origin 指向上述 GitHub）、`pyproject.toml`（ruff+pytest）、`.pre-commit-config.yaml`（`pre-commit install` 已装，提交前跑 ruff+ruff-format+pytest）、`.claude/`（PreToolUse hook 拦截 `git push`/`reset --hard`/`clean -f`/`branch -D` 等）均已就绪。新会话核对：`pre-commit run --all-files` 三项全过、`echo '{"tool_input":{"command":"git push"}}' | .claude/hooks/block-dangerous-git.sh` 应 exit 2。GitHub 侧 Issues/标签（`ready-for-agent`/`hitl`）已配。

---

## 6. 命令参考

> W0 阶段 `pyproject.toml` 未建，以下为**目标命令**，落地前以实际工具链为准（pegasus 子命令随 ACUITY 版本变动，见 `plan-v2.md` 附录 B.8）。

```bash
# 环境（待建 pyproject 后）
uv sync                          # 或 pip install -e .

# 提交前必跑
pytest
ruff check src tests
ruff format src tests

# W1 spike（最高优先 / 最大不确定性）：一个检测器端到端跑通一帧
#   NanoDet-Plus → export_onnx → onnxsim 静态化 → pegasus PTQ → .nb → VIPLite/awnn 上板
#   摸清 ACUITY 算子兼容清单 + INT8 掉点；验收：一帧端到端有合理输出

# 消融（训练侧立起后）
#   Hydra multirun: backbone × 输入分辨率 × 增强 × quant 档；aim 追踪 + DVC 版本化
```

落地节奏见 `engineering.md §7`（W1 spike → 编排骨架 → 训练侧 → 消融 harness → gate+publish）。

---

## 7. 协作约定

- **语言**：文档与注释跟随仓库现状用**中文**；代码标识符/命名用英文，匹配周边代码风格。
- **小步提交**，提交信息关联 issue 号；只在用户要求时 commit/push；当前在默认分支上则先开分支。
- **W1 是 spike**：最大不确定性在 ACUITY 工具链（文档零散、版本易错配、Ubuntu-only）。先打通最小通路，再上业务模型 —— 别先堆训练代码。
- 跨设备 latency/精度回灌（板上 `.nb` → EvalReport）**无现成支撑**，是最大工程盲点（`engineering.md §8.1`）—— 涉及时显式新增回传层。
- 不确定的领域决策 → 走阶段 2（`grill-with-docs` + ADR），不要拍脑袋写进代码。
