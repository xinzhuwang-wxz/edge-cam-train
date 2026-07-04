# 检测正式训练轮 · 数据 + 训练安排（单轮最佳实践）

> **定位**：round1 的零样本谱 + bake-off 是**选型预实验**（已定 骨干 NanoDet-Plus-m 416 · MD 当教师 ·
> 修了 decode/预处理的缝），**不是最终训练**。本轮 = **一次性在全量数据上的最佳实践训练**（不再"接力"）。
> 相关：[[ADR-0004]]（5 类）· [[ADR-0005]]（teacher+区域）· [[ADR-0006]]（acquire seam + iNat/Roboflow adapter）·
> [[ADR-0007]]（ONNX logits，#59 落地）· `results/detect/round1/`（选型结论）· `docs/classify/01-数据集.md`。

## 0. 单轮流水线（best practice，一条龙）

```
① 数据装配 → ② 数据门 → ③ 训练(主模型 + scaling 扫) → ④ 评估 → ⑤ 收口
```

| 步 | 做什么 | 用什么件 |
|---|---|---|
| ① **装配** | 全源合一（ENA24+CCT 复用 + OIV7 + iNat-MD + Roboflow）→ 一 JSONL、确定性 split、**test 固定** | `build`（§2/§4） |
| ② **数据门** | 量+质量过关（bird≥8k/person≥4k/均衡/框合理/许可/署名），**不 pass 不训** | `data/gate.py`（§6） |
| ③ **训练** | 主模型 NanoDet-416（COCO warm-start · **域增强** · 单一真值源预处理 · logits 导出 · SwanLab）+ **scaling 扫**（数据量×参数 两曲线，100%-1.0x=主模型） | `patch_nanodet_config`·`subsample_train`·`swanlab_nanodet`·`DetectorPreprocess`·`onnx_postproc`（§7） |
| ④ **评估** | 固定 test：命门 bird AP50/召回/查准 + 逐类 + mAP。对参照 round1 NanoDet-FT 0.774 | 原生 COCOeval + `zeroshot_eval` |
| ⑤ **收口** | 量化包络 + gate → registry（发布链）+ 报告 | `eval/`·`registry/` |

> **为何 scaling 内嵌一个 round**：主模型就是 scaling 的 100%-1.0x 点，不额外花训练——一箭双雕。
> **全程 SwanLab**：`project=edge-cam` / `workspace=maxen`，train loss + val bird AP50 同 run（`SWANLAB_API_KEY` env）。

## 1. 目标

1. **补真实喂食器域**（当前最大短板：round1 bird 框全来自网图/相机陷阱，无"鸟停喂食器"场景）。
2. **补 bird 覆盖/多样性**（命门；iNat Aves 海量，MD 打框）。
3. **补 person**（round1 训练 person 框 = 0；产品需"人补粮→抑制告警"）。
4. **分布拉合理**（round1 失衡 5:1、squirrel/cat 偏少、person=0）→ 命门 bird 重、其余均衡、无类饿死。

## 2. 数据源 → 类映射（adapter 全就绪，ADR-0006）

| 源 | adapter | 供哪些类 | 许可 | 状态 |
|---|---|---|---|---|
| ENA24 | `ena24` | bird/squirrel/cat/other_animal | CDLA 可商用 | round1 已用 ✅ |
| Caltech CT(ECCV18) | `caltech_ct` | bird/squirrel/other_animal + empty 负样本 | CDLA 可商用 | round1 已用 ✅（[[CCT 坐标 bug]] 已修） |
| **OIV7** | `oiv7_direct` | **person**(cap 8000) + bird + other_animal | CC-BY-4.0 商用 | adapter 就绪，**person 选1 现成源** |
| **iNat Open Data** | `inat_md` | **bird**（MD 伪标注） | 收紧 **CC0/CC-BY** | adapter 就绪（MD 打框 box GPU） |
| **Roboflow feeder** | `roboflow_feeder` | feeder 域 bird + **person(选2)**/squirrel/cat | 逐个核 CC-BY/Public | adapter 就绪，⚠️ **卡 Roboflow 访问** |
| COCO2017 | `coco2017` | eval_only（防许可传染） | — | 评估用 |

## 3. 分布目标（"合理" = 命门重、其余均衡、无饿死）

| 类 | round1 框 | 问题 | **round2 目标(框)** | 来源 |
|---|---|---|---|---|
| **bird** | 1620 | 命门偏少 | **~8–10k** | round1 + OIV7 + iNat(MD) + Roboflow |
| **person** | **0** | 缺一类 | **~4–6k** | OIV7(cap 8000) + Roboflow feeder 人 |
| other_animal | 8428 | 过载 5:1 | **压到 ~4–5k**（`max_per_class` 调低） | ENA24/CCT（够，压即可） |
| squirrel | 1062 | 稀缺 | 尽量补 **~2–3k** | round1 + iNat/Roboflow（数据可得上限，接受偏低） |
| cat | 873 | 偏少 | **~2–3k** | round1 + OIV7/iNat |

总量目标 ~20–25k 框；**bird 最多、person/other_animal 次之、squirrel/cat 保底不饿死**。
负样本（CCT empty 空帧）按 `negative_quota` 保留域真实背景。**build 后必查每类框数**（`audit_unmapped` +
分布打印），不达标回调 caps / 补源。

## 4. 流程（你的四步 → 可执行序列）

1. **扩展数据**（box）：`acquire` 拉 Roboflow feeder（需 `ROBOFLOW_API_KEY`）+ iNat Open Data S3（CC0/CC-BY、
   research-grade、有 geo、per-taxon 配额）；OIV7 person/bird 直下。
2. **教师打框**（box **GPU**，隔离 `pytorch-wildlife` env）：MD 跑 iNat 图 → animal 框（iNat 已筛 Aves →
   animal≈bird）→ COCO，框 `label_provenance=md_pseudo`。
3. **人 LS 复核**（决策：**只审低置信 + 稀有类**）：MD 高置信框直接信（`md_pseudo`）；**低置信框 + squirrel
   等稀有类**进 Label Studio 人审 → `md_human_verified`（信任分层，透明可审）。人力花在刀刃上。
4. **build + 查分布 + 训练**：`build` 合并 6 源 → JSONL；**打印每类框数核对 §3 目标**；不达标调 caps/补源；
   达标 → NanoDet-Plus-m 416 round2 微调（baseline = round1 **bird AP50 0.774**）。

## 5. 已锁定决策

- **person = 双源**：OIV7（CC-BY 商用，选1 现成）+ Roboflow feeder 域的人（选2，待访问）。
- **LS 复核 = 只审低置信 + 稀有类**（非全审；高置信 MD 直接用，先性价比）。
- **许可红线（§4）**：iNat 收紧 CC0/CC-BY（去一切 NC）；Roboflow 逐个核；teacher(MD MIT) 干净。
- **导出 = ONNX 出 logits**（[[ADR-0007]] #59 已落地，round2 训练产的 ONNX 自动剥 Sigmoid，不再喷框）。

## 6. 训练前**数据门**（硬 gate，不达标不训）

用户要求"训练前数据质量和数据量都要过关"。`build` 后、训练前跑 `edge_cam.data.gate.gate(manifest)`
（纯函数、6 测试，`src/edge_cam/data/gate.py`），**不 pass 就拦训练**。查：

| 项 | 判据 | 抓什么 |
|---|---|---|
| **数据量** | 每类框数 ≥ §3 目标（bird 8k / person 4k / squirrel·cat 2k） | 命门够不够 |
| **均衡** | max/min 类框数比 ≤ 6 | round1 曾 5:1 失衡 |
| **框坐标合理** | 无超界 / 零负面积框 | [[CCT 坐标 bug]] 式错位（框比图大 2 倍） |
| **许可(§4)** | 逐图 license 全商用白名单 | NC/unknown 红线 |
| **CC-BY 署名** | CC-BY 图有 author/original_url | §4 逐图署名兑现 |
| **伪标注信任** | md_pseudo 未审占比（提示） | MD 伪标注占比 vs LS 复核策略 |

## 7. Scaling 研究（训练时看**参数 + 数据量**的 scaling）

用户要求"训练时看到 scaling 状况（参数/数据量）"。两条 1D 扫（Hydra multirun，`eval/ablation` harness）：

- **数据量 scaling**：NanoDet-416 固定，训练集取 **{12.5, 25, 50, 100}%** → 同一 held-out test 评 →
  **bird AP50 vs 数据量曲线**。答："数据够没够/边际收益还在不在"（=你的"数据量过关"）。
  **机制**（`data/scaling.py` `subsample_train`，纯函数+4 测试）：**图存一份**（raw_root），JSONL manifest 只存
  引用 → 子集 = records 子集、**零图复制**；**只抽 train、val/test 固定**（同一 held-out 才可比）；按 `path`
  hash 确定性取前 frac、**嵌套**（20%⊂50%⊂100%，"加数据"是真加）。每 run 一个 frac，不改原始文件。
- **参数 scaling**：数据固定 100%，NanoDet backbone **{0.5x, 1.0x, 1.5x}** ShuffleNet → 评 →
  **bird AP50 vs 参数曲线**。答："1.0x 够不够 / 1.5x 值不值"（1T/128MB 尤其看这条）。

**训练时可见**：① 每个 run 逐 epoch 记 bird AP50 val 曲线（NanoDet/mmdet 原生）；② 一个 scaling 汇总器
随 run 完成**实时更新两张曲线 + 表**（数据% / 参数 → bird AP50）。产出 `results/detect/round2/scaling_报告`。
**先过 §6 数据门再开扫**。

## 8. 数据管线打磨（round2 = ADR-0006 管线首次端到端真跑）

round2 是 `acquire → MD 伪标注 → LS → build → gate` 全链路**第一次在新源真跑**——很多毛边只有真跑才暴露，
正是打磨管线的时机。**纪律：每趟真跑遇到的毛边即时修 + 补测试/文档，让管线"跑一次=稳一分"。** 已知毛边：

- **Roboflow adapter `label_map` 是占位**（其 docstring 注明）→ 上 box 拿真实 export 后跑 `audit_unmapped()`
  据实校正类目字符串 + 逐个核许可（§4）。
- **MD 伪标注步骤未进管线**（独立 GPU 阶段）→ 打磨成可复现脚本（iNat 图 → MD → `md_pseudo` COCO + 收据）。
- **LS 复核回流**：`md_pseudo → md_human_verified` 的 Label Studio ↔ manifest 导出/导入闭环打通。
- **数据门进 build 尾**：`build` 后自动跑 `gate()` + 报告（gate 已建 `data/gate.py`，wire 进 build 流）。
- **逐图 provenance/署名**：新源 author/original_url 真填（CC-BY 兑现，gate 已查缺失）。
- **可复现**：`acquire` 收据（sha256/version）+ 确定性 split，换机零考古（ADR-0006 D1）。

## 9. 前置/阻塞

- **你（用户）**：申请 **Roboflow 访问 + 拿 `ROBOFLOW_API_KEY`**（feeder 域 + person 选2 卡这，最大短板源）。
- **box GPU**：iNat MD 伪标注 + scaling 扫（数据×参数 多 run）（GPU 已停，需重开）。
- **LS 实例**：Label Studio 搭起来（低置信+稀有类人审）。
- 相关 issue：#60（round2 微调，本计划的执行）。加源 = box 上 `acquire`+`build`，零改代码（ADR-0006）。
