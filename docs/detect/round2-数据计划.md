# 检测 round2 · 数据扩展计划（扩数据 → 教师打框 → LS 复核 → 分布收口）

> 承 round1（[[决策① MD 教师 PASS]]、[[决策② 留 NanoDet-416]]，报告 `results/detect/round1/`）。
> round1 只在 ENA24+CCT（相机陷阱域）训、分布失衡；round2 **补真实喂食器域 + bird 覆盖 + person，并把分布拉合理**。
> 相关：[[ADR-0004]]（5 类）、[[ADR-0005]]（teacher+区域）、[[ADR-0006]]（acquire seam + iNat/Roboflow adapter）、
> [[ADR-0007]]（ONNX logits，已落地 #59）、`docs/classify/01-数据集.md`（iNat/物种/区域先验）。

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

## 6. 前置/阻塞

- **你（用户）**：申请 **Roboflow 访问 + 拿 `ROBOFLOW_API_KEY`**（feeder 域 + person 选2 卡这，最大短板源）。
- **box GPU**：iNat MD 伪标注 + NanoDet round2 微调（GPU 已停，需重开）。
- **LS 实例**：Label Studio 搭起来（低置信+稀有类人审）。
- 相关 issue：#60（round2 微调，本计划的执行）。加源 = box 上 `acquire`+`build`，零改代码（ADR-0006）。
