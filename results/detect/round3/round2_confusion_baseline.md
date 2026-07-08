# round2 main(1.0x@416) 混淆基线 — round3 M1 里程碑（无卡实测）

> 工具：`src/edge_cam/eval/detect_confusion.py`（IoU 匹配 + background 行列，行=真值/列=预测/对角=类正确）。
> 数据：round2 main 的 **固定 test**（`test_eval/main/.../results0.json`，11656 图）+ **worksite 部署域代理**
> （`worksite_eval/main_detections.json`，870 图）。阈值 conf=0.40（部署）/ iou=0.5。
> **⚠️ 首跑踩坑**：误用了 `main/results0.json`（那是 **val** 预测，3785 图）配 test GT → 全不重叠（对角 0.65 假象）。
> 已改用正确的 test 预测。教训：`main/results0.json`=训练末 val 评估，test 预测在 `test_eval/<run>/<ts>/`。

## 固定 test（相机陷阱/混合域，泛化真值）conf=0.4

- **对角线占比 = 0.899**（已定位框 90% 类判对，判别本身不差）；类判错率 0.101。
- **逐类 recall**（判对/该真类 GT 总，含漏检）：bird **0.80** · squirrel **0.44** · cat **0.40** · person 0.51 · other **0.82**。
- **最亮错分对**（原始计数）：cat→other **271** · squirrel→other **164** · bird→other 68 · other→cat 47 · **cat→bird 29** · **squirrel→bird 24** · other→bird 24 · bird→squirrel 22 · cat→squirrel 22。

| true\pred | bird | squirrel | cat | person | other | bg(漏检) |
|---|---|---|---|---|---|---|
| bird | 2626 | 22 | 4 | 4 | 68 | 567 |
| squirrel | 24 | **493** | 13 | 0 | **164** | **429** |
| cat | 29 | 22 | **552** | 0 | **271** | **520** |
| person | 10 | 0 | 1 | 699 | 1 | 648 |
| other | 24 | 17 | 47 | 2 | 2064 | 359 |
| **bg(误报)** | **442** | 273 | 125 | **442** | 457 | 0 |

## worksite（部署域代理，中大近景，乐观）conf=0.4

- **对角线占比 = 0.988**（几乎无 argmax 混淆）；**squirrel→bird = 0**、bird→squirrel = 1。
- 逐类 recall：bird 0.92 · squirrel **0.72** · cat 0.86 · person 0.89 · other 0.92。
- 唯一可见错分：cat→other 5。

| true\pred | bird | squirrel | cat | person | other | bg |
|---|---|---|---|---|---|---|
| bird | 477 | 1 | 0 | 0 | 1 | 39 |
| squirrel | 0 | 68 | 0 | 0 | 1 | 26 |
| cat | 0 | 0 | 78 | 0 | 5 | 8 |
| person | 0 | 0 | 0 | 115 | 0 | 14 |
| other | 0 | 1 | 1 | 0 | 90 | 6 |
| bg | 29 | 13 | 5 | 22 | 9 | 0 |

## 结论（M1 → 调整 round3 重点）

1. **"松鼠→鸟"argmax 混淆量级小但非零**（worksite 0 / test squirrel→bird **24/1123=2.1%**）。**共激活复核**（数非鸟 GT 被
   bird 预测 IoU≥0.5 点亮）：squirrel **24 = 矩阵值** → 贪心匹配**没有**系统性低估 squirrel→bird；合计非鸟区域点亮 bird 仅
   **104**（squ24/cat37/per12/oth31）。**⚠️ 但这只覆盖 NMS 存活的独立 bird 框，不覆盖"同框 bird 紧随第二"的 margin 现象**
   （须 GPU 全类分）。用户 dogfood 的"站立松鼠→鸟"最可能在**未标注野图**（域外）或是 **margin**（见 3）。→ 补部署域带标 eval + margin。
2. **真正的可测短板 = 次要类 recall + other_animal 黑洞**：
   - squirrel/cat 在难 test 上 recall 仅 44%/40%（大量漏检→bg 或被 other 吸走）；
   - **cat→other 271 / squirrel→other 164** —— other_animal catch-all 在吸收次要类（**新发现**）。
   - → round3 数据主攻**次要类 recall + 与 other_animal 的判别分离**（不只是"别叫成鸟"）。
3. **"触发后置信度都差不多"是 margin 问题、不是 argmax**：squirrel argmax 判对但 bird 可能紧随（§2 sigmoid 非竞争）。
   混淆矩阵取 argmax 看不到 → **top1-top2 margin 指标（需 GPU dump 全类分）是量化用户诉求的唯一工具**，列为 round3 高优。
   佐证：test **bg→bird=442** 误报里，复核仅 **104** 个非鸟 GT 区域被 bird 点亮，**其余落在真背景** → 误报主要是
   "背景误检成鸟"、非"动物误检成鸟"。（margin 现象藏在同框第二分，此复核看不到，须 GPU。）
