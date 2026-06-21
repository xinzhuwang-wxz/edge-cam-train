# feeder_320 · 粗检测训练指标汇总（NanoDet-Plus 320）

> 来源：3090 box `~/autodl-tmp/ect/outputs/detect/feeder_320`，完整训练跑 = `logs-2026-06-21-01-52-54`
> （前两个 01-39 / 01-41 为中断短跑，已忽略）。本目录是该跑**训练→评估阶段**全指标的本地存档。
> 量化/部署阶段指标见文末「流程位置」——**尚未产出**。

## 一句话

NanoDet-Plus / ShuffleNetV2 1.0x / 320×320 / 5 类，30 epoch 收敛干净，**best=epoch30 mAP@.5:.95=0.459 / mAP@.5=0.679**。
类间差异符合数据分布：**cat 最好（mAP 62.3）> squirrel 58.0 ≈ other_animal 56.4 > bird 34.1 ≫ person 18.7**；
小目标几乎检不到（AP_small 0.009）。

## 训练配置（train_cfg.yml 摘）

| 项 | 值 |
|---|---|
| 架构 | NanoDet-Plus（ShuffleNetV2 1.0x backbone + GhostPAN + NanoDetPlusHead，aux head detach_epoch=10）|
| 输入 | 320×320，keep_ratio=false |
| 类别 | bird / squirrel / cat / person / other_animal（5）|
| 损失 | QFL(w=1.0) + GIoU(w=2.0) + DFL(w=0.25)，reg_max=7 |
| 训练 | batch 96 · fp32(precision 32) · 30 epoch · 711 iter/ep（≈21330 step）· grad_clip 35 |
| lr | 余弦，末段 ≈5e-5 |
| 数据 | train `labels/train_train.json` / val `labels/train_val.json`（detect_raw/processed）|
| 评估 | CocoDetectionEvaluator（pycocotools），save_key=mAP |

## 整体指标（COCO，逐 10 epoch）

| epoch | mAP\@.5:.95 | mAP\@.5 | AP\@.75 | AP_small | AP_medium | AP_large |
|---|---|---|---|---|---|---|
| 10 | 0.387 | 0.594 | 0.417 | 0.007 | 0.059 | 0.454 |
| 20 | 0.447 | 0.664 | 0.480 | 0.009 | 0.076 | 0.521 |
| **30 (best)** | **0.459** | **0.679** | **0.496** | 0.009 | 0.080 | 0.535 |

**best(epoch30) 的召回 AR**：AR@1 0.427 · AR@10 0.559 · AR@100 0.595 · AR_small 0.021 · AR_medium 0.284 · AR_large 0.687

> 数据细节见 `overall_metrics.csv`（机读）/ `eval_results.txt`（原始）。

## 每类 AP（AP50 / mAP\@.5:.95，单位 %）

| 类 | epoch10 | epoch20 | epoch30(best) |
|---|---|---|---|
| bird | 51.0 / 29.7 | 56.3 / 33.4 | **57.2 / 34.1** |
| squirrel | 68.3 / 47.0 | 78.6 / 56.1 | **80.6 / 58.0** |
| cat | 75.6 / 53.6 | 83.0 / 60.9 | **84.7 / 62.3** |
| person | 31.8 / 15.0 | 36.5 / 18.0 | **37.9 / 18.7** |
| other_animal | 70.1 / 48.0 | 77.4 / 54.8 | **78.9 / 56.4** |

> 机读见 `per_class_ap.csv`。**bird 召回是产品第一优先**（喂鸟器）——目前 bird mAP 34.1 偏中等、person 最弱、
> small 目标几乎丢；后续 416/1.5x 档与数据补强主要看这几格能否抬起来。

## 🐦 每类召回率（fp32, val 800 子集, conf≥0.3 / IoU≥0.5 / 类别正确）

| 类 | GT | TP | 召回率 |
|---|---|---|---|
| **bird** | 201 | 176 | **0.8756** |
| squirrel | 114 | 86 | 0.7544 |
| cat | 17 | 14 | 0.8235 |
| other_animal | 530 | 497 | 0.9377 |
| 总体 | 862 | 773 | 0.8968 |

> 口径对齐实验1。**bird 召回 87.56% vs 实验1 64.5%（+23pt）** —— 关键反转：召回（识别到）大涨，
> 但 AP/mAP（框准度）偏低。不矛盾——更多/更难数据让模型「看到鸟更敢报」（漏检少→召回高），
> 但 320 输入定位不精 + precision 一般（误报多）→ AP 低。喂鸟器场景 **bird 召回是命门**
> （宁多框勿漏，后接分类器细判）→ feeder_320 实为改善。（person 子集 GT=0）。详见 `quant/per_class_recall.md`。

## 量化（int8 模拟掉点，ORT-QDQ per-channel/calib120，方向性非板子）

| 级 | mAP@.5:.95 | AP50 |
|---|---|---|
| fp32 | 0.459 | 0.679 |
| int8_sim | 0.413 | 0.630 |
| **掉点** | **-4.58pt** | **-4.83pt** |

> per-class 掉点 cat 最多(-7.5)、bird 较稳(-2.5)。4.58pt 明显大于实验1 的 0.19pt，疑 calib 120 张
> 对 4 源杂数据代表性不足 → 可增 calib 重测。fp32 ONNX 推理 mAP 0.4591 == 训练 best（导出零损耗）。
> 详见 `quant/`（report.md / per_class_fp32_vs_int8.md）。

## 训练 loss 收敛（TensorBoard，427 step 采样点）

| loss | 起 | 末 |
|---|---|---|
| loss_qfl | 1.007 | 0.266 |
| loss_bbox | 1.657 | 0.378 |
| loss_dfl | 0.520 | 0.202 |

> 全曲线（含 aux_loss_* 与 lr）见 `train_loss_curve.csv` / `metrics_scalars.json`；原始 TB events 在 `tb_runs/`，可 `tensorboard --logdir` 重画。

## 文件清单

| 文件 | 内容 |
|---|---|
| `README.md` | 本汇总 |
| `overall_metrics.csv` | 整体 AP（逐 10ep）机读 |
| `per_class_ap.csv` | 每类 AP（逐 10ep）机读 |
| `val_metrics_curve.csv` | Val 6 项指标曲线（=整体表，TB 抽）|
| `train_loss_curve.csv` | 训练 7 条曲线（6 loss + lr，逐 step）|
| `metrics_scalars.json` | TB 全 13 个 scalar tag 原始点 |
| `eval_results.txt` | box 原始评估输出（每 10ep）|
| `train_full.log` | 完整训练日志（96.8 KB）|
| `train_cfg.yml` | 训练配置原件 |
| `tb_runs/` | 原始 TensorBoard event 文件（可重画）|

> best 权重 `nanodet_model_best.pth`(17M) 不在 git，落 `outputs/detect/feeder_320/model_best/`（导 ONNX 用）。
> 完整预测 dump `results0.json`(42M) 留在 box（重跑 COCOeval 才需要）。

## 流程位置（这份在哪一段）

```
[训练 ✅本档]──导出FP32 ONNX──ORT-QDQ模拟INT8──四级包络──gate──总表  ← 本地全流程，尚未跑
                                                              └─ACUITY/.nb（真上板）← W1，无工具链
```

- **已完成**：训练 + COCO 评估（本目录全部指标）。
- **待跑（本地 CPU，不上 box）**：FP32 ONNX 导出 → `quant_estimate` ORT-QDQ 模拟 INT8 掉点 →
  `evaluators/detect` 四级包络 → `gate` → `detect_ablation.csv` 总表。跑完后量化阶段指标续存到本目录。
- **缺口**：检测目前没有像分类 `run_envelope` 那样的端到端入口（`run_envelope` 是分类专用），
  需补一条检测编排把上面这串串起来。
