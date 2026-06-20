# GPU 实跑记录 — 2026-06-20（消融 + 检测，双卡并行）

承接首跑 `../gpu_run_20260620/`（单模型闭环）。本轮做两件事：
**① 分类 B.3 受控消融（6 格）+ 最优候选 test 全包络对比；② 检测 B.2 NanoDet-Plus 全量微调 + ONNX + int8。**
两卡并行：**RTX 5090（westd）= 分类**；**RTX 3080（nmb2）= 检测**（NanoDet pin torch<2.0，跑不了 5090 的 sm_120）。

---

## ① 分类消融（plan §B.3，birds525 / 525 类）

受控网格：backbone × input_size，单变量。80 epoch · AdamW lr1e-3/wd1e-4 · ls0.1 · cosine · 退化增强 · batch128 · aim 追踪。
**选型只在 val（不碰 test，§B.0）：**

| backbone | input | fp32_val_top1 | top5 |
|---|---|---|---|
| efficientnet_lite0 | 192 | 0.9154 | 0.9737 |
| efficientnet_lite0 | 224 | 0.9334 | 0.9809 |
| mobilenetv3_large_100 | 192 | 0.9274 | 0.9777 |
| mobilenetv3_large_100 | 224 | **0.9362** | 0.9809 |
| repvgg_a0 | 192 | 0.9146 | 0.9629 |
| repvgg_a0 | 224 | 0.9206 | 0.9657 |

val 头名 mobilenetv3@224，仅比 eff_lite0@224 高 0.28pt → 取两强进 test 全包络对比。

### 最优候选 test 全包络对比（含纯 int8 掉点）

额外加 fp32_test（排除 val→test 分布差），int8 掉点 = fp32_test − int8_sim（同 test 集）：

| backbone | fp32_val | fp32_test | int8_sim | **int8掉点** | field | field退化 |
|---|---|---|---|---|---|---|
| efficientnet_lite0 | 0.9334 | 0.9212 | **0.9193** | **0.0019** | 0.809 | 0.1122 |
| mobilenetv3_large_100 | 0.9362 | 0.9327 | **0.8956** | **0.0371** | 0.8436 | 0.0891 |

**决定性结论（翻盘）：** mobilenetv3 fp32 领先 1.15pt，但 int8 掉点是 eff_lite0 的 **~20 倍（3.71 vs 0.19pt）**
→ **量化后 eff_lite0 反超 2.37pt（0.9193 vs 0.8956）**。V85x 是 INT8-only，ships 的就是 int8 这个数
→ **efficientnet_lite0 是明确赢家**（已发布 registry stable）。坐实 §4：mobilenetv3 的 h-swish+SE 量化不友好。
（注：mobilenetv3 fp32 下 field 鲁棒性更好，但端侧不相关。）

---

## ② 检测 NanoDet-Plus（plan §B.2，feeder-cam 11 大类，416 输入）

COCO 预训练**全量微调**：加载 backbone+FPN(neck) → 头因 80→11 类重置（head.gfl_cls 43=11类+32回归；aux_head 11）
→ 8.4M 参数全部可训（非冻结）。50 epoch · batch32。

### val mAP（全集 3335 图，COCOeval）

| epoch | mAP@.5:.95 | AP_50 | AP_75 | AP_small | AP_large |
|---|---|---|---|---|---|
| 10 | 0.344 | 0.469 | 0.385 | 0.064 | 0.368 |
| 30 | 0.559 | 0.693 | 0.617 | 0.157 | 0.586 |
| 50 | **0.591** | **0.720** | 0.644 | **0.165** | 0.619 |

mAP 50ep 仍微升（40→50: +0.7pt），未完全饱和；延长到 80-100ep 或有边际收益。

### per-class AP（最终）

| 类 | AP50 | mAP || 类 | AP50 | mAP |
|---|---|---|---|---|---|---|
| **bird** | **61.6** | **45.5** || squirrel | 82.1 | 69.3 |
| cat | 91.2 | 76.6 || dog | 93.1 | 79.9 |
| rabbit | 87.1 | 70.1 || fox | 80.5 | 73.8 |
| hedgehog | 90.6 | 73.6 || deer | 78.7 | 60.8 |
| bear | 71.2 | 57.4 || raccoon | 55.1 | 42.1 |
| **skunk** | **1.3** | **0.9** ⚠️ |||||

### FP32 ONNX 导出（§4 铁律）

`nanodet_feeder_416.onnx`：输出 **(1, 3598, 43)** = 3598 anchor × (11 类 + 32 回归分布)，
**纯裸 head 输出、无 NMS/decode**（decode/NMS 留 A7 CPU）。onnxsim 简化通过，5.4MB。

### 检测 int8 掉点（800 图子集，新建评估链）

为"检测也要看量化"专门搭：`int8 ONNX → NanoDet head.post_process(decode+NMS) → COCOeval`，
复用 NanoDet val dataset 预处理 + 评估器。ORT-QDQ per-channel（opset 升 11→13）：

| | mAP@.5:.95 | AP_50 |
|---|---|---|
| fp32_onnx | 0.6813 | 0.8113 |
| int8_onnx | 0.6794 | 0.7917 |
| **int8掉点** | **0.0019（0.19pt）** | 0.0196（1.96pt） |

**检测也量化友好**：mAP 掉点 0.19pt（AP_50 掉 1.96pt，int8 略伤宽阈值定位/置信）。
（子集绝对值高于全集 0.591，因前 800 图偏易；**掉点看同子集**才有效。ORT 仅方向性，真值待板子 ACUITY。）

---

## 级联量化总结

两段都对 INT8 PTQ 友好（**分类 −0.19pt top1 / 检测 −0.19pt mAP**），契合 V85x（0.5–1 TOPS，INT8-only）。

## 待办风险（建议进 issue / 后续优化）

1. ⚠️ **bird AP50=61.6 偏低**（多数类 80-93）。根因 **AP_small=0.165** —— 喂食器场景鸟又小又远。
   级联里粗检测器的 bird 召回 = 命门，**漏检风险真实**。改善：输入 416→512、补鸟数据、按小目标调尺度。
2. ⚠️ **skunk 几乎全军覆没（AP 1.3）**：大概率样本极少（类不均衡），需补数据或并类。
3. 检测 mAP 未饱和（延长训练）；检测 int8 仅 800 子集（可全量复核）。

## 环境/踩坑（下次省时）

- **3080 检测**：base env python3.8 + torch1.11+cu113；NanoDet 依赖须 `pip install -r requirements.txt`（含 cv2，手列易漏）。
- **数据完整性**：detection_feeder/train 有 1 张 0 字节图（`000000007627.jpg`，COCO 源就缺）→ cv2.imread None 致训练崩；
  已从 labels.json 剔除（15000→14999，备份 .bak）。**data prep 管线建议加 0字节/损坏过滤**。
- **沙箱 SSH 启远端长任务**：含 `&` 命令整条静默失败、`pkill -f` 自匹配杀父 → 一律 `screen -dmS` 脱离 + `pkill [t]` 括号技巧。详见 memory。
- 检测 int8：NanoDet 导出 opset 11，per-channel QDQ 需 ≥13 → 脚本内 `version_converter` 升级。
