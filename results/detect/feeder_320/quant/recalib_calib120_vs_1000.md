# int8 增 calib 对比实验（控制变量，val 1600 子集）

| 档 | mAP@.5:.95 | AP50 |
|---|---|---|
| fp32 | 0.4719 | 0.7266 |
| int8 @calib120 | 0.4082 | 0.6436 |
| int8 @calib1000 | 0.4102 | 0.6484 |

**掉点（同子集相对值）**：mAP calib120 -6.37pt → calib1000 -6.17pt（压回 0.20pt）；AP50 -8.30pt → -7.82pt（压回 0.48pt）。

**结论：增 calib 收益极微 —— 掉点主因是 ShuffleNetV2 INT8 固有量化损失，非校准样本不足。**
要压掉点需换方向：混合精度(敏感层保 fp32，但 V85x INT8-only) / QAT / backbone 选型。
ORT-QDQ 仅方向性，真值待板子 ACUITY（VeriSilicon 量化器或更友好）。
