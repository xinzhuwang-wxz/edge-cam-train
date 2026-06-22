### 检测可行性包络 · nanodet_plus_320 (5 类) · feeder 4src v1 (74978)
| 级 | mAP@.5:.95 | mAP@.5 | bird_recall@.5 | vs fp32 (mAP) | 备注 |
|---|---|---|---|---|---|
| fp32 | 0.459 | 0.679 | — | — | FP32 ONNX |
| int8_sim | 0.413 | 0.630 | — | -0.046 | ORT-QDQ 模拟，非板子实测 |

**Gate: PASS ✅**

- （未设阈值，不设硬门 —— 先看包络再定，ADR-0001）
