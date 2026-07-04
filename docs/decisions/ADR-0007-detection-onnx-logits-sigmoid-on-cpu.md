# ADR-0007：检测 ONNX 出 logits，sigmoid + 后处理统一留 A7 CPU

- 状态：Accepted
- 日期：2026-07-04
- 相关：CLAUDE.md §4（检测后处理 NMS/decode/grid/**sigmoid**/anchor 留 A7 CPU）、§6（避坑算子 / 激活软算子回退）、[[ADR-0003]]（模型族 backend seam）、`src/edge_cam/cascade/adapters.py`（`decode_nanodet`）、`results/detect/round1/报告.md` §7（喷框根因）

## 背景

round1 揭出一个**产品级隐患**（报告 §7）：`decode_nanodet`（本仓 numpy 后处理）在真实 NanoDet ONNX 上
**喷框**——feeder 1261 框/图、NanoDet-FT 1394 框/图、查准≈0.001。

根因诊断（`feeder_diag.py`，非猜）：**NanoDet 的 `export_onnx.py` 把 sigmoid 烤进了图**——导出 ONNX 的
cls 通道输出范围 [0, 0.8]（**已是概率**）。而 `decode_nanodet` 假设输入是 logits、**又做一次 sigmoid**：
- 真目标分数 0.8 → sigmoid → 0.69；背景 0.0 → sigmoid → **0.50**。
- 于是背景（0.50）也高于 conf 阈值（0.2/0.35）→ **全部 3598 anchor 都"过关"** → NMS 后喷 ~1300 框。
- 框本身是**对的**（最高分框正压在目标上），坏的是**分数被双重 sigmoid 压平、conf 阈值失效**。

这暴露一个**未统一的约定**：ONNX 到底出 logits 还是概率？两处代码假设不一致。若不定死，cascade 的
`OnnxDetector` 上线会喷框（白烧算力 / 可能超时），且换模型族/换导出脚本会反复踩。

## 决策

**检测 ONNX 只输出 logits（裸 backbone+head，sigmoid 前）；sigmoid + decode + grid + NMS 全部在 A7 CPU。**

即回归并**收紧** CLAUDE.md §4——sigmoid 明确属于「留 CPU」那一段，不进 NPU 图。

### 为什么 sigmoid 留 CPU（而非留在图里）

1. **§4 硬规则本就如此**——一致性，不是新发明。
2. **INT8 量化 / 软算子风险**：sigmoid 进 INT8 NPU 图有精度损失（饱和）+ **Vivante 上可能回退软算子**
   （§6 明确要求激活函数实测）。NPU 图保持「纯 conv/线性 head」对 ACUITY/pegasus 最可预测。
3. **CPU 上极便宜**：cls head 仅 `num_anchors × num_classes ≈ 3598×5 = 1.8 万`个值，A7 微秒级；且
   sigmoid 之后紧接 argmax/阈值/NMS（本就在 CPU）——它天然属于 CPU 后处理段。
4. **seam 干净**：ONNX = 纯 NPU 算（出 logits），CPU = decode+sigmoid+NMS，正好是「裸 backbone+head」。

### 落地（三条，round2 工程项）

1. **导出出 logits**：NanoDet `export_onnx.py` 在 head forward 里做了 sigmoid。**不 fork nanodet（§3）**，
   改用**导出后 ONNX 图手术**——剥掉 cls 分支尾部的 Sigmoid 节点（onnx graph surgery），得到 logits 输出。
2. **契约门（durable guard）**：新增校验——检测 ONNX 的 cls 输出必须是 logits（范围跨出 [0,1]，如
   `max > 5` 或 `min < -1`）；否则**报错**，与 FP32-only 门同款防呆。位置：`onnx_artifact` 导出校验 /
   `contracts/schemas/detection`。
3. **`decode_nanodet` 不改**——它对 logits 输入的 sigmoid **本就正确**；只需保证喂进来的是 logits。
   （round1 临时验证用的「去 sigmoid 版 decode」是针对当前 sigmoid'd ONNX 的**权宜**，不进主线。）

## 后果

- **正面**：喷框根除；NPU 图更小更稳（少一个激活算子的量化/回退风险）；ONNX↔decode 约定唯一、可门禁强制。
- **成本**：多一步导出后图手术 + 一条契约测试（round2 落地）。
- **不影响 round1 结论**：决策②（留 NanoDet）用的是原生 COCOeval，不走 `decode_nanodet`，结论不变。
- **待验证**：图手术后的 logits ONNX 上板量化是否与预期一致（W1 盲点，需板子）。
