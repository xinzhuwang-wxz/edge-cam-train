# GPU 实跑记录 — 2026-06-20（RTX 5090 / westd seetacloud）

首次租卡端到端执行：数据→训练→FP32 ONNX→四级可行性包络→质量门→ModelCard→registry→promote。

## 模型
- efficientnet_lite0（timm ImageNet 预训练全量微调）· 525 类 · BIRDS-525
- 80 epoch · AdamW lr1e-3/wd1e-4 · label_smoothing0.1 · cosine · 退化增强 · input 224 · batch 128
- 训练 ~12 分钟（RTX 5090）；FP32 ONNX 导出 + onnxruntime 对齐校验 = True

## 可行性包络（plan §B.3/§8）
| 级 | top-1 | top-5 | vs FP32 | 说明 |
|---|---|---|---|---|
| fp32_val | **0.933** | 0.976 | — | 干净验证集 |
| int8_sim | **0.925** | 0.971 | −0.008 | ORT-QDQ 模拟 INT8，**掉点极小（量化友好）** |
| field | 0.812 | 0.920 | −0.120 | 退化代理 domain-gap（≠真现场） |
| regional | 0.417 | 0.437 | −0.516 | ⚠️ 见「发现 1」：测法 artifact，非真实回归 |

Gate: PASS（默认无硬阈值，ADR-0001）。已 promote 到 stable channel。

## 结论
- **量化几乎不掉点（−0.8pt）** → efficientnet_lite0 对 INT8 PTQ 友好，契合边侧 NPU。
- 真现场退化估计 ~81%（field 代理），需真机回采校准。
- 全链路闭环验证成功。

## 实跑发现（待修/已修）
1. **regional 级测法 artifact**：在全球 525 类 test 上加区域 mask，把非美国种真值压 -inf → 必错；
   应只在 in-region 子集比 on/off。地域过滤本身有益，此数读反。→ 需修 envelope regional 口径（issue）。
2. **eval 默认走 CPU + num_workers=0**，慢（~15 分钟）→ 已修：run_full_eval device 自动选 GPU、
   run_envelope num_workers 默认 4。
3. **int8 导出目录未先 mkdir** → FileNotFoundError → 已修：full_eval 写盘前 mkdir。

## 环境要点（下次租卡省时）
- RTX 5090=Blackwell sm_120，须 torch cu128（torch 2.10.0+cu128，CUDA 12.8）。
- 该云外网慢（pytorch.org/HF/github ~1MB/s）；解法=**mac 跳板**：mac 下 wheel(15MB/s)→scp(16MB/s)→box 离线装。
- HF 走 hf-mirror.com；pip 走 aliyun；github 不通 → 用 git archive+scp 传仓库。
