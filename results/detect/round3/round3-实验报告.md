# round3 检测段 · 实验报告

> 版本化落盘（项目 §5）。外部镜像见飞书 `docx/OAx3dq7v3or1rhx8Cbocum9dnRg`。数据/日志在 GPU box bjb1 `/root/autodl-tmp/`。
> 生成 2026-07-08，随 P5/后续实验滚动更新。

## 1. 一句话结论

round3 = **两个正交的赢**：**数据**（round2→P0，feeder 部署域 **+21.6** AP50）+ **模型**（ppyoloe-s，相机陷阱远域 **+9.6**）；crop 再补 feeder **+1.6** → NanoDet+数据+crop 的 feeder 达 89.3，离大模型 ppyoloe（90.5）**只差 1.2**。

## 2. 四模型对比（同 round3 评测，整体 AP50）

| 模型 | feeder（部署域）| test_test（相机陷阱域）| V861 部署 |
|---|---|---|---|
| round2-best | 66.1 | 65.5 | — |
| round3-P0（NanoDet + round3 数据）| 87.7 | 65.5 | 快 / 1.35MB / 零回退 |
| **P2（P0 + crop-to-feeder）** | **89.3** | 65.3 | 同上 |
| ppyoloe-s（M 线，640²）| 90.5 | **75.1** | 4.24fps / 18MB / 支持表在列 |

- **数据**（round2→P0）：feeder +21.6，纯度（对角率）0.899→0.995、squ→bird 混淆 28→1、other recall 0.31→0.74。feeder 部署域的主引擎。
- **crop**（P0→P2）：feeder +1.6（cat→100、other/squ/bird 各 +1.4~2.0，person −0.7），相机陷阱持平；无遗忘（canary bird −1.7，test_test squ +1.5，canary squ 仅 2 样本=噪声）。
- **模型**（→ppyoloe）：相机陷阱 +9.6，主要来自小/中目标（small-AP 9×、medium 3.5× P0）——640² 分辨率 + 容量对小/远的红利，**P 线数据补不上**（416² 天花板）。
- **旋转（P0→P5，负结果）**：feeder **−1.4**（person −3.6/other −2.4 最惨）、test_test −0.3 —— 旋转伤性能（竖直目标旋转造不真实样本）。**弃用**。验证归因价值：不是所有 aug 都好。⇒ 最佳 NanoDet = **P2（P0+crop）feeder 89.3**，不含旋转。

## 3. 其它评测口径

- **阈值 0.4/0.5/0.6**：纯度已封顶（0.994），提阈值只掉 recall（bird 0.84→0.74）→ **0.4~0.5 甜点，默认取 0.5**；阈值是 CPU decode 侧旋钮、不焊模型，移动端/板端可调。
- **top1-top2 margin**（回答最初"置信度都差不多"）：ppyoloe 中位 margin 0.662、97% 检测 >0.3、仅 1% <0.1 → 预测自信清晰，无模棱两可。
- **逐尺度 AP**（test_test）：small ppyoloe 0.274 vs P0 0.030、medium 0.576 vs 0.166、large ~0.76 vs 0.486。

## 4. 选型决策

- **feeder 近景为主 → NanoDet + 数据 + crop**：89.3（离大模型 1.2）、快得多、1.35MB 上 V861 零回退。
- **要看清远/小（相机陷阱域）→ ppyoloe-s**：75.1 vs 65.3（+9.8）；4.24fps/18MB 能上但慢。
- 取决于产品：观鸟器看多远、要多少帧率。精度都够，权衡速度。

## 5. 交付物

- **移动交接包**（按类型/轮组织，`detect/` 下，同类型=同接口）：
  - `mobile_handoff_nanodet/{round2, round3/p0, round3/p2}/`：ONNX + TFLite(fp32/fp16) + `ckpt`(续训) + `nanodet_detect.py`。P2=最佳 NanoDet 部署。**三模型 tflite≡onnx 全验**（P0 max|Δ|1.8e-5、P2 2.3e-5、cosine 1.0）。
  - `mobile_handoff_ppyoloe/round3/`：ONNX + TFLite(fp32/fp16) + `ckpt`(pdparams/pdopt/pdema 续训) + `ppyoloe_detect.py` + demos。接口对齐 round2（喂 BGR→`{label,score,box}`）。三方验证 cos 1.0、真图 IoU 0.99。
  - ⚠️ 导出坑：ppyoloe 训练用 paddle 3.3.1（PIR）paddle2onnx 不支持 → 建 paddle 2.6.2 专用 env 绕过；NanoDet onnx2tf 前须 onnxsim 简化（ShuffleNet channel-shuffle）。
- **视觉对比网格**：`grid_4model.jpg`（72行×4列 GT|P0|P2|ppyoloe-s，每框标注 类别+置信度）+ grid 统计图，已进飞书 §13。ppyoloe 召回/置信更高但负样本误报更多（7 vs 1–2）。
- **板端交接包** `detect/v861_awnn_nanodet/round2/`（AWNN `_ipu`，从 `round2/v861_awnn` 迁来统一管理）。
- **SwanLab** `@maxen/edge-cam`：round3-p0 / round3-M-ppyoloe-s / round3-P2-crop / round3-P5-rotation / round3-M-ppyoloe-m。
- **配置**：`configs/nanodet_round3_p{0,2,5}.yml`、`ppyoloe_round3_m.yml`、`ppyoloe_round3_ftm.yml`（m 容量参考）。

## 6. 进展 / 待完成

- ✅ **P5 旋转**：负结果（feeder −1.4），已弃用（见 §2）。
- ✅ **ppyoloe-m 容量参考**（ep23 停）：val best AP **0.694 < ppyoloe-s 0.703** —— 2× 模型没赢小模型，**容量非瓶颈、640² 分辨率才是**。m 1.81fps/43MB 更难部署 → 确认 **ppyoloe-s 是 M 线答案**。产物留服务器 `output_ftm/`，不拉本地。
- ⏸ **P3 copy-paste 小/远鸟**：据数据偏低价值（NanoDet 小目标短板是分辨率天花板、非数据量）→ 暂不跑。
- ⏸ round2 test_test 精确逐类（ckpt 在，可补；round2≈P0 该域 Δ0）。

**round3 P 线已收敛**：数据(P0)+crop(P2)=真赢，旋转(P5)=负，P3=低价值。**最佳 NanoDet 部署模型 = P2**（feeder 89.3、快、1.35MB）；要远/小则 ppyoloe-s。
