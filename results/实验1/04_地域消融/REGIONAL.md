# 地域过滤(likely-species mask)on/off 消融

本地 ORT 跑 eff_lite0 fp32 onnx 在 birds525 test(2539)上,US eBird 区域清单(regions/us.json)。

| 口径 | mask OFF | mask ON | 说明 |
|---|---|---|---|
| 全局 525 类 (n=2539) | 0.9212 | 0.4155 | ⚠️ artifact:非美国种真值被 mask 压成 -inf 必错(issue#11),非真增益 |
| **in-region 子集**(真值∈US, n=1130) | 0.9204 | **0.9336** | ✅ 真实地域增益 **+1.3pt** |

- 地域覆盖:228/525 = 43.4%(US 区域内物种占比)。
- **真实结论:加地域过滤 vs 不加 = +1.3pt**(部署场景=鸟确实是本地种),免费(推理期软先验、不重训、可 OTA)。
- 那个全局 -50pt 暴跌是**测法错误**(全球 test 上加区域 mask 把外地真值压掉),不是地域过滤的实际效果——修正了 实验1 早期 envelope 的 regional artifact。
- 增益温和原因:birds525 模型本强,mask 仅纠正"误判成外地相似种"的少数样本;部署地相似种更多/模型更弱时增益更大。
