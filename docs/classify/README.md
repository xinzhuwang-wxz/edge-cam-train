# 细分类流程（鸟种识别）— 文档家

> 边侧喂鸟器**鸟种细分类器**的全过程细化文档。级联第二段：粗检测出 **bird 框 → crop → 本细分类器**
> 出种。端侧只认**常见种**，长尾/罕见/疑难交**云端**。目标板 Allwinner V85x（Vivante VIP NPU，INT8-only）。
> 配根 `CONTEXT.md`、`docs/plan-v2.md`、`docs/decisions/`（[[ADR-0001]] 可行性优先、[[ADR-0002]] eBird
> taxonomy seam、[[ADR-0003]] 模型族 seam、[[ADR-0005]] 分类数据许可红线 + teacher）读。
>
> 📍 **当前阶段焦点之一**（与 `docs/detect/` 对等）。方案据**联网调研核验 + 实验1 结论**落定，许可从严。

## 文档索引
| 文档 | 内容 |
|---|---|
| [01-数据集.md](01-数据集.md) | 数据源盘点（iNat/GBIF/自建；排除 NABirds/CUB/eBird-freq）、license 分层、下载清洗、物种表+taxonomy crosswalk、区域候选集、数据量目标 |
| [02-训练与评估.md](02-训练与评估.md) | backbone（Lite0，实验1 证据）、训练阶段、类不均衡、层级输出+置信门控、评估口径、端云部署、量化 |
| [03-实操日志.md](03-实操日志.md) | 活文档：分类阶段环境/下载/训练/调试一切实操 |
| (后续) 04-导出与部署 | FP32 ONNX → ACUITY PTQ → .nb → VIPLite（待板子） |

## 整体链路

```
检测器出 bird 框 → 框外扩 10~20% 裁剪 → 图像质量判断
   → 端侧 EfficientNet-Lite0 细分类（仅常见种）
        ├─ 高置信        → 直接出种 + top-5
        ├─ 中置信/近似种 → 留云 API 锚点（不实现云端）
        └─ 低置信/不在地域清单/稀有种 → 层级回退（genus/family/bird）
   → 结合经纬度+月份+历史 重排候选（GBIF 自建先验，非 eBird）
端侧常见种、云端长尾/罕见/疑难。
```

## 关键决策速查（据调研 + 实验1，[[ADR-0005]]）

- **端侧 backbone：EfficientNet-Lite0**。实验1 实测 INT8：lite0 掉 0.19pt vs mobilenetv3 掉 3.71pt
  （MobileNetV3 的 SE/h-swish 在 INT8 NPU 上掉点大）→ INT8-only 板子上 lite0 是赢家。Lite1/2 留扩容。
- **训练数据只用可商用**：iNaturalist Open Data **逐图过滤 `CC0/CC-BY`**（默认是 CC-BY-NC，可商用是少数）
  + GBIF **逐图 `media.license` 过滤** + 自建喂鸟器。权重天生可发行、无传染（§4）。
- **区域/季节先验：GBIF CC0/CC-BY occurrence 自建**（eBird 频率数据**商用禁止**，不能发行）。
  评估**必出 with/without 对比**（量化区域增益，承实验1 +1.3pt 口径）。
- **第一版不蒸馏**：BioCLIP/BioCLIP2 训练数据含 NC（污染）、SpeciesNet 是 Apache 但服务器级+输出授权灰区
  → teacher 只能**自训 clean**；soft_label hook 已在（#7 未来）。
- **评测**：主指标 = 自有 CC0/CC-BY **test split**（防泄漏切分）；泛化 = **自建跨区域/跨源 holdout**；
  **NABirds/CUB 排除出商用线**（Cornell/Caltech 非商用、不得做产品）。
- **类别**：稳定**物种表**（taxon_id/scientific_name/genus/family/ebird_code/inat_taxon_id/gbif_taxon_key/
  region/edge_enabled），**层级输出**（species→genus→family→bird），不强猜种。eBird 物种**代码**当 ID 串可用
  （受限的是频率数据）。

## 一图看懂：数据源 → 角色 → 用在哪
```
可商用(CC0/CC-BY): iNaturalist(逐图过滤) + GBIF(逐图过滤) ──训练──► EfficientNet-Lite0 端侧分类器
自建喂鸟器(未来)  : feeder crop ──────────────────────────微调──►  ↑（决定真实效果）
GBIF occurrence   : ──聚合每区域每月频率──► 区域候选集 + Top-K 重排（推理期，OTA）
排除(非商用/污染) : NABirds·CUB(非商用) · eBird频率(商用禁止) · BioCLIP(NC污染) → 不进商用线
评测             : 自有 test(主) + 自建跨源 holdout(泛化) + (未来)feeder test
```
