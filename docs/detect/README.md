# 检测流程(粗检测：粗分类 + 框)— 文档家

> 📍 **当前阶段焦点**（见 `CLAUDE.md §2 文档地图`）。基座方向看 `docs/plan-v2.md`/`docs/engineering.md`，
> 本阶段「怎么落检测」看这里。
>
> 边侧喂鸟器**粗检测器**的全过程细化文档。粗检测 = 在整帧上找 **bird/squirrel/cat/person/other_animal** 5 类
> 并出框；**bird 框 → crop → 交细分类器**出种。目标板 Allwinner V85x（Vivante VIP NPU，INT8-only）。
> 配合根 `CONTEXT.md`、`docs/plan-v2.md`、`docs/decisions/`（[[ADR-0003]] 框架 seam、[[ADR-0004]] 5 类体系）读。

## 文档索引
| 文档 | 内容 |
|---|---|
| [01-数据集.md](01-数据集.md) | 数据集**设计**：开源盘点、5 类体系、license 分层、DatasetAdapter 抽象、负样本策略、存放结构 |
| [数据管线.md](数据管线.md) | **当前数据管线全流程**（[[ADR-0006]] 重写）：`acquire`(获取+收据) → MD 伪标注 → `build`(组装) → 产物(JSONL/署名清册) → 训练消费；命令速查、7 源现状、加新源 |
| [02-训练与评估.md](02-训练与评估.md) | 模型变体(NanoDet 三档)、COCO 预训练微调、数据增强、split、评估口径、MegaDetector 角色 |
| [检测实验计划.md](检测实验计划.md) · [round2-数据计划.md](round2-数据计划.md) | **round1** 实验计划（零样本谱+bake-off，已跑完 → `results/detect/round1/`）；**round2 数据计划**（扩数据→MD打框→LS复核低置信+稀有→分布收口，含分布目标+源映射，决策锁定） |
| [03-实操日志.md](03-实操日志.md) | **检测阶段实操日志（活文档）**：环境/下载/训练/调试一切动手遇到的——实际类目核验、踩坑(py3.8/107GB CCT)、每源处理 |
| (后续) 导出与部署 | FP32 ONNX → ACUITY → .nb → VIPLite(待板子) |

## 一图看懂:数据集 → 角色 → 用在哪步

```
                         ┌─────────────────── 训练集(只用可商用,权重天生可商用)───────────────┐
可商用(CDLA/CC-BY):       │  OpenImagesV7 · ENA24 · Caltech-CT · Roboflow喂食器               │
                         └──────────────────────────────┬──────────────────────────────────┘
                                                         ▼
                              COCO 预训练 NanoDet ──微调──► 5 类粗检测器(320/416/1.5x 三档对照)
                                                         │
仅可行性(COCO/NABirds): ─────────────────────────────────┤ 仅作额外评估集(泛化/bird 召回),绝不进训练
                                                         ▼
                              评估:test 集 + 可行性评估集 + 场景分组 + 设备指标
                                                         │
自建喂食器(未来,无框) ──[MegaDetector 伪标注]──► 进训练(v-next)│  MegaDetector 现在:在带框集上评它的框准(基线)
                                                         ▼
                              bird 框 → crop → 细分类器(种)   ← 级联(cascade 模块)
```

## 关键决策速查
- **5 类**:bird / squirrel / cat / person / other_animal（[[ADR-0004]]，治实验1 稀有类饿死）
- **训练只用可商用数据** → 权重无许可传染、可发行；可行性数据仅评估
- **模型**:NanoDet-Plus-m(ShuffleNetV2，无 Focus/无 SE-hswish，NPU 安全)；**不用 YOLOX-Nano**(Focus，§4)、PicoDet(ESNet 的 SE/h-swish 风险，实验1 有前车之鉴)
- **MegaDetector**:伪标注自有数据(未来)+ 现在评其框准；不上板、不当微调起点
- 每数据集 = 一个自注册 **DatasetAdapter**(自带 map/license/role/清洗/split 单位)
