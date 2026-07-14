# V861 板端评测 · PP-YOLOE-s INT8 ≡ 上游 FP32（等价性验证）

> 2026-07-14 在 **V861 PER2 真板**上跑完。模型 = round3 `ppyoloe-s`（SPPF 优化版，`_ipu` 7.5MB）。
> **这不是一个新的精度基准** —— 精度的权威数字仍是 round3 的（`../../round3/round3-实验报告.md`：
> feeder 部署域 AP50 **90.5** / 相机陷阱域 75.1）。**本评测回答的是一个正交问题：
> 板子上的 INT8，是不是忠实复现了上游那个 FP32 模型？**

## 1. 方法

- **93 张** Wikimedia Commons 图（CC0/PD/CC-BY，署名见 `CREDITS.md`），喂食器/庭院视角，5 类 × 尺度分层。
- **两边喂完全相同的输入字节**：`cv2.imread → resize(640,640) → BGR2RGB → uint8 HWC`。
  ⚠️ 必须用 **cv2**：PIL 的 JPEG 解码器与 cv2 差最大 **34/255**（IDCT 实现不同），
  混进去会把"解码差异"算成"量化掉点"。
- FP32 侧：`ppyoloe_s_640_sppf_logits.onnx` @ onnxruntime CPU。
- 板端侧：`awnn_batch`（见 `vs861/awnn_batch/`）→ 6 个 FP32 blob → 同一份 `decode_ref.py` 解码。
- **观鸟器场景子集 61 张**（`data/scope.json`）：从 93 张里剔除域外图（刺猬/街猫/室内猫/版画/停车场
  —— 喂食器摄像头根本看不到的东西）。混淆矩阵只在这个子集上算。

## 2. 结论：板端 INT8 ≡ 上游 FP32（`assets/chart_align.png`）

| 指标 | 结果 |
|---|---|
| cls 头逐输出余弦相似度 | **0.9993 ~ 0.9996** |
| reg 头（DFL）余弦相似度 | 0.9885 ~ 0.9947 |
| 检出框数 | FP32 78 / 板端 80 |
| 框匹配率（IoU≥0.5 且同类） | **74/78 = 94.9%** |
| 匹配框平均 IoU | **0.9728** |
| 匹配框置信度绝对差 | **0.034** |
| 逐图完全一致 | **51/61 = 84%** |
| **板端 NPU 延迟** | **中位 226.1 ms**（min 221.8 / max 234.0, n=93, `use_awnn_profiler=0`） |

板端延迟对得上官方支持列表的 235.71ms（官方应是开 profiler 测的）。
⚠️ **口径**：`use_awnn_profiler=1` 会**多花 ~24ms**（同图 244.4 vs 220.6ms）但**不多占一个字节 NPU 内存**
（两边都是 18.0175MB）。部署态一律关 profiler → **226ms**。早期记的 238ms 是开着 profiler 测的。

**量化几乎无损。**

## 3. ⚠️ 关于混淆矩阵：域不匹配，不可当性能读

`assets/chart_confusion.png` 是这 61 张 web 图上的板端混淆矩阵。bird/person/other_animal **100%**、
squirrel 88%，但 **cat 只有 11%（12/18 被判成 other_animal）**。

**不要把这个 11% 当成部署性能** —— round3 在**真部署域（feeder）**测的 cat 是 **AP50 96.4**：

| 类 | round2 feeder | **round3 P0 feeder** |
|---|---|---|
| squirrel | 59.9 | **89.6** |
| bird | 76.1 | **86.9** |
| **cat** | 85.0 | **96.4** |
| person | 78.6 | **85.6** |

差距来自**域**：Commons 上的猫是**长焦浅景深的庭院写真**（人眼高度、猫占画面 30~50%），
而喂食器摄像头看到的是 1~3m 广角、草坪上走动、低分辨率的猫。round3 报告 §5.5 已明确
*"round2 固定 test 对 squirrel/cat 是错的域"* —— 同一个道理。

**但这里有个真正有价值的推论**：cat 在 **FP32 上是 17%、板端 INT8 是 11%** —— **两边同塌**。
⇒ 这是**模型/域**的问题，**不是量化的问题**。板子忠实地复现了上游模型的行为，**连它的弱点一起复现**。
这反而是等价性最有力的证据。

（若要真正评估 cat 的域外鲁棒性，需要**喂食器摄像头视角**的猫图，不是 web 写真。留作后续。）

## 4. ★ NPU 内存硬约束（`assets/chart_npu_mem.png`）

dynamic mode（`use_static_mode=0`）下，**每次 `awnn_instance_inference` 都新分配 IPU blob 内存且不释放**：

| 一个进程里推几张 | blobMemory | npuMemory | 结果 |
|---|---|---|---|
| **1 张** | 10.60 MB | **18.02 MB** | ✅ 正常 |
| **2 张** | 15.28 MB | **22.71 MB** | ❌ 顶破 20MB 保留池 → `dma_mem_alloc fail` |

**分配失败后继续硬砸会把内核挂死**（adb offline，需物理断电）。→ 批跑必须**一进程一张**；
真实产品必须解决这个（static mode / 每帧重建实例 / 调大 DTS `size_pool_mem`）。详见
`vs861/V861真板部署实录.md §10.5`。

## 5. 目录

```
assets/   gallery_bird.jpg      鸟类 demo（板端真实输出，小/中/大/多目标）
          gallery_others.jpg    其余四类 demo
          chart_align.png       ★ 等价性：余弦/IoU/置信度差
          chart_accuracy.png    round3 部署域逐类 AP50（权威精度）
          chart_npu_mem.png     ★ NPU 内存约束
          chart_confusion.png   板端混淆矩阵（域外，见 §3 警告）
          chart_class_recall.png FP32 vs 板端逐类召回（域外，见 §3）
data/     align.json / confusion_scoped.json / scope.json
          board_dets.json / fp32_dets.json / meta.json
CREDITS.md  93 张图的 CC-BY 署名清册
```
