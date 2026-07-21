# PP-YOLOE-s @ V861 NPU 板端交接包（round3 · v2）

> **✅ 真板跑通 + 93 张等价性验证 + autoresearch 调优收口**（2026-07-14 上板，07-21 升级 v2）。
> v2 变化：**静态模式（43ms/帧，快 5.3 倍）** + 自带**一体化推理工具 `tools/ppyoloe_run`**（push 即跑，
> JPG/NV21 输入 → JSONL + 标框图，**后端零后处理**）。
> 这是目前唯一能上 V861 NPU 的检测模型——NanoDet 因 ShuffleNet channel-split 不满足 16 通道对齐跑不了（§7）。

---

## 0. 包内容

```
model/    ppyoloe_s_640_ipu.param / .bin   模型（7.5MB，INT8，SPPF 优化版）
tools/    ppyoloe_run                      ★一体化推理工具（22KB RISC-V 可执行，源码 deploy/v861_ppyoloe_run/）
          libaw_simpleocv.so               官方图形库（画框/JPEG，固件里没带，需一起 push）
labels.txt                                 5 类：bird / squirrel / cat / person / other_animal
config.txt + ref/                          awnn_verify 对拍用（研发调试，见 §6；生产不用它）
```

**前置（每块新板只做一次）**：让同事用 `deploy/v861_firmware_base/` 烧固件基座
（PhoenixSuit + UBOOT 键）。烧完 `adb shell 'ls /dev/nna'` 存在 = NPU 就绪。**换模型永远不用重烧。**

---

## 1. 快速上手（生产集成就用这个）

```bash
# 布置（板子 /tmp 是 46MB RAM 盘，重启即空，需重推）
adb shell "mkdir -p /tmp/ppf/model /tmp/ppf/lib /tmp/ppf/res"
adb push model/ppyoloe_s_640_ipu.param model/ppyoloe_s_640_ipu.bin /tmp/ppf/model/
adb push tools/ppyoloe_run /tmp/ppf/ && adb push tools/libaw_simpleocv.so /tmp/ppf/lib/
adb shell "chmod +x /tmp/ppf/ppyoloe_run"

# 跑（JPG / 相机原生 NV21 都行；--no-draw = 生产形态只出 JSONL）
adb shell "cd /tmp/ppf && LD_LIBRARY_PATH=/tmp/ppf/lib:/usr/lib ./ppyoloe_run -m model -o res \
    --no-draw --nv21 1280x720 frame1.nv21 frame2.nv21"
adb shell "cat /tmp/ppf/res/results.jsonl"
```

**输出契约（后端只消费这个，无需写任何后处理代码）**——每帧一行 JSONL：

```json
{"image":"frame1.nv21","w":1280,"h":720,"infer_ms":43.0,"dets":[
  {"label":"bird","score":0.854,"box":[7.8,2.6,1265.2,711.6]}]}
```

- `box` = 原图像素坐标 `[x1,y1,x2,y2]`；`label` 见 labels.txt；conf 阈值默认 0.45（`-c` 可调）
- **空帧也必出一行**：无检出时 `"dets":[]`（实测验证）——不丢帧、不缺行
- 去掉 `--no-draw` 会额外输出 `<name>_det.jpg` 标框图（验证用，每帧 +110~220ms）

**常驻服务模式（相机循环推荐）**：`--stdin` —— 路径逐行喂 stdin，**JSONL 逐帧从 stdout 流出**
（诊断走 stderr），模型常驻，precompile 只付一次，之后每帧 43ms；EOF 退出。

```bash
# 后端进程管道对接示例：写路径进去、读 JSONL 出来
mkfifo /tmp/ppf/in.pipe
LD_LIBRARY_PATH=/tmp/ppf/lib:/usr/lib ./ppyoloe_run -m model -o res --no-draw --stdin     < /tmp/ppf/in.pipe > /tmp/ppf/out.jsonl.stream 2>/dev/null &
echo "--nv21 1280x720" > /tmp/ppf/in.pipe   # 之后每来一帧写一行路径
```

---

## 2. 权威指标（2026-07-21 实测，静态模式）

### 延迟（生产形态：NV21 1280×720 → JSONL）

| 阶段 | 耗时 |
|---|---|
| precompile（进程启动一次性） | 163 ms |
| NV21→RGB 转换+缩放（CPU，定点化） | 28.6 ms |
| **NPU 推理** | **43.0 ms**（恒定，132 帧三轮验证） |
| 解码+NMS（CPU） | 0.9 ms |
| **每帧总计** | **82.9 ms ≈ 12.1 fps** |

### 内存

| 方面 | 数值 |
|---|---|
| NPU 保留池占用 | **14.43 MB**（blob 7.23 + weight 7.20；池 20MB 余 5.6MB；**串行任意帧数恒定零增长**） |
| CPU 进程峰值 | VmHWM 25.8 MB（工具退出时自报） |
| 存储 | 模型 7.5MB + 工具 22KB + 库 399KB ≈ 8MB |

### 精度（板端 INT8 ≡ 上游 FP32，93 张同字节输入对拍）

框匹配 **94.9%** · 平均 IoU **0.9728** · 置信度差 **0.034** · cls 头余弦 **0.9993~0.9996** ⇒ 量化几乎无损。
精度权威数字 = round3 训练评估：**部署域（feeder）AP50 90.5**。

---

## 3. ★必须知道的四条（都是真板血泪，v2 有一条大反转）

1. **★生产必须用静态模式**（`ppyoloe_run` 已内置，`precompiler_enable=true`）。
   **v1 说"static 会死锁别用"是错的**——那是 NanoDet 时代的记录（栽的是模型不是模式）。
   官方文档明确：dynamic 是给动态 shape（OCR 类）的，每帧重建计算图；固定 shape 就该 static。
   实测：**43ms vs 226ms（快 5.3 倍）+ 内存恒定 + 串行任意帧数**。
   dynamic 模式下推第 2 帧就 `dma_mem_alloc fail`，硬砸会挂内核到 USB 消失（只能断电）。
2. **NPU 出错立即停，绝不重试**；进程 **segfault 崩溃会泄漏 NPU 池**（正常退出会还）→
   之后建实例全失败，`adb reboot` 软重启即可恢复（无需断电）。
3. **别用混合精度**（cls 头 FP32 → 回退 CPU +282ms）；**别降输入到 416**（整类漏检）。
   INT8 掉的 ~0.05 分用 conf 0.50→0.45 补偿（已是默认）。
4. **输出必须 FP32**（v2 实测论证）：输出张量改 INT8 后 cls 量化 scale=10.45（背景 logit 极端负值
   撑爆动态范围），分数只剩 0.5/1.0 两档、框吸附 8px 网格。这是量化物理，不是习惯。

---

## 4. I/O 契约（自己写集成时才需要；用 tools/ppyoloe_run 可跳过）

输入：**RGB** HWC uint8 0-255，**640×640**（`/255` 已折进 NPU ImageProcess 层，喂原始像素）。
输出：6 个 FP32 CHW blob（NPU 出裸 logits，后处理留 CPU）：

| blob | shape | 含义 |
|---|---|---|
| `conv2d_81.tmp_0` / `conv2d_84.tmp_0` | `[5,80,80]` / `[68,80,80]` | cls / reg-DFL，stride 8 |
| `conv2d_74.tmp_0` / `conv2d_77.tmp_0` | `[5,40,40]` / `[68,40,40]` | stride 16 |
| `conv2d_67.tmp_0` / `conv2d_70.tmp_0` | `[5,20,20]` / `[68,20,20]` | stride 32 |

reg 68 通道 = 4 边 × 17 DFL bin。解码参考实现：C 版 = `deploy/v861_ppyoloe_run/ppyoloe_run.c` 的
`decode()`；Python 版 = `../decode_ref.py`（numpy 自包含）。二者输出对齐。
**AWNN 不支持 YUV 输入**（`color_space` 仅 RGB 系）→ NV21 转换在 CPU 做（工具已内置定点化实现）。

---

## 5. 复现 / 重建

- **工具源码**：`deploy/v861_ppyoloe_run/`（`ppyoloe_run.c` + `build.sh`，编译机 openwrt musl rv32 工具链）
- **转换工程**：`../_build/round3/`（`awnn.sh build configs/config_ppyoloe_sppf.yml`，docker `awnn:1.0.2`）
- **裸 ONNX**：`_build/round3/onnx/ppyoloe_s_640_sppf_logits.onnx`（SPPF 改写 + 切后处理）

## 6. awnn_verify 对拍（研发调试用，生产不用）

`config.txt` + `ref/` 供官方 `awnn_verify` 逐层 profile / 与 FP32 对拍。
⚠ 对拍/逐层 profile **必须 `use_static_mode=0`**（官方 FAQ 要求 dynamic）——这与生产用 static 不矛盾，
是两个场景。批量对拍用 `deploy/v861_awnn_batch/`（带熔断）。

## 7. 为什么是 PP-YOLOE 不是 NanoDet

**VIP9000PICO 按 16 通道分块，channel-split 要求通道 ÷16**。NanoDet 的 ShuffleNetV2 半分出
58/116/232 通道 → 板端 `slice_ipu inplace requires channels divisible by 16`，跑不了。
PP-YOLOE 的 CSPRepResNet 无 channel-split → 一次通过。
（同理：**PicoDet 的 ESNet 预计同坑**；分类段 backbone 定 **MobileNetV2/RepVGG-A0**，
EfficientNet 在这颗 NPU 上慢 19 倍，见飞书候选清单。）

完整过程与 know-how：`docs/detect/05-V861-真板部署.md`（含 §7.5 指标表、§9 增量实验记录）。
