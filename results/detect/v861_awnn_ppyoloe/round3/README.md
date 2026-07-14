# PP-YOLOE-s @ V861 NPU 板端交接包（round3）

> **✅ 已在真板跑通并对拍验证**（2026-07-14）。这是目前**唯一能在 V861 NPU 上跑的检测模型**——nanodet 因 ShuffleNet 的 channel-split 不满足 NPU 16 通道对齐，跑不了（见文末）。

---

## 0. 前置：先烧一次固件基座（只烧一次，跟模型无关）

**换模型不用重烧固件。** 新板子先让同事烧一次通用基座：
👉 **`deploy/v861_firmware_base/`**（固件 `.img` + PhoenixSuit 烧录指引）

烧完 `adb shell 'ls /dev/nna'` 存在 = NPU 就绪，之后所有模型都只是 `adb push` 文件。

---

## 1. 板端 I/O 契约（⚠ 与 nanodet 不同，别搞混）

| | PP-YOLOE（本包）| nanodet（旧）|
|---|---|---|
| 输入 | **RGB** HWC uint8 **0-255**, **640×640** | BGR, 416×416 |
| 归一化 | `/255` **已折进 NPU**（ImageProcess 层），板端喂原始像素 | mean/std 折进 NPU |
| 输出 | **6 个 FP32 CHW blob**（见下）| 单个 logits |

**6 个输出**（NPU 出裸 logits，后处理留 CPU）：

| blob | shape | 含义 |
|---|---|---|
| `conv2d_81.tmp_0` | `[5, 80, 80]` | cls logits, stride 8 |
| `conv2d_84.tmp_0` | `[68, 80, 80]` | reg DFL logits, stride 8 |
| `conv2d_74.tmp_0` | `[5, 40, 40]` | cls, stride 16 |
| `conv2d_77.tmp_0` | `[68, 40, 40]` | reg, stride 16 |
| `conv2d_67.tmp_0` | `[5, 20, 20]` | cls, stride 32 |
| `conv2d_70.tmp_0` | `[68, 20, 20]` | reg, stride 32 |

`reg` 的 68 通道 = **4 边 × 17 个 DFL bin**（reg_max=16）。类别顺序见 `labels.txt`：`bird / squirrel / cat / person / other_animal`。

**CPU 后处理** = `../decode_ref.py`（自包含，只依赖 numpy）：sigmoid(cls) + DFL(softmax+积分) + anchor 解码 + 逐类 NMS。

---

## 2. 上板怎么跑

```bash
# 1) 推模型 + 输入 + config（板子已烧好基座固件）
adb push round3/ /tmp/ppyoloe

# 2) 推板端验证工具 awnn_verify（在编译机上单独交叉编译, 见 §5）
adb push awnn_verify /tmp/ppyoloe/awnn_verify
adb shell 'chmod +x /tmp/ppyoloe/awnn_verify'

# 3) 跑
adb shell 'cd /tmp/ppyoloe && ./awnn_verify config.txt'
```

### ⚠⚠ 必须用动态模式 `use_static_mode=0`
`config.txt` 里已设好。**`use_static_mode=1`（静态预编译）会死锁 NPU 硬件、把板子搞挂**（软重启救不回，要冷断电）——血的教训，别改。

---

## 3. 真板实测（2026-07-14，V861M3 PER2）—— 本包 = 已优化的 SPPF 版

```
AWNN Test Pass        EXIT=0
```

| 指标 | 实测 |
|---|---|
| **层落点** | **152 层全部在 NPU，零 CPU 回退** ✅ |
| **推理耗时** | **240 ms** @640×640（≈4.2 fps）|
| **NPU 内存** | **18.02 MB**（blob 10.6 + weight 7.2）—— 板子 NPU 预留池 20MB |
| 模型 | PP-YOLOE-s 640 + **SPPF 改写**，全 INT8，21 张校准 |

### 精度（板端 INT8 vs FP32 ONNX 真值，demo_bird.jpg，**conf=0.45**）

| | FP32 ONNX | 板端 NPU INT8 |
|---|---|---|
| bird #1 | 0.800 `[4.4, 2.3, 633.9, 625.8]` | **0.782 `[5.4, 1.5, 630.1, 626.9]`** ✓ |
| bird #2 | 0.600 `[147.6, 503.5, 207.5, 557.4]` | **0.566 `[144.4, 503.3, 208.0, 559.3]`** ✓ |
| bird #3 | 0.508 `[293.1, 139.6, 353.7, 211.7]` | **0.456 `[289.0, 139.2, 353.3, 211.9]`** ✓ |

**3/3 全中，框误差仅几像素。** INT8 分数系统性低约 0.05 → **`decode_ref.py` 默认 conf 已设 0.45 补偿**（免费，别用混合精度，见下）。

---

## 4. ★三条实测出来的优化结论（都经过真板验证，别踩坑）

### ✅ 优化1：SPP → SPPF（白赚 24% 速度，零精度损失）
NPU 只支持 **5×5 MaxPool**；SPP 的 **9×9/13×13 会回退 CPU**（吃掉 44ms）。
**把 9×9 换成 2 串 5×5、13×13 换成 3 串 5×5**——stride=1 same-pad 下**数学完全等价**（实测输出误差 **0.000e+00，逐位相同**），但全部能上 NPU。
→ **315ms → 240ms，CPU 回退 3→0。不用重训。**（脚本见 `_build/round3/`）

### ❌ 优化2：**千万别用混合精度（cls 头 FP32）**——真板上是灾难
**V861 的 NPU 是 INT8-only（不支持 FP32）**。把 cls 头设成 FP32 → **NPU 跑不了，强制回退 CPU**，而 80×80 的 head 卷积在 RISC-V 上奇慢：
```
CPU Layer Conv.81  134.5 ms
CPU Layer Conv.74   96.2 ms
CPU Layer Conv.67   51.5 ms   ← 合计 +282ms!
总耗时 240ms → 545ms（慢一倍多）
```
> ⚠ **离线模拟器会骗你**：nanodet 那轮"cls头保FP32"的结论是在**模拟器**里得的（模拟器 FP32 免费），**真板上完全相反**。这就是必须上板验证的原因。
> **INT8 掉分改用「降低置信阈值」补偿**（0.50→0.45），免费且有效（3/3 全中）。

### ❌ 优化3：**别降输入尺寸到 416**（模型是 640 训的，会严重掉精度）
实测 416（FP32 ONNX，排除量化因素）：
| 图 | 640 | 416 |
|---|---|---|
| other_animal | 0.90 / 0.79 | **完全漏检** |
| person | 0.72 | **完全漏检** |
| cat | 0.82 | 0.67 + **误检 other_animal** |
| bird | 3 个检出 | 只剩 1 个 |

→ **不重训就别降尺寸。**

### 💡 级联内存怎么办（检测 18MB / NPU 池 20MB，分类器塞不下）
**NPU 预留池 20MB 是我们自己在固件 DTS 里定的**（`size_pool_mem`）——**固件是我们编的，直接调大即可**（板子 128MB，Linux 现用 95MB，可让出一部分）。**不用牺牲模型精度换内存。**

### 其它
- 校准集目前 21 张（本机可得）；用完整校准集重转可能再提升一点。

## 5. 复现 / 重建

- **转换工程**：`../_build/round3/`（`configs/config_ppyoloe.yml` + `awnn.sh` + `calib/`）。
  `cd _build/round3 && ./awnn.sh build configs/config_ppyoloe.yml` → 出 `_ipu`（需 docker `awnn:1.0.2`）。
- **裸 ONNX**：`_build/round3/onnx/ppyoloe_s_640_logits.onnx`（从 `ppyoloe_s_640.onnx` 用 `onnx.utils.extract_model` 切掉后处理，只留 `image` 输入 + 6 个 head 输出）。
- **awnn_verify**：编译机 `/home/yechen/build_awnn_verify.sh`（openwrt musl rv32 工具链）。
  ⚠ 已打补丁修 vendor 的 **FP32 dump bug**（原代码把"元素数"当"字节数"写，FP32 输出被截断到 1/4）。

## 6. ★为什么是 PP-YOLOE 不是 nanodet

**V861 的 VIP9000PICO NPU 按 16 通道分块，`slice`(channel-split) 要求通道 ÷16。**
- **nanodet 的 ShuffleNetV2** channel split 半分出 **58/116/232** 通道——全不是 16 倍数 → 板端报 `slice_ipu inplace requires the number of channels to be divisible by 16`，**静态/动态都跑不了**。
- **PP-YOLOE 的 CSPRepResNet** 全是标准 Conv+Concat，**backbone 里没有 channel-split**（ONNX 里那 2 个 Split 在后处理，已被切掉）→ **一次通过**。

**★项目级铁律：上 V861 NPU 的模型必须 16 通道对齐、避开 channel-split。** 分类段的 EfficientNet-Lite（MBConv/SE，无 split）应天然安全。

完整上板实录见 `vs861/V861真板部署实录.md`。
