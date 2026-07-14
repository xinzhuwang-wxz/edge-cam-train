# V861 通用固件基座（PER2 板 · NPU-ready）

> **这是板子的"操作系统"，跟具体模型无关。每块新板烧一次即可，之后我们的模型全部 `adb push` 直推、不用再烧。**
> 详细上板实录见 `vs861/V861真板部署实录.md`。

---

## 这是什么 / 为什么单独放这

- `v861m3-xxx_linux_perf2_uart0_nor.img`（15.4MB，md5 `2705249bc507f776fc4be9fd02cf9f37`）
  = V861 **PER2 板 · V861M3(128MB)** 的完整 Tina Linux 固件，**内含 NPU 栈**：
  - NPU 内核驱动 `aw_nna_sunxi`（开机创建 **`/dev/nna`**，AWNN driver 1.0.0.2，864MHz）
  - 运行时库 `/usr/lib/libawnn.so`(1.5.1) + `libawipubsp.so`
  - ADB（免密 root）
- **固件是通用基座**：一次烧录，之后 nanodet / ppyoloe / 分类等模型都只是 `_ipu` 文件，`adb push` 到 `/tmp` 或 `/mnt/UDISK` 直接跑。**换模型 ≠ 重烧固件。**

> `.img` 是构建产物（可重建，见下），按项目惯例不强推荐进 git；给同事时直接发这个文件。

---

## 给同事的烧录指引（Windows PhoenixSuit）

1. Windows PC 打开 **PhoenixSuit** → **一键刷机** → **浏览** 选 `v861m3-xxx_linux_perf2_uart0_nor.img` → 模式 **全盘擦除升级**。
2. **断开板子 DC 电源** → **按住板子丝印 `UBOOT` 键**（8 键组左上角；不是 RESET / XR806-FEL）不放 → MicroUSB(JU1) 插电脑 → **出进度条后松开 UBOOT 键** → 等走完 = 成功。中途勿拔线/断电。
3. 烧完把 USB 插回操作机，`adb shell` 里 `ls /dev/nna` 应存在 = NPU 就绪。

## 烧完之后（我们自己）——不用再烧

```bash
adb push <model>_ipu.param <model>_ipu.bin /tmp/xxx/      # 换模型只推文件
adb push awnn_verify /tmp/xxx/                            # 板端验证工具(单独交叉编, 见下)
adb shell 'cd /tmp/xxx && ./awnn_verify config.txt'      # 跑; 建议 use_static_mode=0(动态)
```
- ⚠**教训**：`use_static_mode=1`(静态预编译) 曾在 ppyoloe-640 上**死锁板子**（需物理重启）。**优先用 `use_static_mode=0` 动态模式**。
- NPU 设备节点是 **`/dev/nna`**（不是 /dev/galcore）。

## 怎么重建这个固件（编译机 `<编译机IP>`，全 Linux）

见 `vs861/V861真板部署实录.md §7 构建配方`。要点：`lunch v861-perf2`（答 V861 / M3-XXX / 0.92V / **Y免责声明**）→ `quick_config config_awnn_runtime -f`（开 NPU）→ host 需 `gawk`（`sudo apt install gawk`）→ `make -j$(nproc)` → `pack` → 产物 `out/v861m3-xxx_linux_perf2_uart0_nor.img`。脚本：编译机 `/home/<用户>/aw_build_npu.sh`。
