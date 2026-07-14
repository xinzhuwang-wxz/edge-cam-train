# awnn_batch —— V861 板端批量推理工具

官方 `awnn_verify` 只能**一次一张**。想批量跑（评测/对拍/回归）时，如果用 shell 循环反复 exec 它，
**NPU 的 DMA carveout 不会归还** → 第 2 次就 `DMA_HEAP_IOCTL_ALLOC failed` → 继续硬砸会**把内核挂死**
（adb offline，需物理断电）。详见 `../V861真板部署实录.md` §10.5。

`AWNN_batch.c` = 官方 `AWNN_verify.c` + 3 个补丁：

1. **批量列表**：新增 config 键 `batch_list=`（每行一个输入 `.bin` 路径）、`out_dir=`。
   实例只 `awnn_instance_create` 一次，循环 `set_in_tensors → inference → get_out_tensors`。
2. **每图打印** `npu_ms`，收尾报 min/max/avg。
3. **修 FP32 dump bug**：官方 `fwrite(data, 1, size, fp)` 把元素数当字节数 → FP32 被截成 1/4。

## 编译（编译机上）

```bash
O=/home/<用户>/tina-v861/out/v861/perf2/openwrt
R=$O/build_dir/target/awnn/awnn_sdk/awnn_runtime
$O/staging_dir/toolchain/bin/riscv32-unknown-linux-musl-gcc AWNN_batch.c -o awnn_batch \
  -I$R/libawnn/include -I$R/libaw_utilities/include -I$R/libawipubsp/include \
  -L$O/staging_dir/target/usr/lib -lawnn -lawipubsp -lm -O2
```
→ ELF32 RISC-V musl，~22KB。

## 跑（`run_board.sh`）

板子 `/tmp` 是 **46MB RAM 盘**，装不下大批输入 → 脚本**分块**（默认 6 张/块）推-跑-拉-清，
并**带熔断**：任一块出现 `DMA_HEAP|invalid|inference error` 立刻停，绝不重试。

## 三条铁律

- **一进程一张**：dynamic mode 下每次 `inference` 都新分配 IPU blob 内存且不释放
  （1 张 = 18.02MB，2 张 = **22.71MB** → 顶破 20MB 保留池）。进程退出才归还。
  `run_board.sh` 因此把 `batch_list` 每次只填 1 行，外层循环起进程。
- **看到 `dma_mem_alloc fail` / `DMA_HEAP_IOCTL_ALLOC failed` 立即停手**，别重试 ——
  硬砸会把内核写坏，板子挂死到 USB 都不枚举，只能物理断电。
- **`use_awnn_profiler=0`**（部署态）—— 实测开着**多花 ~24ms**（244.4 vs 220.6ms），
  但**不多占 NPU 内存**（两边都是 18.0175MB）。profiler 只在单张调试时开。
