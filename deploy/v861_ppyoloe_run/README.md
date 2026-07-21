# ppyoloe_run —— V861 板端一体化推理工具（板端 C 集成落地）

> 2026-07-21 真板实测通过（30 帧混合串行长跑 30/30，温度 <37°C）。
> 一个 22KB 的 RISC-V 可执行 + 官方 `libaw_simpleocv.so`，不依赖 Python/仓库。

```
输入   JPG/PNG/BMP（awcv_imread）或 raw NV21（--nv21 WxH，相机原生格式）
推理   AWNN 静态模式（precompiler）—— 实例常驻，43ms/帧，NPU 内存恒定 14.43MB
解码   CPU：sigmoid + DFL + anchor + 逐类 NMS（与 decode_ref.py 对齐）
输出   <out>/results.jsonl（一行一图）+ <out>/<name>_det.jpg（标框图，--no-draw 可关）
```

## 用法

```bash
# 板上（/tmp/ppf 布置好 model/ 和 lib/ 后）
LD_LIBRARY_PATH=/tmp/ppf/lib:/usr/lib ./ppyoloe_run -m model -o res \
    img1.jpg img2.jpg \
    --nv21 1280x720 frame1.nv21 frame2.nv21     # --nv21 可中途切换尺寸

# 选项: -c conf(0.45)  -n nms(0.50)  --no-draw(不出标框图)
```

JSONL 每行：
```json
{"image":"frame1.nv21","w":1280,"h":720,"infer_ms":43.1,"dets":[{"label":"bird","score":0.866,"box":[8.0,2.6,1263.9,708.2]}]}
```

## 实测性能（V861 真板，PP-YOLOE-s 640²）

| 路径 | 每帧总耗时 | 分解 |
|---|---|---|
| **NV21 1280×720，--no-draw（生产形态）** | **82.9 ms ≈ 12.1 fps** | NV21 定点转换 28.6 + NPU 43.0 + 解码 0.9 |
| NV21 640×640，--no-draw | 80.2 ms | |
| JPG + 标框图输出 | 210~360 ms | JPG 解码 + 全幅画框 + imwrite 是大头 |
| precompile（一次性） | 163 ms | 进程启动时一次 |

NPU 内存恒定 **14.43 MB**（blob 7.23 + weight 7.20），30 帧长跑零增长。

## 构建 / 部署

```bash
bash build.sh                # 编译机上；工具链=固件构建产物(openwrt musl rv32)
bash provision（参考）:      # 板子重启后 /tmp(RAM盘)清空，需重新布置：
adb shell "mkdir -p /tmp/ppf/model /tmp/ppf/lib /tmp/ppf/res"
adb push ppyoloe_s_640_ipu.{param,bin} /tmp/ppf/model/
adb push ppyoloe_run /tmp/ppf/ && adb push libaw_simpleocv.so /tmp/ppf/lib/
# 编译机上有现成一键脚本 /home/<用户>/provision_ppf.sh
```

## 实现要点（都踩过坑）

1. **必须静态模式**（`precompiler_enable=true`）：dynamic 模式每帧重建计算图（+180ms）
   且不还 IPU 内存——第 2 帧 `dma_mem_alloc fail`。静态模式 43ms/帧、内存恒定。
   官方文档：dynamic 是给动态 shape（OCR 类）用的；固定 shape 就该 static。
2. **NPU 出错立即 break，绝不重试**——硬砸会把内核挂死到 USB 消失，只能物理断电。
3. **进程崩溃（segfault）会泄漏 NPU carveout**（正常退出会归还）→ 后续任何进程
   `instance_create` 都失败。恢复：`adb reboot` 软重启即可（无需断电）。
4. **simpleocv 是 ncnn 风格**：`CV_8UC3 = 3`（通道数），不是 OpenCV 的 16——传 16 段错误。
   `awcv_imread` 底图为 BGR；画框/写字/imwrite(JPEG_QUALITY) 都可用。
5. **AWNN 不支持 YUV 输入**（转换工具链 `color_space` 只有 RGB/BGR/RGBA/BGRA/GRAY）→
   NV21→RGB 在 CPU 做。本工具用**融合单遍**（NV21 直接采样出 640²RGB，Y 双线性 + UV 最近邻，
   BT.601 定点）：比"全幅转换再缩放"省 ~50ms/帧。全幅转换只在需要画框时做。
6. 输出缓冲一次分配常驻复用；解码先用 logit 阈值筛 anchor，过筛才算 DFL（softmax 17 bin）。

## 空帧语义（后端对接要点）

**每帧必出一行 JSONL**，无检出时 `"dets":[]`（实测 neg_bg.jpg 验证）——不会丢帧、不会缺行。
后端只需解析 JSONL：`label` / `score` / `box[x1,y1,x2,y2]`（原图像素坐标），**无需写任何后处理代码**。

## 与交接包的关系

- 模型/校准/对拍参考：`results/detect/v861_awnn_ppyoloe/round3/`（那边 `config.txt`
  的 `use_static_mode=0` 是给 awnn_verify **对拍/逐层 profile** 用的，官方 FAQ 要求 dynamic；
  **生产集成用本工具的静态模式**）。
- 批量对拍工具（研发用）：`deploy/v861_awnn_batch/`。
