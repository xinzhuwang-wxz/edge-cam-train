# NanoDet · 移动端交接包（ONNX + TFLite + 续训权重）· 按轮组织

NanoDet 检测器（ShuffleNetV2 1.0x, 416, 5 类）的移动端交接包。**同类型 = 同接口**，`nanodet_detect.py` 对所有轮通用；**按 round 分层**，取哪轮进哪个子目录。

```
mobile_handoff_nanodet/
├── nanodet_detect.py   ← decode+NMS 参考（onnx/tflite 通用，喂 BGR 0-255 resize 416）
├── labels.txt          ← 5 类名（行号=id，与 round2 同序）
├── round2/             ← round2 最佳 feeder 模型
│   ├── onnx/feeder_416.onnx
│   └── tflite/feeder_416.{fp16,fp32}.tflite
└── round3/
    ├── p2/  ← ★最佳 NanoDet 部署（P0+数据+crop，feeder AP50 89.3、V861 零回退）
    │   ├── onnx/p2_416.onnx
    │   ├── tflite/p2_416.{fp16,fp32}.tflite
    │   └── ckpt/model_best.ckpt   （续训用；gitignore，80M）
    └── p0/  ← 纯数据基线（feeder 87.7）· 同样 onnx/ + tflite/ + ckpt/
```

| 项 | 说明 |
|---|---|
| `*/onnx/*.onnx` | sigmoid+归一化焊进图，输出 [3598,37] 概率 |
| `*/tflite/*.{fp16,fp32}.tflite` | onnxsim→onnx2tf，**已验 ≡ onnx，max\|Δ\|~2e-5、cosine 1.0** |
| `round3/p{0,2}/ckpt/model_best.ckpt` | 续训用（gitignore，80M）|

**接口**（与 round2 一致）：喂任意尺寸 BGR 0-255 → resize 416 → 输出 [3598,37] 概率 → `nanodet_detect.py` 的 DFL decode + NMS → `{label,score,box}`。默认 conf=0.4/nms=0.5（阈值 CPU 侧可调）。

**续训**（nanodet env）：`python tools/train.py --config <p2 config> --resume <ckpt>`。

导出坑（供参考）：onnx2tf 直转 ShuffleNet channel-shuffle 会报 conv 0 维 → **必须先 onnxsim 简化**（round2 同管线）；且 onnx2tf 下载校准图 np.load 坏 → 塞个 dummy npy 绕过。
