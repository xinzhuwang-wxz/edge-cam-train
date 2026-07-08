"""板端交接包全面测试（无板下能做的完整验收）。

验证：① 结构完整 ② labels ③ config.txt 字段自洽 ④ 参考输出 decode 出正确 bird 检测
     ⑤ 交接包 INT8 参考 ≈ FP32 onnxruntime（同预处理输入）⑥ ref 输入/输出尺寸自洽。
run: python test_board_handoff.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

V = Path(__file__).resolve().parent  # .../v861_awnn_nanodet/_build/round2
BH = V.parents[1] / "round2"  # 交付物在 _build/round2/ 的上两级 → v861_awnn_nanodet/round2
sys.path.insert(0, str(V.parents[4] / "src"))  # 仓库根/src
from edge_cam.cascade.adapters import decode_nanodet  # noqa: E402

fails, checks = [], []


def ok(name: str, cond: bool, detail: str = ""):
    checks.append(name)
    if not cond:
        fails.append(f"{name}: {detail}")
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  '+detail if detail else ''}")


print("== ① 结构完整 ==")
need = [
    "model/nanodet_feeder5_v861_416_ipu.param",
    "model/nanodet_feeder5_v861_416_ipu.bin",
    "config.txt",
    "labels.txt",
    "README.md",
    "ref/demo_bird_input_int8.bin",
    "ref/demo_bird_output_fp32.bin",
    "ref/demo_bird.jpg",
]
for f in need:
    ok(f, (BH / f).exists())
ok("_ipu.bin 体积合理(<3MB, 克制)", (BH / "model/nanodet_feeder5_v861_416_ipu.bin").stat().st_size < 3e6,
   f"{(BH / 'model/nanodet_feeder5_v861_416_ipu.bin').stat().st_size/1e6:.2f}MB")

print("== ② labels ==")
labels = (BH / "labels.txt").read_text().split()
ok("5 类且 bird 为 id0", labels[:5] == ["bird", "squirrel", "cat", "person", "other_animal"], str(labels[:5]))

print("== ③ config.txt 字段自洽 ==")
cfg = dict(
    ln.split("=", 1) for ln in (BH / "config.txt").read_text().splitlines() if "=" in ln and not ln.startswith("#")
)
ok("input blob=data", cfg.get("input_blob_names") == "data")
ok("output blob=output", cfg.get("output_blob_names") == "output")
ok("416x416x3", (cfg.get("inputs_w"), cfg.get("inputs_h"), cfg.get("inputs_c")) == ("416", "416", "3"))
ok("output dtype=FP32", cfg.get("output_data_type") == "DATA_TYPE_FP32")
ok("model_path 存在", (BH / cfg.get("model_path", "x")).exists(), cfg.get("model_path", ""))
ok("input_paths 存在", (BH / cfg.get("input_paths", "x")).exists())
ok("output_paths 存在", (BH / cfg.get("output_paths", "x")).exists())

print("== ④ 参考输出 decode → bird 检测 ==")
out = np.fromfile(BH / "ref/demo_bird_output_fp32.bin", np.float32).reshape(1, 3598, 37)
w, h = Image.open(BH / "ref/demo_bird.jpg").size
dets = decode_nanodet(out, (w, h), num_classes=5, conf_thr=0.4, nms_iou=0.5)
ok("至少 1 个检测", len(dets) >= 1, f"{len(dets)} 个")
if dets:
    top = max(dets, key=lambda d: d.score)
    ok("最高分是 bird", top.class_id == 0, f"class={labels[top.class_id]} score={top.score:.3f}")
    ok("bird 分数 > 0.5", top.score > 0.5, f"{top.score:.3f}")
    # 契约：decode 出裸框，画框时由 app 裁到图内 → 裁后判非退化 + 覆盖合理
    x1, y1, x2, y2 = top.box
    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    ok("裁后框非退化", cx2 > cx1 and cy2 > cy1, f"clip=({cx1:.0f},{cy1:.0f},{cx2:.0f},{cy2:.0f}) img={w}x{h}")
    ok("框覆盖合理(1%~100%图面积)", 0.01 <= (cx2 - cx1) * (cy2 - cy1) / (w * h) <= 1.0)

print("== ⑤ 交接包 INT8 ≈ FP32(同预处理输入) ==")
dump = V / "build_out/logits_final/results/data_demo_bird.jpg"
if not (dump / "data_fp32.bin").exists() or not (V / "onnx/main_416_fp32_logits.onnx").exists():
    print("  [SKIP] 需 build_out dump + onnx(gitignore, 由 awnn.sh 重建)；见 docs/detect/04 §4.1")
    print(f"\n{'='*50}\n{len(checks)-len(fails)}/{len(checks)} 通过", "✅ 交接包验收通过" if not fails else f"❌ {fails}")
    sys.exit(1 if fails else 0)
inp = np.fromfile(dump / "data_fp32.bin", np.float32).reshape(1, 3, 416, 416)
sess = ort.InferenceSession(str(V / "onnx/main_416_fp32_logits.onnx"), providers=["CPUExecutionProvider"])
fp_out = sess.run(None, {sess.get_inputs()[0].name: inp})[0].reshape(1, 3598, 37)
fp_dets = decode_nanodet(fp_out, (w, h), num_classes=5, conf_thr=0.4, nms_iou=0.5)
ok("FP32 也检出 bird", any(d.class_id == 0 for d in fp_dets), f"{len(fp_dets)} 检测")
if dets and fp_dets:
    ft = max((d for d in fp_dets if d.class_id == 0), key=lambda d: d.score)
    a, b = top.box, ft.box
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    iou = inter / ua if ua > 0 else 0
    ok("INT8-vs-FP32 bird 框 IoU>0.9", iou > 0.9, f"IoU={iou:.3f} 分差={abs(top.score-ft.score):.3f}")

print(f"\n{'='*50}\n{len(checks)-len(fails)}/{len(checks)} 通过", "✅ 交接包验收通过" if not fails else f"❌ {fails}")
sys.exit(1 if fails else 0)
