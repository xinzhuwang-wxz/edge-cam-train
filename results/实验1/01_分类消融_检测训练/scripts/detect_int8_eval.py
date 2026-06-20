"""检测 int8 掉点评估(800 图子集,快):量化 FP32 ONNX -> fp32/int8 各跑 COCOeval 对比 mAP。
复用 NanoDet val dataset + head.post_process + CocoEvaluator。ORT-QDQ 仅方向性(真值来自板子 ACUITY)。"""

import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config
from onnx import version_converter
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

CFG = "outputs/detect/nanodet_feeder_416_sub.yml"  # 800 图子集
FP32 = "outputs/detect/nanodet_feeder_416.onnx"
INT8 = "outputs/detect/nanodet_feeder_416.int8.onnx"
SAVE = "outputs/detect/int8_eval"
os.makedirs(SAVE, exist_ok=True)

load_config(cfg, CFG)
val_dataset = build_dataset(cfg.data.val, "test")
loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=naive_collate,
    drop_last=False,
)
model = build_model(cfg.model)
model.eval()
evaluator = build_evaluator(cfg.evaluator, val_dataset)
print("val 子集图数:", len(val_dataset), flush=True)


def prep(b):
    if isinstance(b["img"], list):
        b["img"] = stack_batch_img(b["img"], divisible=32)
    return b


class CalibReader(CalibrationDataReader):
    def __init__(self, n=120):
        self.data = []
        for i, b in enumerate(loader):
            if i >= n:
                break
            self.data.append({"data": prep(b)["img"].numpy().astype(np.float32)})
        self.it = iter(self.data)

    def get_next(self):
        return next(self.it, None)


print("=== 量化 fp32->int8 (per-channel, opset13, calib=120) ===", flush=True)
FP32Q = FP32
try:
    m = onnx.load(FP32)
    ver = m.opset_import[0].version
    if ver < 13:
        m = version_converter.convert_version(m, 13)
        FP32Q = FP32.replace(".onnx", "_op13.onnx")
        onnx.save(m, FP32Q)
        print("opset %d -> 13" % ver, flush=True)
    quantize_static(
        FP32Q,
        INT8,
        CalibReader(120),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
    )
    print("量化模式: per-channel(opset13)", flush=True)
except Exception as e:
    print("per-channel 失败(%s)->回退 per-tensor" % type(e).__name__, flush=True)
    quantize_static(
        FP32,
        INT8,
        CalibReader(120),
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QInt8,
    )
    print("量化模式: per-tensor(fallback)", flush=True)
print("int8 onnx:", os.path.getsize(INT8) // 1024, "KB", flush=True)

so = ort.SessionOptions()
so.intra_op_num_threads = os.cpu_count() or 8


def run_eval(onnx_path, tag):
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    results = {}
    for k, b in enumerate(loader):
        b = prep(b)
        preds = sess.run(None, {iname: b["img"].numpy().astype(np.float32)})[0]
        results.update(model.head.post_process(torch.from_numpy(preds), b))
        if (k + 1) % 200 == 0:
            print("  [%s] %d/%d" % (tag, k + 1, len(val_dataset)), flush=True)
    res = evaluator.evaluate(results, SAVE)
    print("[%s] mAP=%.4f AP_50=%.4f" % (tag, res.get("mAP", -1), res.get("AP_50", -1)), flush=True)
    return res


print("\n=== FP32 ONNX COCOeval ===", flush=True)
r32 = run_eval(FP32, "fp32_onnx")
print("\n=== INT8 ONNX COCOeval ===", flush=True)
r8 = run_eval(INT8, "int8_onnx")
print("\n############ 检测量化对比(800图子集) ############", flush=True)
print(
    "fp32_onnx mAP=%.4f | int8_onnx mAP=%.4f | int8掉点=%.4f"
    % (r32["mAP"], r8["mAP"], r32["mAP"] - r8["mAP"]),
    flush=True,
)
print("=== DETECT INT8 EVAL DONE ===", flush=True)
