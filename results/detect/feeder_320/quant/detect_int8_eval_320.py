import os, numpy as np, onnx, onnxruntime as ort, torch
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config, load_model_weight, Logger
from onnx import version_converter
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

CFG = "outputs/detect/feeder_320.yml"
FP32 = "outputs/detect/feeder_320.onnx"
INT8 = "outputs/detect/feeder_320.int8.onnx"
PTH = "outputs/detect/feeder_320/model_best/nanodet_model_best.pth"
SAVE = "outputs/detect/int8_eval_320"
N_CALIB = 120
os.makedirs(SAVE, exist_ok=True)

load_config(cfg, CFG)
NUM_CLASSES = cfg.model.arch.head.num_classes
print("num_classes(from cfg)=%d  class_names=%s" % (NUM_CLASSES, list(cfg.class_names)), flush=True)

val_dataset = build_dataset(cfg.data.val, "test")
loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=naive_collate, drop_last=False,
)
print("val images = %d" % len(val_dataset), flush=True)

model = build_model(cfg.model)
_logger = Logger(-1, SAVE, False)
load_model_weight(model, torch.load(PTH, map_location="cpu"), _logger)
model.eval()
evaluator = build_evaluator(cfg.evaluator, val_dataset)


def prep(b):
    if isinstance(b["img"], list):
        b["img"] = stack_batch_img(b["img"], divisible=32)
    return b


# --- 关键修正：本 nanodet 版本 _forward_onnx 对类别通道做了 sigmoid，
# 而 get_bboxes 内部 scores = cls_preds.sigmoid() 会再 sigmoid 一次 -> 双重 sigmoid。
# post_process 期望 raw logits，因此把 ONNX 输出前 num_classes 通道反 sigmoid(logit) 还原。
EPS = 1e-7
def delogit_cls(preds):
    # preds: (B, 2125, 37) ; 前 NUM_CLASSES 通道是已 sigmoid 的概率
    p = np.clip(preds[..., :NUM_CLASSES], EPS, 1.0 - EPS)
    preds = preds.copy()
    preds[..., :NUM_CLASSES] = np.log(p / (1.0 - p))
    return preds


class CalibReader(CalibrationDataReader):
    def __init__(self, n=N_CALIB):
        self.data = []
        for i, b in enumerate(loader):
            if i >= n:
                break
            self.data.append({"data": prep(b)["img"].numpy().astype(np.float32)})
        self.it = iter(self.data)
        print("calib samples collected = %d" % len(self.data), flush=True)

    def get_next(self):
        return next(self.it, None)


# --- 量化：per-channel QDQ QInt8，opset<13 先升 13，失败回退 per-tensor ---
FP32Q = FP32
QUANT_MODE = None
try:
    m = onnx.load(FP32)
    if m.opset_import[0].version < 13:
        m = version_converter.convert_version(m, 13)
        FP32Q = FP32.replace(".onnx", "_op13.onnx")
        onnx.save(m, FP32Q)
    quantize_static(
        FP32Q, INT8, CalibReader(N_CALIB),
        quant_format=QuantFormat.QDQ, per_channel=True, weight_type=QuantType.QInt8,
    )
    QUANT_MODE = "per-channel(opset13)"
    print("量化模式: %s" % QUANT_MODE, flush=True)
except Exception as e:
    QUANT_MODE = "per-tensor(fallback:%s)" % type(e).__name__
    print("per-channel 失败(%s)->per-tensor" % type(e).__name__, flush=True)
    quantize_static(
        FP32, INT8, CalibReader(N_CALIB),
        quant_format=QuantFormat.QDQ, per_channel=False, weight_type=QuantType.QInt8,
    )

so = ort.SessionOptions()
so.intra_op_num_threads = os.cpu_count() or 8


def run_eval(onnx_path, tag):
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    results = {}
    for k, b in enumerate(loader):
        b = prep(b)
        preds = sess.run(None, {iname: b["img"].numpy().astype(np.float32)})[0]
        preds = delogit_cls(preds)  # 还原 raw logits，喂给 post_process
        results.update(model.head.post_process(torch.from_numpy(preds), b))
        if (k + 1) % 500 == 0:
            print("  [%s] %d/%d" % (tag, k + 1, len(val_dataset)), flush=True)
    res = evaluator.evaluate(results, SAVE)
    print("[%s] mAP=%.4f AP_50=%.4f" % (tag, res.get("mAP", -1), res.get("AP_50", -1)), flush=True)
    return res


# --- sanity：先用 1 张图核对 ONNX(delogit) 路径与 PyTorch 原生 eval 路径数值是否一致 ---
def sanity_one():
    b = prep(next(iter(loader)))
    inp = b["img"].numpy().astype(np.float32)
    # pytorch 原生 eval forward（raw logits 路径）
    with torch.no_grad():
        pt = model(torch.from_numpy(inp)).numpy()  # (1,2125,37) raw logits
    sess = ort.InferenceSession(FP32, sess_options=so, providers=["CPUExecutionProvider"])
    on = sess.run(None, {sess.get_inputs()[0].name: inp})[0]  # sigmoid 过的
    on_fixed = delogit_cls(on)
    cls_diff = np.abs(pt[..., :NUM_CLASSES] - on_fixed[..., :NUM_CLASSES]).max()
    reg_diff = np.abs(pt[..., NUM_CLASSES:] - on_fixed[..., NUM_CLASSES:]).max()
    print("SANITY cls_logit_maxdiff=%.5f reg_maxdiff=%.5f (pt-eval vs onnx-delogit)" % (cls_diff, reg_diff), flush=True)


sanity_one()

r32 = run_eval(FP32, "fp32_onnx")
r8 = run_eval(INT8, "int8_onnx")
print("RESULT quant_mode=%s calib=%d" % (QUANT_MODE, N_CALIB), flush=True)
print("RESULT fp32_mAP=%.4f int8_mAP=%.4f drop=%.4f" % (r32["mAP"], r8["mAP"], r32["mAP"] - r8["mAP"]), flush=True)
print("RESULT fp32_AP50=%.4f int8_AP50=%.4f drop=%.4f" % (
    r32.get("AP_50", -1), r8.get("AP_50", -1), r32.get("AP_50", -1) - r8.get("AP_50", -1)), flush=True)
print("=== DETECT INT8 EVAL DONE ===", flush=True)
