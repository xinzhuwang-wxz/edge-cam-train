"""地域 mask on/off 真测(修正 issue#11 artifact)。
全局口径(把非美国种真值压掉=artifact) + in-region 子集口径(真值是美国种=真实地域增益)。"""
import sys
import numpy as np
from PIL import Image
import onnxruntime as ort

sys.path.insert(0, "src")
from edge_cam.contracts.schemas.dataset import DatasetManifest
from edge_cam.eval.regional import RegionalMask

MANIFEST = "data/processed/birds525/manifest.json"
ONNX = "/tmp/efficientnet_lite0_fp32.onnx"
REGION = "regions/us.json"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

m = DatasetManifest.load(MANIFEST)
c2i = m.class_to_idx
taxon_of = {r.label: r.taxon_key for r in m.records if r.taxon_key}
mask = RegionalMask.from_json(REGION, c2i, taxon_of)
allowed = mask.allowed_idx
keep = np.zeros(len(c2i), bool)
keep[list(allowed)] = True
print("地域覆盖: %d/%d 类 = %.1f%%" % (len(allowed), len(c2i), 100 * len(allowed) / len(c2i)), flush=True)

test = [r for r in m.records if r.split == "test"]
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name


def load(r):
    im = Image.open(m.resolve_path(r)).convert("RGB").resize((224, 224), Image.BILINEAR)
    x = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
    return np.transpose(x, (2, 0, 1))[None].astype(np.float32)


g_off = g_on = ir_off = ir_on = ir_n = n = 0
for r in test:
    gt = c2i[r.label]
    logits = sess.run(None, {inp: load(r)})[0][0]
    n += 1
    pred_off = int(np.argmax(logits))
    g_off += pred_off == gt
    masked = logits.copy()
    masked[~keep] = -1e30
    pred_on = int(np.argmax(masked))
    g_on += pred_on == gt
    if gt in allowed:
        ir_n += 1
        ir_off += pred_off == gt
        ir_on += pred_on == gt
    if n % 500 == 0:
        print("  %d/%d" % (n, len(test)), flush=True)

print("\n=== 全局 525 类 test (n=%d) ===" % n, flush=True)
print("mask OFF top1: %.4f" % (g_off / n), flush=True)
print("mask ON  top1: %.4f  <- artifact(非美国种真值被压掉,issue#11,非真增益)" % (g_on / n), flush=True)
print("\n=== in-region 子集(真值是美国种, n=%d)—— 真实地域增益 ===" % ir_n, flush=True)
print("mask OFF top1: %.4f" % (ir_off / ir_n), flush=True)
print("mask ON  top1: %.4f" % (ir_on / ir_n), flush=True)
print("地域增益: %+.4f (%+.1fpt)" % ((ir_on - ir_off) / ir_n, 100 * (ir_on - ir_off) / ir_n), flush=True)
print("=== REGIONAL EVAL DONE ===", flush=True)
