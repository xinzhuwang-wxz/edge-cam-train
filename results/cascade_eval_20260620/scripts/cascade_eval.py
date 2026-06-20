"""级联端到端评估:检测(NanoDet ONNX)-> 取 bird 框裁剪 -> 分类(eff_lite0 ONNX) -> species top-1。
在 birds525 test 子集(有 species GT)上跑,对比级联 top-1 vs 分类器直评(0.921),含 fp32/int8 分类器。
检测在 NanoDet env 用 head.post_process 解码(原图坐标)。"""
import os, json, numpy as np, torch, cv2
import onnxruntime as ort
from PIL import Image
from nanodet.data.collate import naive_collate
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.dataset import build_dataset
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config

CFG = "outputs/detect/nanodet_cascade.yml"
DET = "outputs/detect/nanodet_feeder_416.onnx"
CLF32 = "/root/autodl-tmp/efficientnet_lite0_fp32.onnx"
CLF8 = "/root/autodl-tmp/eff_lite0_int8.onnx"
IMGDIR = "/root/autodl-tmp/cascade_data/images"
BIRD_CLASS = 0
THRESH = 0.3
IM_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IM_STD = np.array([0.229, 0.224, 0.225], np.float32)

gt = json.load(open("/root/autodl-tmp/cascade_data/gt_species.json"))
lab = json.load(open("/root/autodl-tmp/cascade_data/labels.json"))
id2file = {im["id"]: im["file_name"] for im in lab["images"]}

load_config(cfg, CFG)
ds = build_dataset(cfg.data.val, "test")
loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=naive_collate)
model = build_model(cfg.model); model.eval()

det = ort.InferenceSession(DET, providers=["CPUExecutionProvider"]); det_in = det.get_inputs()[0].name
clf32 = ort.InferenceSession(CLF32, providers=["CPUExecutionProvider"]); c32_in = clf32.get_inputs()[0].name
clf8 = ort.InferenceSession(CLF8, providers=["CPUExecutionProvider"]); c8_in = clf8.get_inputs()[0].name

def prep(b):
    if isinstance(b["img"], list):
        b["img"] = stack_batch_img(b["img"], divisible=32)
    return b

def classify(sess, iname, rgb_crop):
    im = Image.fromarray(rgb_crop).resize((224, 224), Image.BILINEAR)
    x = np.asarray(im, np.float32) / 255.0
    x = (x - IM_MEAN) / IM_STD
    x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)
    return int(np.argmax(sess.run(None, {iname: x})[0][0]))

n = hit = 0
c32_ok = c8_ok = 0
hit_ok32 = fb_ok32 = hit_n = fb_n = 0
for b in loader:
    b = prep(b)
    preds = det.run(None, {det_in: b["img"].numpy().astype(np.float32)})[0]
    dets = model.head.post_process(torch.from_numpy(preds), b)
    img_id = list(dets.keys())[0]
    fn = id2file[img_id]
    if fn not in gt: continue
    n += 1
    g = gt[fn]
    bgr = cv2.imread(os.path.join(IMGDIR, fn)); H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    birds = dets[img_id].get(BIRD_CLASS, [])
    birds = [bx for bx in birds if bx[4] >= THRESH]
    if birds:
        hit += 1; is_hit = True
        bx = max(birds, key=lambda z: z[4])
        x1, y1, x2, y2 = [int(max(0, v)) for v in bx[:4]]
        x2 = min(W, x2); y2 = min(H, y2)
        crop = rgb[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else rgb
    else:
        is_hit = False; crop = rgb
    p32 = classify(clf32, c32_in, crop); p8 = classify(clf8, c8_in, crop)
    c32_ok += (p32 == g); c8_ok += (p8 == g)
    if is_hit: hit_n += 1; hit_ok32 += (p32 == g)
    else: fb_n += 1; fb_ok32 += (p32 == g)
    if n % 150 == 0: print("  %d/%d hit=%d c32=%.3f" % (n, len(ds), hit, c32_ok / n), flush=True)

print("\n############ 级联端到端(birds525 test 子集 n=%d) ############" % n, flush=True)
print("bird 检出率(>=%.1f): %.4f (%d/%d)" % (THRESH, hit / n, hit, n), flush=True)
print("级联 top-1  fp32分类器: %.4f" % (c32_ok / n), flush=True)
print("级联 top-1  int8分类器: %.4f" % (c8_ok / n), flush=True)
print("  └ 检出鸟时 top-1(fp32): %.4f (n=%d)" % (hit_ok32 / max(hit_n,1), hit_n), flush=True)
print("  └ 回退全图 top-1(fp32): %.4f (n=%d)" % (fb_ok32 / max(fb_n,1), fb_n), flush=True)
print("(对照:分类器直评 test top-1 = 0.921)", flush=True)
print("=== CASCADE EVAL DONE ===", flush=True)
