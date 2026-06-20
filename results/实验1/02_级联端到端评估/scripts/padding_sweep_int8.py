"""缓解验证:框 padding 扫描。一次检测得 bird 框,对多个 padding 因子各裁剪+分类(fp32),
看级联 top-1 随 padding 恢复多少(贴近分类器训练取景)。p=full=整张图(=训练分布上限)。"""
import os, json, numpy as np, torch, cv2
import onnxruntime as ort
from PIL import Image
from nanodet.data.collate import naive_collate
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.dataset import build_dataset
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config

CFG = "outputs/detect/nanodet_cascade.yml"
DET = "outputs/detect/nanodet_feeder_416.int8.onnx"
CLF = "/root/autodl-tmp/eff_lite0_int8.onnx"
IMGDIR = "/root/autodl-tmp/cascade_data/images"
BIRD, THRESH = 0, 0.3
PADS = [0.0, 0.15, 0.3, 0.5]   # + full(整图)
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)

gt = json.load(open("/root/autodl-tmp/cascade_data/gt_species.json"))
id2file = {im["id"]: im["file_name"] for im in json.load(open("/root/autodl-tmp/cascade_data/labels.json"))["images"]}
load_config(cfg, CFG)
ds = build_dataset(cfg.data.val, "test")
loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=naive_collate)
model = build_model(cfg.model); model.eval()
det = ort.InferenceSession(DET, providers=["CPUExecutionProvider"]); din = det.get_inputs()[0].name
clf = ort.InferenceSession(CLF, providers=["CPUExecutionProvider"]); cin = clf.get_inputs()[0].name

def prep(b):
    if isinstance(b["img"], list): b["img"] = stack_batch_img(b["img"], divisible=32)
    return b
def classify(rgb):
    im = Image.fromarray(rgb).resize((224,224), Image.BILINEAR)
    x = (np.asarray(im, np.float32)/255.0 - MEAN)/STD
    return int(np.argmax(clf.run(None, {cin: np.transpose(x,(2,0,1))[None].astype(np.float32)})[0][0]))

n = 0
ok = {p: 0 for p in PADS}; ok_full = 0
for b in loader:
    b = prep(b)
    preds = det.run(None, {din: b["img"].numpy().astype(np.float32)})[0]
    dets = model.head.post_process(torch.from_numpy(preds), b)
    iid = list(dets.keys())[0]; fn = id2file[iid]
    if fn not in gt: continue
    n += 1; g = gt[fn]
    bgr = cv2.imread(os.path.join(IMGDIR, fn)); H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    birds = [bx for bx in dets[iid].get(BIRD, []) if bx[4] >= THRESH]
    ok_full += (classify(rgb) == g)   # 整图参照
    if not birds:
        for p in PADS: ok[p] += (classify(rgb) == g)
        continue
    bx = max(birds, key=lambda z: z[4]); x1,y1,x2,y2 = bx[:4]; bw,bh = x2-x1, y2-y1
    for p in PADS:
        X1=int(max(0,x1-p*bw)); Y1=int(max(0,y1-p*bh)); X2=int(min(W,x2+p*bw)); Y2=int(min(H,y2+p*bh))
        crop = rgb[Y1:Y2, X1:X2] if (X2>X1 and Y2>Y1) else rgb
        ok[p] += (classify(crop) == g)
    if n % 150 == 0: print("  %d/%d p0=%.3f p0.3=%.3f full=%.3f" % (n, len(ds), ok[0.0]/n, ok[0.3]/n, ok_full/n), flush=True)

print("\n############ Padding 缓解扫描 全INT8 (n=%d, int8检测+int8分类) ############" % n, flush=True)
for p in PADS: print("  框 padding %4.0f%%:  级联 top-1 = %.4f" % (p*100, ok[p]/n), flush=True)
print("  整图(=训练取景上限):  top-1 = %.4f" % (ok_full/n), flush=True)
print("  (对照: 分类器直评 = 0.921)", flush=True)
print("=== PADDING SWEEP DONE ===", flush=True)
