"""检测直观指标:置信度>=0.3 + IoU>=0.5 + 大类正确口径的逐类 召回(检出率)/查准。
比 mAP 直观:召回=真实目标找到的比例,查准=报出的框里对的比例。用 800 子集(快)。"""
import os, json, numpy as np, torch
import onnxruntime as ort
from nanodet.data.collate import naive_collate
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.dataset import build_dataset
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config

CFG = "outputs/detect/nanodet_feeder_416_sub.yml"  # 800 子集
DET = "outputs/detect/nanodet_feeder_416.onnx"
CONF, IOU = 0.3, 0.5
NAMES = ["bird","squirrel","cat","dog","raccoon","rabbit","deer","fox","skunk","hedgehog","bear"]

load_config(cfg, CFG)
ds = build_dataset(cfg.data.val, "test")
loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=naive_collate)
model = build_model(cfg.model); model.eval()
det = ort.InferenceSession(DET, providers=["CPUExecutionProvider"]); din = det.get_inputs()[0].name
coco = ds.coco_api if hasattr(ds, "coco_api") else ds.coco

def prep(b):
    if isinstance(b["img"], list): b["img"] = stack_batch_img(b["img"], divisible=32)
    return b
def iou(a, g):
    x1=max(a[0],g[0]); y1=max(a[1],g[1]); x2=min(a[2],g[2]); y2=min(a[3],g[3])
    iw=max(0,x2-x1); ih=max(0,y2-y1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(g[2]-g[0])*(g[3]-g[1])-inter
    return inter/ua if ua>0 else 0

TP=np.zeros(11); FP=np.zeros(11); GT=np.zeros(11)
for b in loader:
    b=prep(b)
    iid=b["img_info"]["id"]; iid=int(iid[0] if hasattr(iid,"__len__") else iid)
    # GT
    anns=coco.loadAnns(coco.getAnnIds(imgIds=[iid]))
    gts={c:[] for c in range(11)}
    for an in anns:
        x,y,w,h=an["bbox"]; gts[an["category_id"]-1].append([x,y,x+w,y+h])
    for c in range(11): GT[c]+=len(gts[c])
    preds=det.run(None,{din:b["img"].numpy().astype(np.float32)})[0]
    dets=model.head.post_process(torch.from_numpy(preds),b)[iid]
    for c in range(11):
        boxes=sorted([bx for bx in dets.get(c,[]) if bx[4]>=CONF], key=lambda z:-z[4])
        used=[False]*len(gts[c])
        for bx in boxes:
            best=-1; bi=-1
            for j,g in enumerate(gts[c]):
                if used[j]: continue
                v=iou(bx[:4],g)
                if v>=IOU and v>best: best=v; bi=j
            if bi>=0: TP[c]+=1; used[bi]=True
            else: FP[c]+=1

print("=== 检测直观指标(conf>=0.3, IoU>=0.5, 大类正确)===", flush=True)
print("%-10s %6s %6s %6s %8s %8s" % ("类","GT","TP","FP","召回率","查准率"), flush=True)
for c in range(11):
    rec=TP[c]/GT[c] if GT[c]>0 else 0
    prec=TP[c]/(TP[c]+FP[c]) if (TP[c]+FP[c])>0 else 0
    print("%-10s %6d %6d %6d %7.1f%% %7.1f%%" % (NAMES[c],GT[c],TP[c],FP[c],rec*100,prec*100), flush=True)
tr=TP.sum()/GT.sum(); tp_=TP.sum()/(TP.sum()+FP.sum())
print("%-10s %6d %6d %6d %7.1f%% %7.1f%%" % ("总体",GT.sum(),TP.sum(),FP.sum(),tr*100,tp_*100), flush=True)
print("=== DET PR DONE ===", flush=True)
