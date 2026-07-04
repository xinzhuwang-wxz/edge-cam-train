"""全维度数据集合理性复查（训练前）。"""
import sys
from collections import defaultdict, Counter
sys.path.insert(0, "/root/autodl-tmp/ect/src")
from edge_cam.contracts.schemas.detection_manifest import DetectionManifest
from edge_cam.data.gate import gate

INV = {0: "bird", 1: "squirrel", 2: "cat", 3: "person", 4: "other_animal"}
DOMAIN = {
    "roboflow_meproject": "feeder-cam", "roboflow_feeder": "feeder-cam",
    "roboflow_birdv2": "clear-photo", "ena24": "camera-trap", "caltech_ct": "camera-trap",
    "open_images_v7": "web", "inat_md": "inat-natural",
}
R = "/root/autodl-tmp/detect_round2"
mtr = DetectionManifest.load(f"{R}/manifest_train.jsonl")
mte = DetectionManifest.load(f"{R}/manifest_test.jsonl")

print("="*70)
print("维度1 量 + 维度2 均衡 + 维度4 署名 + 维度9 许可 → 数据门")
print("="*70)
g = gate(mtr, split="train")
print(g.summary())
print(f"\n>>> 数据门总判定: {'PASS ✅' if g.passed else 'FAIL ❌'}")

print("\n" + "="*70)
print("维度3 域混合（bird 各域占比——要 feeder 域显著，非 web 主导）")
print("="*70)
def box_by(m, split):
    cls_box = defaultdict(Counter); dom_bird = Counter(); src_bird = Counter()
    for r in m.records:
        if r.split not in split: continue
        for b in r.boxes:
            c = INV[b.category_id]; cls_box[c][r.source] += 1
            if c == "bird":
                dom_bird[DOMAIN.get(r.source, r.source)] += 1; src_bird[r.source] += 1
    return cls_box, dom_bird, src_bird
cls_box, dom_bird, src_bird = box_by(mtr, {"train", "val"})
tot_bird = sum(dom_bird.values())
print(f"bird 总框 {tot_bird}，按域:")
for d, n in dom_bird.most_common():
    print(f"    {d:14s} {n:6d}  {100*n/tot_bird:.1f}%")
feeder = dom_bird.get("feeder-cam", 0) + dom_bird.get("clear-photo", 0)
print(f">>> feeder+clear 域占 bird: {100*feeder/tot_bird:.1f}%  (web 占 {100*dom_bird.get('web',0)/tot_bird:.1f}%)")

print("\n" + "="*70)
print("维度5 provenance + 维度8 负样本 + 维度11 test 集")
print("="*70)
for tag, m in [("TRAIN+VAL", mtr), ("TEST", mte)]:
    sp = {"train","val"} if tag=="TRAIN+VAL" else {"test"}
    prov = Counter(); n=0; neg=0; cls=Counter()
    for r in m.records:
        if r.split not in sp: continue
        n+=1
        if not r.boxes: neg+=1
        for b in r.boxes: prov[b.label_provenance]+=1; cls[INV[b.category_id]]+=1
    tot=sum(prov.values()) or 1
    print(f"{tag}: 图{n} 负{neg}({100*neg/n:.1f}%) prov={dict(prov)} pseudo={100*prov.get('md_pseudo',0)/tot:.1f}%")
    print(f"    每类框: " + " ".join(f"{c}={cls[c]}" for c in INV.values()))

print("\n" + "="*70)
print("维度4 框尺度（bird 中位/远景残留——验 tiny 滤生效）")
print("="*70)
for tag, m in [("TRAIN+VAL", mtr), ("TEST", mte)]:
    sp = {"train","val"} if tag=="TRAIN+VAL" else {"test"}
    af=defaultdict(list)
    for r in m.records:
        if r.split not in sp or not r.width or not r.height: continue
        for b in r.boxes:
            af[INV[b.category_id]].append(100*b.bbox[2]*b.bbox[3]/(r.width*r.height))
    for c in ["bird","squirrel","cat","person"]:
        a=sorted(af[c])
        if a:
            p=lambda q:a[min(len(a)-1,int(q*len(a)))]
            print(f"  {tag[:5]} {c:9s} 中位{p(0.5):.1f}% p90={p(0.9):.1f}% <1%残留:{sum(1 for x in a if x<1)} n={len(a)}")

print("\n" + "="*70)
print("维度7 split 泄漏（相机陷阱按 location 整组；查图路径跨 split 重复）")
print("="*70)
seen=defaultdict(set)
for tag,m in [("train",mtr),("test",mte)]:
    for r in m.records: seen[r.path].add(r.split if tag=="train" else "test")
leak=[p for p,s in seen.items() if len(s)>1]
print(f"跨 split 同图: {len(leak)} 张 (0=无泄漏)")
print("\nDIM_CHECK_DONE")
