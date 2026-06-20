"""本地备级联 test 子集:从 birds525 test 选 600 张(跨种) -> 拷到 staging +
给 NanoDet 的 dummy COCO labels.json(仅 images)+ 物种 GT 映射(file_name->species_idx)。"""
import json, os, random, shutil
from PIL import Image

MANIFEST = "data/processed/birds525/manifest.json"
ROOT = "/Users/bamboo/Downloads/bird_species(DST1192)"
STAGE = "/tmp/cascade_data"
IMGDIR = os.path.join(STAGE, "images")
os.makedirs(IMGDIR, exist_ok=True)
N = 600

d = json.load(open(MANIFEST))
cls2idx = d["class_to_idx"]
test = [r for r in d["records"] if r["split"] == "test"]
random.seed(0)
random.shuffle(test)
sel = test[:N]

images, gt = [], {}
feeder_cats = [{"id": i, "name": n} for i, n in enumerate(
    ["bird","squirrel","cat","dog","raccoon","rabbit","deer","fox","skunk","hedgehog","bear"])]
ok = 0
for i, r in enumerate(sel):
    src = os.path.join(ROOT, r["path"])
    if not os.path.exists(src):
        continue
    fn = "img%04d.jpg" % i
    dst = os.path.join(IMGDIR, fn)
    try:
        im = Image.open(src).convert("RGB")
        w, h = im.size
        im.save(dst, "JPEG")
    except Exception:
        continue
    images.append({"id": ok, "file_name": fn, "height": h, "width": w})
    gt[fn] = cls2idx[r["label"]]
    ok += 1

coco = {"images": images, "annotations": [], "categories": feeder_cats}
json.dump(coco, open(os.path.join(STAGE, "labels.json"), "w"))
json.dump(gt, open(os.path.join(STAGE, "gt_species.json"), "w"))
# 物种 idx->name(评分/报告用)
json.dump({v: k for k, v in cls2idx.items()}, open(os.path.join(STAGE, "idx2species.json"), "w"))
print("staged %d 张 -> %s" % (ok, IMGDIR))
print("labels.json images=%d, gt_species=%d, 物种类数=%d" % (len(images), len(gt), len(cls2idx)))
