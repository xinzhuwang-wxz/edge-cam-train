"""M-zero:ppyoloe_s COCO 零样本(不微调)在固定 test 上,把 COCO 类映射到我们 5 类后 COCOeval。
用途:量"预训练本身知道多少"(尤其 bird),对比 M-ft(微调后)与 NanoDet P0。

输入:
  --det   ppyoloe COCO 推理输出的 det json(list[{image_id,category_id,bbox[xywh],score}])
  --gt    我们的固定 test GT(detect_round3/labels/test_test.json,category_id 1..5)
  --idscheme  coco80(连续0-79,PaddleDetection常见) | coco91(原始1-90)
映射(按 COCO 类名):bird→bird · cat→cat · person→person · 8种COCO动物→other_animal;
  squirrel COCO 无 → 天然测不到(≈0,预期)。remapped det → COCOeval vs GT → 逐类 AP50(重点 bird)。
"""

import argparse
import json

# COCO80 连续顺序(index=输出类id when idscheme=coco80)
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
# COCO91 原始 id → 名(仅列我们关心的)
COCO91_NAME = {1: "person", 16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
               21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe"}

# COCO 名 → 我们 GT 的 category_id(1..5)
OTHER = {"dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
NAME_TO_OURS = {"bird": 1, "cat": 3, "person": 4, **{n: 5 for n in OTHER}}


def coco_id_to_name(cid: int, scheme: str) -> str | None:
    if scheme == "coco80":
        return COCO80[cid] if 0 <= cid < len(COCO80) else None
    return COCO91_NAME.get(cid)  # coco91:只解析我们关心的


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--idscheme", default="coco80", choices=["coco80", "coco91"])
    ap.add_argument("--out", default="/root/autodl-tmp/m_zero_det_mapped.json")
    a = ap.parse_args()

    with open(a.det) as f:
        dets = json.load(f)
    mapped, kept = [], {"bird": 0, "cat": 0, "person": 0, "other": 0}
    for d in dets:
        name = coco_id_to_name(d["category_id"], a.idscheme)
        our = NAME_TO_OURS.get(name) if name else None
        if our is None:
            continue
        mapped.append({**d, "category_id": our})
        kept[{1: "bird", 3: "cat", 4: "person", 5: "other"}[our]] += 1
    with open(a.out, "w") as f:
        json.dump(mapped, f)
    print("mapped dets:", len(mapped), "of", len(dets), "| by our-class:", kept)

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(a.gt)
    dt = gt.loadRes(a.out)
    ev = COCOeval(gt, dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    name_of = {c["id"]: c["name"] for c in gt.dataset["categories"]}
    print("=== M-zero 逐类 AP50（IoU=0.5，对齐 round2；重点 bird；squirrel COCO 无≈0）===")
    for i, cid in enumerate(gt.getCatIds()):
        p50 = ev.eval["precision"][0, :, i, 0, 2]  # [0]=IoU0.5
        v50 = p50[p50 > -1]
        p5095 = ev.eval["precision"][:, :, i, 0, 2]
        v9 = p5095[p5095 > -1]
        ap50 = 100 * float(v50.mean()) if v50.size else 0.0
        ap9 = 100 * float(v9.mean()) if v9.size else 0.0
        print(f"  {name_of[cid]:<14s} AP50={ap50:.1f}  (AP.5:.95={ap9:.1f})")


if __name__ == "__main__":
    main()
