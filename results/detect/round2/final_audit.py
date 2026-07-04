"""终审：查前面维度没覆盖的边角问题（严格找茬）。"""
import sys
from collections import Counter, defaultdict
sys.path.insert(0, "/root/autodl-tmp/ect/src")
from edge_cam.contracts.schemas.detection_manifest import DetectionManifest

INV = {0: "bird", 1: "squirrel", 2: "cat", 3: "person", 4: "other_animal"}
issues = []
for tag, p in [("TRAIN+VAL", "/root/autodl-tmp/detect_round2/manifest_train.jsonl"),
               ("TEST", "/root/autodl-tmp/detect_round2/manifest_test.jsonl")]:
    m = DetectionManifest.load(p)
    recs = m.records
    print(f"\n===== {tag} ({len(recs)} 图) =====")

    # 1. 极端长宽比框（w/h>15 或 <1/15 → 退化/坏框）
    extreme_ar = 0
    for r in recs:
        for b in r.boxes:
            w, h = b.bbox[2], b.bbox[3]
            if w > 0 and h > 0:
                ar = w / h
                if ar > 15 or ar < 1 / 15:
                    extreme_ar += 1
    print(f"1. 极端长宽比框(>15:1): {extreme_ar}")
    if extreme_ar > len([b for r in recs for b in r.boxes]) * 0.005:
        issues.append(f"{tag}: 极端长宽比框偏多 {extreme_ar}")

    # 2. 框数异常图（>25 框 → 可能标注噪声/穷尽源密集场景）
    boxcnt = Counter(len(r.boxes) for r in recs)
    many = sum(v for k, v in boxcnt.items() if k > 25)
    mx = max((len(r.boxes) for r in recs), default=0)
    print(f"2. 框数: 最多 {mx}/图；>25 框的图 {many}")

    # 3. 坏图尺寸（<64px 或 0）
    tiny = sum(1 for r in recs if r.width and (min(r.width, r.height) < 64))
    zero = sum(1 for r in recs if not r.width or not r.height)
    print(f"3. 图尺寸: <64px {tiny}；0/缺尺寸 {zero}")
    if zero:
        issues.append(f"{tag}: {zero} 图缺尺寸")

    # 4. 跨源重复文件名（basename 撞 → 潜在重复图）
    base = defaultdict(set)
    for r in recs:
        base[r.path.split("/")[-1]].add(r.source)
    dup = {k: v for k, v in base.items() if len(v) > 1}
    print(f"4. 跨源同名文件: {len(dup)}（{list(dup.items())[:2] if dup else '无'}）")

    # 5. 单类占图比（某类是否被某源垄断到失衡）——informational
    # 6. 负面积/零框正样本（应为 0，gate 已查）
    degen = sum(1 for r in recs for b in r.boxes if b.bbox[2] <= 0 or b.bbox[3] <= 0)
    print(f"5. 退化框(零负面积): {degen}")
    if degen:
        issues.append(f"{tag}: {degen} 退化框")

    # 7. 每类图数 vs 框数（框/图比，异常高=密集）
    print("6. 每类 框/图比:")
    for c in INV.values():
        imgs = sum(1 for r in recs if any(INV[b.category_id] == c for b in r.boxes))
        boxes = sum(1 for r in recs for b in r.boxes if INV[b.category_id] == c)
        if imgs:
            print(f"     {c:13s} {boxes}框/{imgs}图 = {boxes / imgs:.2f}")

print("\n" + "=" * 50)
print("终审问题清单:", issues if issues else "无 ✅")
print("FINAL_AUDIT_DONE")
