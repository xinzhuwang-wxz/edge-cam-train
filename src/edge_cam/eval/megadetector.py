"""MegaDetector（MDV6）框准评估（plan §5「现在」：评它当伪标注器/基线靠不靠谱）。

MD 不上板、不当训练起点。**许可**：MD 经 `pytorch-wildlife` 跑会拉 ultralytics(AGPL)——故
**lazy import**、仅在**隔离 env** 作一次性评估工具，不进本仓依赖、不发行、不碰我们权重（§4）。
默认 `MDV6-apa-rtdetr`(Apache)。

口径：MD 出 animal/person/vehicle；GT 5 类折叠到 animal/person 对齐 → 复用
`eval.detect_metrics.evaluate_coco` 出 AP50（animal/person/整体）+ 额外算 **bird 类被 MD 任意框
召回率@IoU0.5**（最关心：MD 能不能把鸟框出来）。纯函数（折叠/IoU/召回）可测；MD 推理在 box 跑。
"""

from __future__ import annotations

import json
from pathlib import Path

# GT 5 类名 → 折叠类 id（与 MD 对齐）：非人动物→animal(1)，person→person(2)。
GT5_TO_COLLAPSED: dict[str, int] = {
    "bird": 1,
    "squirrel": 1,
    "cat": 1,
    "other_animal": 1,
    "person": 2,
}
COLLAPSED_CATS = [{"id": 1, "name": "animal"}, {"id": 2, "name": "person"}]
# pytorch-wildlife MDV6 class_id → 折叠类 id（0=animal,1=person,2=vehicle；vehicle 丢）。
MD_CLASSID_TO_COLLAPSED: dict[int, int] = {0: 1, 1: 2}


def build_gt_coco_collapsed(manifest, split: str) -> tuple[dict, list]:
    """manifest 某 split → 折叠版 GT COCO dict + 有序记录列表（image_id=列表下标，pred 须对齐）。"""
    inv = {v: k for k, v in manifest.categories.items()}  # id → 5 类名
    records = [r for r in manifest.records if r.split == split]
    images, anns, aid = [], [], 1
    for img_id, r in enumerate(records):
        images.append({"id": img_id, "file_name": r.path, "width": r.width, "height": r.height})
        for b in r.boxes:
            cc = GT5_TO_COLLAPSED.get(inv.get(b.category_id, ""))
            if cc is None:
                continue
            x, y, w, h = b.bbox
            anns.append(
                {
                    "id": aid,
                    "image_id": img_id,
                    "category_id": cc,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            aid += 1
    return {"images": images, "annotations": anns, "categories": COLLAPSED_CATS}, records


def iou_xywh(a: list[float], b: list[float]) -> float:
    """两个 COCO [x,y,w,h] 框的 IoU。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def class_recall_by_any(
    gt_boxes_by_img: dict[int, list[list[float]]],
    pred_boxes_by_img: dict[int, list[list[float]]],
    iou_thr: float = 0.5,
) -> float:
    """GT 某类框被 pred **任意**框命中(IoU≥thr)的召回率（MD 不分鸟种 → 类无关匹配）。"""
    total = matched = 0
    for img_id, gts in gt_boxes_by_img.items():
        preds = pred_boxes_by_img.get(img_id, [])
        for g in gts:
            total += 1
            if any(iou_xywh(g, p) >= iou_thr for p in preds):
                matched += 1
    return matched / total if total else 0.0


def run_megadetector(
    records: list,
    raw_root: str,
    version: str,
    conf: float = 0.2,
    *,
    weights: str | None = None,
    device: str = "cuda",
) -> list[dict]:
    """跑 MDV6（lazy import pytorch-wildlife，隔离 env）→ COCO 结果列表
    [{image_id, category_id, bbox[x,y,w,h], score}]。image_id 与 build_gt_coco_collapsed 对齐。

    `weights`：本地权重路径。传了 → **离线直接加载**（绕开 pytorch-wildlife 用 wget 下 zenodo
    权重，GFW 下被拒）；version 仍需传（PW 高层要它校验）。留空则按 version 走在线下载。

    ⚠️ pytorch-wildlife 版本间 API 略有差异；上 box 跑时按其文档核对 version 串与返回结构。
    """
    import numpy as np
    from PIL import Image
    from PytorchWildlife.models import detection as pw_detection

    model = pw_detection.MegaDetectorV6(
        weights=weights, pretrained=True, version=version, device=device
    )
    preds: list[dict] = []
    for img_id, r in enumerate(records):
        img = np.array(Image.open(Path(raw_root) / r.path).convert("RGB"))
        res = model.single_image_detection(img)
        det = res["detections"]  # supervision.Detections: xyxy / confidence / class_id
        for (x1, y1, x2, y2), cid, score in zip(
            det.xyxy, det.class_id, det.confidence, strict=False
        ):
            cc = MD_CLASSID_TO_COLLAPSED.get(int(cid))
            if cc is None or score < conf:
                continue
            preds.append(
                {
                    "image_id": img_id,
                    "category_id": cc,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )
    return preds


def evaluate(manifest, split: str, raw_root: str, out_dir: str, version: str) -> dict:
    """端到端：跑 MD → 写 gt/pred json → AP50（复用 evaluate_coco）+ bird 召回 → 汇总。"""
    from edge_cam.eval.detect_metrics import evaluate_coco

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gt_coco, records = build_gt_coco_collapsed(manifest, split)
    preds = run_megadetector(records, raw_root, version)
    (out / "md_gt.json").write_text(json.dumps(gt_coco))
    (out / "md_pred.json").write_text(json.dumps(preds))
    metrics = evaluate_coco(out / "md_gt.json", out / "md_pred.json")

    inv = {v: k for k, v in manifest.categories.items()}
    bird_gt = {
        i: [b.bbox for b in r.boxes if inv.get(b.category_id) == "bird"]
        for i, r in enumerate(records)
    }
    pred_boxes = {}
    for p in preds:
        pred_boxes.setdefault(p["image_id"], []).append(p["bbox"])
    bird_recall = class_recall_by_any({i: g for i, g in bird_gt.items() if g}, pred_boxes)

    summary = {
        "version": version,
        "split": split,
        "ap50": metrics.map_50,
        "ap5095": metrics.map_5095,
        "per_class_ap": metrics.per_class_ap,
        "bird_recall@0.5": round(bird_recall, 4),
        "n_images": len(records),
        "n_preds": len(preds),
    }
    (out / "md_eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[md-eval] {summary}")
    return summary
