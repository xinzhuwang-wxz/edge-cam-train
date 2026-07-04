"""量 MD 当 bird 教师的「难度」：在**已有 GT 带框**的 bird 图上跑 MD，看它框得准不准、多自信。

回答「这批鸟对 MD 好不好框」（数据管线 §3b「先量 MD 可信度」）：对每个 GT bird 框，找 IoU 最大的
MD 框——命中(IoU≥thr)则记 MD 置信度，未命中=MD 漏框。汇总：
  - **召回**（MD 框中的 GT bird 比例）/ **漏框率**
  - 命中框的**置信度分布**（落进 triage 桶：<lo 会被丢 / [lo,hi) 人审 / ≥hi 自动收）
  - 「若按 triage 阈值」GT bird 的自动收 / 人审 / 丢 占比 → 直接照见 iNat 伪标注的预期产出结构

`match_bird_to_md` / `difficulty_summary` 纯函数可测；MD 推理复用 `run_megadetector`（box/GPU）。
"""

from __future__ import annotations

from edge_cam.eval.megadetector import iou_xywh


def match_bird_to_md(
    gt_by_img: dict[int, list[list[float]]],
    md_by_img: dict[int, list[tuple[list[float], float]]],
    *,
    iou_thr: float = 0.5,
) -> list[dict]:
    """对每个 GT bird 框，取同图 IoU 最大的 MD 框 → 匹配记录（**纯函数可测**）。

    gt_by_img: {image_id: [bbox[x,y,w,h], …]}（仅 bird 框）。
    md_by_img: {image_id: [(bbox, score), …]}（MD 所有框，低阈值全量）。
    返回 [{iou, conf, matched}]（每 GT 框一条；conf=最佳 MD 框置信，未命中 matched=False）。
    """
    out: list[dict] = []
    for img_id, gts in gt_by_img.items():
        preds = md_by_img.get(img_id, [])
        for g in gts:
            best_iou, best_conf = 0.0, 0.0
            for pbox, pscore in preds:
                iou = iou_xywh(g, pbox)
                if iou > best_iou:
                    best_iou, best_conf = iou, pscore
            out.append({"iou": best_iou, "conf": best_conf, "matched": best_iou >= iou_thr})
    return out


def difficulty_summary(matches: list[dict], *, conf_hi: float = 0.7, conf_lo: float = 0.2) -> dict:
    """匹配记录 → 难度汇总（**纯函数可测**）：召回/漏框 + 命中置信分布 + triage 预期产出占比。"""
    n = len(matches)
    if n == 0:
        return {"n_gt_bird": 0}
    hit = [m for m in matches if m["matched"]]
    n_hit = len(hit)
    confs = sorted(m["conf"] for m in hit)
    # 命中框置信度落桶
    buckets = {"<lo(会丢)": 0, "[lo,hi)(人审)": 0, ">=hi(自动收)": 0}
    for c in confs:
        if c < conf_lo:
            buckets["<lo(会丢)"] += 1
        elif c < conf_hi:
            buckets["[lo,hi)(人审)"] += 1
        else:
            buckets[">=hi(自动收)"] += 1
    # 若按 triage 阈值，每个 GT bird 会走哪条路（未命中或命中但<lo → 丢）
    auto = buckets[">=hi(自动收)"]
    review = buckets["[lo,hi)(人审)"]
    drop = n - auto - review
    med = confs[n_hit // 2] if n_hit else 0.0
    return {
        "n_gt_bird": n,
        "recall@iou": round(n_hit / n, 4),
        "miss_rate": round((n - n_hit) / n, 4),
        "hit_conf_median": round(med, 3),
        "hit_conf_mean": round(sum(confs) / n_hit, 3) if n_hit else 0.0,
        "hit_conf_buckets": buckets,
        "triage_expectation": {
            "auto_%": round(100 * auto / n, 1),
            "review_%": round(100 * review / n, 1),
            "drop_%": round(100 * drop / n, 1),
        },
    }


def run_bird_difficulty(
    manifest,
    split: str,
    raw_root: str,
    *,
    version: str = "MDV6-yolov9-c",
    weights: str | None = None,
    device: str = "cuda",
    conf_floor: float = 0.01,
    iou_thr: float = 0.5,
    max_images: int | None = None,
) -> dict:
    """box/GPU：manifest 里 GT bird 框 → 跑 MD（低阈全量）→ 匹配 → 难度汇总。

    conf_floor 取很低（默认 0.01）以看**完整置信分布**（含 MD 弱检测），而非被 0.2 截断。
    """
    from edge_cam.eval.megadetector import run_megadetector

    inv = {v: k for k, v in manifest.categories.items()}
    records = [r for r in manifest.records if r.split == split]
    # 只保留含 bird GT 框的图（量的是「有鸟的图 MD 好不好框」）
    bird_records = [r for r in records if any(inv.get(b.category_id) == "bird" for b in r.boxes)]
    if max_images:
        bird_records = bird_records[:max_images]

    preds = run_megadetector(
        bird_records, raw_root, version, conf=conf_floor, weights=weights, device=device
    )
    md_by_img: dict[int, list[tuple[list[float], float]]] = {}
    for p in preds:
        md_by_img.setdefault(p["image_id"], []).append((p["bbox"], p["score"]))
    gt_by_img = {
        i: [b.bbox for b in r.boxes if inv.get(b.category_id) == "bird"]
        for i, r in enumerate(bird_records)
    }
    matches = match_bird_to_md(gt_by_img, md_by_img, iou_thr=iou_thr)
    summary = difficulty_summary(matches)
    summary["n_bird_images"] = len(bird_records)
    summary["md_version"] = version
    return summary
