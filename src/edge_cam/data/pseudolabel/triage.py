"""③ 置信分层（数据管线 §3b「MD 高置信自动收 / 中置信 Label Studio 人审 / 低置信丢」）。

**分层 QA 的意义**：全人审不可扩展、全自动信 MD 会漏错框；按框置信度把图分三层——只人审"拿不准"的
那批，既保规模又控质量。整图路由 + 逐框过滤（纯函数可测）：

  - **auto**  （该图存在框 score≥conf_hi）→ 整图收，只留 score≥conf_hi 的框，provenance=md_pseudo
  - **review**（该图 max box ∈ [conf_lo, conf_hi)）→ 送 LS 人审，预标注=score≥conf_lo 的框
  - **dropped**（该图 max box < conf_lo）→ 丢

只留 auto 图里 ≥hi 的框（不掺中置信框）：iNat 非穷尽（`exhaustive=False`），漏掉同图第二只弱框
不会当负样本污染；宁缺毋滥。conf_lo 默认对齐 MD 推理阈 0.2。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TriageResult:
    """分层产出：三份 COCO 视图（auto/review 各自可写盘）+ 计数统计。"""

    auto: dict = field(default_factory=dict)  # 高置信自动收（md_pseudo）
    review: dict = field(default_factory=dict)  # 中置信 → Label Studio 人审
    dropped_image_ids: list[int] = field(default_factory=list)  # 低置信丢
    stats: dict[str, int] = field(default_factory=dict)


def _img_index(coco: dict) -> dict[int, dict]:
    return {img["id"]: img for img in coco.get("images", [])}


def _anns_by_img(coco: dict) -> dict[int, list[dict]]:
    by: dict[int, list[dict]] = {}
    for a in coco.get("annotations", []):
        by.setdefault(a["image_id"], []).append(a)
    return by


def _rebuild(images: list[dict], anns: list[dict], categories: list[dict]) -> dict:
    """按选中的 images/anns 重排 id（image_id 连续、ann_id 连续），产干净 COCO。"""
    old_to_new = {img["id"]: i for i, img in enumerate(images, 1)}
    new_images = [{**img, "id": old_to_new[img["id"]]} for img in images]
    new_anns = []
    for aid, a in enumerate(anns, 1):
        new_anns.append({**a, "id": aid, "image_id": old_to_new[a["image_id"]]})
    return {"images": new_images, "annotations": new_anns, "categories": categories}


def triage_by_confidence(coco: dict, *, conf_hi: float = 0.7, conf_lo: float = 0.2) -> TriageResult:
    """按框 score 把伪标注 COCO 分三层（纯函数可测）。要求 annotation 带 `score`。"""
    imgs = _img_index(coco)
    anns_by = _anns_by_img(coco)
    cats = coco.get("categories", [])

    auto_imgs, auto_anns, review_imgs, review_anns, dropped = [], [], [], [], []
    for img_id, img in imgs.items():
        boxes = anns_by.get(img_id, [])
        max_score = max((b.get("score", 0.0) for b in boxes), default=0.0)
        if max_score >= conf_hi:
            auto_imgs.append(img)
            auto_anns.extend(b for b in boxes if b.get("score", 0.0) >= conf_hi)
        elif max_score >= conf_lo:
            review_imgs.append(img)
            review_anns.extend(b for b in boxes if b.get("score", 0.0) >= conf_lo)
        else:
            dropped.append(img_id)

    stats = {
        "n_images": len(imgs),
        "auto_images": len(auto_imgs),
        "auto_boxes": len(auto_anns),
        "review_images": len(review_imgs),
        "review_boxes": len(review_anns),
        "dropped_images": len(dropped),
    }
    return TriageResult(
        auto=_rebuild(auto_imgs, auto_anns, cats),
        review=_rebuild(review_imgs, review_anns, cats),
        dropped_image_ids=dropped,
        stats=stats,
    )
