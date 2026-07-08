"""检测混淆矩阵 + 类判别（round3 §6：把"对角线大不大 / 松鼠↔鸟混淆"量化，进回归护栏）。

与 `eval/analyze.py`（**分类**混淆：logits→top1）互补：本模块是**检测**混淆——预测框↔GT 框按 IoU
匹配，含 **background 行/列**（漏检=真值→背景、误报=背景→预测）。
约定 **行=真值、列=预测、对角=类正确**（Ultralytics/Tenyks 口径）。
纯函数、**可无卡跑**（喂 COCO GT dict + COCO 检测预测 list）。

匹配语义（关键）：预测↔GT **只按 IoU 匹配、不看类**（贪心，按分数高→低），再记录 (真类, 预测类) 对——
这样"框对了但类判错"（如松鼠框被判成 bird）才会落到非对角格，正是要量的混淆。

指标：
- `diagonal_rate`：已定位框里类判对的占比（对角线占比，越大越好）——用户"对角线要大"的直接量化。
- `class_confusion_rate`：已定位框里**类判错**的占比（非对角、非背景），越小越好。
- `confused_pairs`：最亮的 (真类→预测类) 错分对（如 squirrel→bird 具体多少）。

注：top1-top2 置信度 margin 需**全类分向量**（NanoDet NMS 后只留最终类+分，须 GPU 定制推理 dump），
本模块不含，留 §6 hook。
"""

from __future__ import annotations

from dataclasses import dataclass


def _iou_xywh(a: list[float], b: list[float]) -> float:
    """两个 COCO [x, y, w, h] 框的 IoU。"""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class ConfusionMatrix:
    """检测混淆矩阵。matrix[t][p]：真类 t → 预测类 p；末行/末列 = background（误报源/漏检）。"""

    names: list[str]  # 类名（不含 background）
    matrix: list[list[int]]  # (n+1) x (n+1)，索引 n = background
    iou_thr: float
    conf_thr: float

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def bg(self) -> int:
        return len(self.names)

    def _localized(self) -> int:
        """已定位（真类∈类 且 预测类∈类）的 GT 框总数 = 类×类子矩阵之和。"""
        return sum(self.matrix[t][p] for t in range(self.n) for p in range(self.n))

    def diagonal_rate(self) -> float:
        """已定位框里类判对占比（对角线占比）。"""
        loc = self._localized()
        if loc == 0:
            return 0.0
        diag = sum(self.matrix[c][c] for c in range(self.n))
        return diag / loc

    def class_confusion_rate(self) -> float:
        """已定位框里类判错占比（非对角、非背景）。"""
        loc = self._localized()
        return 0.0 if loc == 0 else 1.0 - self.diagonal_rate()

    def per_class_recall(self) -> dict[str, float]:
        """每真类：类判对数 / 该真类 GT 总数（含漏检）。"""
        out: dict[str, float] = {}
        for c in range(self.n):
            total = sum(self.matrix[c])  # 该真类的全部去向（含背景=漏检）
            out[self.names[c]] = self.matrix[c][c] / total if total else 0.0
        return out

    def confused_pairs(self, top: int = 8) -> list[tuple[str, str, int]]:
        """最亮的 (真类→预测类) 错分对（类↔类，排除对角与背景）。"""
        pairs = [
            (self.names[t], self.names[p], self.matrix[t][p])
            for t in range(self.n)
            for p in range(self.n)
            if t != p and self.matrix[t][p] > 0
        ]
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top]

    def to_markdown(self, normalize: bool = True) -> str:
        """行=真值、列=预测（末=bg）。normalize=True 按真值行归一化（看漏去哪）。"""
        labels = [*self.names, "bg"]
        head = "| true\\pred | " + " | ".join(labels) + " |"
        sep = "|" + "---|" * (len(labels) + 1)
        lines = [head, sep]
        for t in range(self.n + 1):
            row_total = sum(self.matrix[t]) or 1
            cells = []
            for p in range(self.n + 1):
                v = self.matrix[t][p]
                cells.append(f"{v / row_total:.2f}" if normalize else str(v))
            name = labels[t] if t < self.n else "bg"
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        return "\n".join(lines)


def build_confusion(
    gt_coco: dict,
    preds: list[dict],
    *,
    iou_thr: float = 0.5,
    conf_thr: float = 0.4,
) -> ConfusionMatrix:
    """COCO GT dict + COCO 检测预测 list → ConfusionMatrix（纯函数、无卡）。

    gt_coco: COCO dict（images / annotations[bbox] / categories）。
    preds:   COCO 检测 list [{image_id, category_id, bbox[x,y,w,h], score}]（results0.json）。
    类顺序取 categories 出现序；category_id 映射到 0..n-1。
    """
    cats = sorted(gt_coco["categories"], key=lambda c: c["id"])
    names = [c["name"] for c in cats]
    cid2idx = {c["id"]: i for i, c in enumerate(cats)}
    n = len(names)
    bg = n
    mat = [[0 for _ in range(n + 1)] for _ in range(n + 1)]

    # 按图归拢 GT / 预测
    gts_by_img: dict[int, list[tuple[int, list[float]]]] = {}
    for a in gt_coco["annotations"]:
        gts_by_img.setdefault(a["image_id"], []).append((cid2idx[a["category_id"]], a["bbox"]))
    preds_by_img: dict[int, list[tuple[int, list[float], float]]] = {}
    for p in preds:
        if p["score"] < conf_thr:
            continue
        idx = cid2idx.get(p["category_id"])
        if idx is None:
            continue
        preds_by_img.setdefault(p["image_id"], []).append((idx, p["bbox"], p["score"]))

    all_img_ids = {im["id"] for im in gt_coco["images"]}
    for img_id in all_img_ids:
        gts = gts_by_img.get(img_id, [])
        dets = sorted(preds_by_img.get(img_id, []), key=lambda d: d[2], reverse=True)  # 分数高→低
        gt_used = [False] * len(gts)
        det_used = [False] * len(dets)
        # 贪心：每个预测（按分数）匹配 IoU 最高且未用的 GT（只看 IoU，不看类）
        for di, (pcls, pbox, _score) in enumerate(dets):
            best_iou, best_gi = iou_thr, -1
            for gi, (_gcls, gbox) in enumerate(gts):
                if gt_used[gi]:
                    continue
                iou = _iou_xywh(pbox, gbox)
                if iou >= best_iou:
                    best_iou, best_gi = iou, gi
            if best_gi >= 0:
                gt_used[best_gi] = True
                det_used[di] = True
                mat[gts[best_gi][0]][pcls] += 1  # 真类→预测类
        for gi, used in enumerate(gt_used):
            if not used:
                mat[gts[gi][0]][bg] += 1  # 漏检：真类→背景
        for di, used in enumerate(det_used):
            if not used:
                mat[bg][dets[di][0]] += 1  # 误报：背景→预测类

    return ConfusionMatrix(names=names, matrix=mat, iou_thr=iou_thr, conf_thr=conf_thr)
