"""检测数据集 → Label Studio 浏览（肉眼验域质量）。

把 `DetectionManifest`（5 类带框）导成 LS 导入任务：GT 框作 `predictions` 预标注（5 类各色），
task.meta 带 source/split（LS 里可按源/类筛）。**分层抽样**（每源取 N）保证各域都看得到
（feeder-cam / 网图 / 相机陷阱 / iNat / 清晰照）+ 掺负样本（空帧）。

纯函数（抽样/转任务/像素↔百分比）可测；LS 服务另起（docs/detect/伪标注-labelstudio.md 同法）。
"""

from __future__ import annotations

import hashlib

# 5 类各配色（LS RectangleLabels）——与 run.py 预览色系一致，肉眼分类
CLASS_COLORS = {
    "bird": "#3CC83C",
    "squirrel": "#F0A028",
    "cat": "#3C78F0",
    "person": "#F03C3C",
    "other_animal": "#9B59B6",
}
_FROM, _TO = "label", "image"


def label_config() -> str:
    """LS 项目标注配置（5 类 RectangleLabels）。建项目时贴这段。"""
    labels = "\n".join(
        f'    <Label value="{c}" background="{col}"/>' for c, col in CLASS_COLORS.items()
    )
    rl = f'  <RectangleLabels name="{_FROM}" toName="{_TO}">'
    return (
        '<View>\n  <Image name="image" value="$image"/>\n'
        f"{rl}\n{labels}\n  </RectangleLabels>\n</View>"
    )


def _rect_result(bbox: list[float], w: int, h: int, label: str) -> dict:
    x, y, bw, bh = bbox
    return {
        "type": "rectanglelabels",
        "from_name": _FROM,
        "to_name": _TO,
        "original_width": w,
        "original_height": h,
        "value": {
            "x": 100.0 * x / w,
            "y": 100.0 * y / h,
            "width": 100.0 * bw / w,
            "height": 100.0 * bh / h,
            "rectanglelabels": [label],
        },
    }


def stratified_sample(manifest, *, per_source: int = 40, seed: str = "ls", split=None) -> list:
    """每源确定性抽 per_source 张（sha256 排序，可复现）；split 限定则只取该 split。
    保各域都有代表（含负样本）。返回 DetImageRecord 列表。"""
    by_src: dict[str, list] = {}
    for r in manifest.records:
        if split and r.split not in split:
            continue
        by_src.setdefault(r.source, []).append(r)
    out = []
    for recs in by_src.values():
        recs_sorted = sorted(
            recs, key=lambda r: hashlib.sha256(f"{seed}:{r.path}".encode()).hexdigest()
        )
        out.extend(recs_sorted[:per_source])
    out.sort(key=lambda r: (r.source, r.path))
    return out


def records_to_ls_tasks(
    records: list, categories: dict[str, int], *, image_url_prefix: str = "/data/local-files/?d="
) -> list[dict]:
    """DetImageRecord 列表 → LS 任务（GT 框作 predictions；纯函数可测）。"""
    inv = {v: k for k, v in categories.items()}
    tasks = []
    for r in records:
        w, h = int(r.width), int(r.height)
        results = []
        if w > 0 and h > 0:
            for b in r.boxes:
                results.append(_rect_result(b.bbox, w, h, inv.get(b.category_id, "other_animal")))
        tasks.append(
            {
                "data": {"image": f"{image_url_prefix}{r.path}"},
                "predictions": [{"model_version": "gt", "result": results}],
                "meta": {
                    "source": r.source,
                    "split": r.split,
                    "n_boxes": len(r.boxes),
                    "classes": sorted({inv.get(b.category_id, "?") for b in r.boxes}),
                    "path": r.path,
                },
            }
        )
    return tasks
