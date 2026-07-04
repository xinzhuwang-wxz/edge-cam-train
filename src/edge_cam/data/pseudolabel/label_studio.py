"""④ Label Studio 人审往返（纯函数可测）。

中置信图 → LS 导入 JSON（MD 框作 `predictions` 预标注，人只需微调/删/确认）→ 人审导出 →
回读为 COCO（框 provenance=md_human_verified）。LS 矩形坐标是**百分比**（x/y/w/h ∈ [0,100]），
与 COCO 像素 bbox 互转在此收口。

LS 项目最小标注配置（RectangleLabels，label=bird）：
    <View><Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image"><Label value="bird"/></RectangleLabels></View>
"""

from __future__ import annotations

MODEL_VERSION = "megadetector_v6"
_FROM, _TO, _LABEL = "label", "image", "bird"


def _to_pct_rect(bbox: list[float], w: int, h: int) -> dict:
    """COCO [x,y,bw,bh] 像素 → LS value(x/y/width/height 百分比 + rectanglelabels)。"""
    x, y, bw, bh = bbox
    return {
        "x": 100.0 * x / w,
        "y": 100.0 * y / h,
        "width": 100.0 * bw / w,
        "height": 100.0 * bh / h,
        "rectanglelabels": [_LABEL],
    }


def to_ls_tasks(coco: dict, *, image_url_prefix: str = "/data/local-files/?d=") -> list[dict]:
    """review COCO → LS 导入任务列表，MD 框作预标注（纯函数可测）。

    `image_url_prefix` 拼 file_name 成 LS 可取的图 URL（本地文件服务默认 /data/local-files/?d=）。
    """
    anns_by: dict[int, list[dict]] = {}
    for a in coco.get("annotations", []):
        anns_by.setdefault(a["image_id"], []).append(a)

    tasks: list[dict] = []
    for img in coco.get("images", []):
        w, h = int(img["width"]), int(img["height"])
        results = []
        for a in anns_by.get(img["id"], []):
            results.append(
                {
                    "type": "rectanglelabels",
                    "from_name": _FROM,
                    "to_name": _TO,
                    "original_width": w,
                    "original_height": h,
                    "image_rotation": 0,
                    "value": _to_pct_rect(a["bbox"], w, h),
                    "score": float(a.get("score", 0.0)),
                }
            )
        tasks.append(
            {
                "data": {"image": f"{image_url_prefix}{img['file_name']}"},
                "predictions": [{"model_version": MODEL_VERSION, "result": results}],
                "meta": {"file_name": img["file_name"], "width": w, "height": h},
            }
        )
    return tasks


def _from_pct_rect(value: dict, w: int, h: int) -> list[float]:
    """LS value(百分比) → COCO [x,y,bw,bh] 像素。"""
    return [
        value["x"] / 100.0 * w,
        value["y"] / 100.0 * h,
        value["width"] / 100.0 * w,
        value["height"] / 100.0 * h,
    ]


def from_ls_export(tasks: list[dict], *, category_id: int = 1, category_name: str = "bird") -> dict:
    """LS 人审导出（任务列表，每任务 annotations[].result[]）→ COCO（纯函数可测）。

    优先读人审 `annotations`（人的最终标注）；某任务被删空 = 人判无有效框 → 该图 0 框（仍入 images，
    交上层决定是否当负样本/丢）。宽高从 result.original_width/height 取，回退 meta。
    """
    images, anns, ann_id = [], [], 1
    for img_id, task in enumerate(tasks, 1):
        meta = task.get("meta", {})
        fn = meta.get("file_name") or task.get("data", {}).get("image", "")
        results = []
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            results = ann.get("result", [])
            break  # 取第一份未取消的人审
        w = h = 0
        for r in results:
            w = int(r.get("original_width") or meta.get("width") or 0)
            h = int(r.get("original_height") or meta.get("height") or 0)
            if r.get("type") != "rectanglelabels":
                continue
            anns.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": _from_pct_rect(r["value"], w, h),
                }
            )
            ann_id += 1
        w = w or int(meta.get("width", 0))
        h = h or int(meta.get("height", 0))
        images.append({"id": img_id, "file_name": fn, "width": w, "height": h})
    return {
        "images": images,
        "annotations": anns,
        "categories": [{"id": category_id, "name": category_name}],
    }
