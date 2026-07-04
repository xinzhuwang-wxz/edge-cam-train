"""Label Studio 往返：COCO→LS 预标注任务、LS 导出→COCO（纯函数，像素↔百分比互转对称）。"""

from __future__ import annotations

from edge_cam.data.pseudolabel.label_studio import from_ls_export, to_ls_tasks


def _review_coco():
    return {
        "images": [{"id": 1, "file_name": "inat/999.jpg", "width": 200, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 20, 40, 30], "score": 0.45}
        ],
        "categories": [{"id": 1, "name": "animal"}],
    }


def test_to_ls_task_shape_and_pct() -> None:
    tasks = to_ls_tasks(_review_coco(), image_url_prefix="/data/local-files/?d=")
    assert len(tasks) == 1
    t = tasks[0]
    assert t["data"]["image"] == "/data/local-files/?d=inat/999.jpg"
    res = t["predictions"][0]["result"][0]
    assert res["type"] == "rectanglelabels"
    v = res["value"]
    # bbox[50,20,40,30] on 200x100 → x=25%, y=20%, w=20%, h=30%
    assert v["x"] == 25.0 and v["y"] == 20.0 and v["width"] == 20.0 and v["height"] == 30.0
    assert v["rectanglelabels"] == ["bird"]
    assert res["score"] == 0.45


def test_roundtrip_pixels_preserved() -> None:
    """COCO → LS → 回读，像素框在整数下无损（对称转换）。"""
    tasks = to_ls_tasks(_review_coco())
    # 模拟人审：直接确认 MD 预标注（result 原样成为 annotation）
    export = [
        {
            "meta": tasks[0]["meta"],
            "data": tasks[0]["data"],
            "annotations": [{"result": tasks[0]["predictions"][0]["result"]}],
        }
    ]
    back = from_ls_export(export)
    assert back["annotations"][0]["bbox"] == [50.0, 20.0, 40.0, 30.0]
    assert back["images"][0]["file_name"] == "inat/999.jpg"
    assert back["categories"][0]["name"] == "bird"


def test_import_respects_human_edit() -> None:
    """人删框 → 导出该任务无 result → 该图 0 框（人判无有效鸟）。"""
    export = [
        {
            "meta": {"file_name": "inat/999.jpg", "width": 200, "height": 100},
            "annotations": [{"result": []}],
        }
    ]
    back = from_ls_export(export)
    assert back["images"][0]["file_name"] == "inat/999.jpg"
    assert back["annotations"] == []


def test_import_skips_cancelled_annotation() -> None:
    """被取消(was_cancelled)的人审不算数。"""
    r = [
        {
            "type": "rectanglelabels",
            "original_width": 200,
            "original_height": 100,
            "value": {"x": 25, "y": 20, "width": 20, "height": 30, "rectanglelabels": ["bird"]},
        }
    ]
    export = [
        {
            "meta": {"file_name": "a.jpg", "width": 200, "height": 100},
            "annotations": [{"was_cancelled": True, "result": r}, {"result": r}],
        }
    ]
    back = from_ls_export(export)
    assert len(back["annotations"]) == 1  # 跳过 cancelled，取第二份
