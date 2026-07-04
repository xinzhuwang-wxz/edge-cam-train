"""检测数据集 → LS 浏览导出（分层抽样 + 多类框转任务，纯函数）。"""

from __future__ import annotations

from edge_cam.contracts.schemas.detection_manifest import (
    FEEDER5_CATEGORIES,
    DetBox,
    DetectionManifest,
    DetImageRecord,
)
from edge_cam.eval.ls_review import (
    label_config,
    records_to_ls_tasks,
    stratified_sample,
)


def _rec(src, cls, split="train"):
    boxes = (
        [] if cls is None else [DetBox(bbox=[10, 20, 30, 40], category_id=FEEDER5_CATEGORIES[cls])]
    )
    return DetImageRecord(
        path=f"{src}/{cls}.jpg", split=split, width=200, height=100, boxes=boxes, source=src
    )


def _mani(recs):
    return DetectionManifest(
        name="t", version="v", categories=dict(FEEDER5_CATEGORIES), records=recs
    )


def test_label_config_has_5_classes() -> None:
    cfg = label_config()
    for c in ("bird", "squirrel", "cat", "person", "other_animal"):
        assert f'value="{c}"' in cfg
    assert "RectangleLabels" in cfg


def test_records_to_ls_multiclass_pct() -> None:
    recs = [_rec("ena24", "squirrel")]
    tasks = records_to_ls_tasks(recs, dict(FEEDER5_CATEGORIES))
    r = tasks[0]["predictions"][0]["result"][0]
    # bbox[10,20,30,40] on 200x100 → x=5%,y=20%,w=15%,h=40%
    assert r["value"]["x"] == 5.0 and r["value"]["y"] == 20.0
    assert r["value"]["rectanglelabels"] == ["squirrel"]  # 类别正确回查
    assert tasks[0]["meta"]["source"] == "ena24" and tasks[0]["meta"]["n_boxes"] == 1


def test_negative_image_no_boxes() -> None:
    tasks = records_to_ls_tasks([_rec("caltech_ct", None)], dict(FEEDER5_CATEGORIES))
    assert tasks[0]["predictions"][0]["result"] == []  # 空帧无框
    assert tasks[0]["meta"]["n_boxes"] == 0


def test_stratified_sample_covers_each_source() -> None:
    recs = [_rec("web", "bird") for _ in range(10)] + [_rec("feeder", "bird") for _ in range(3)]
    got = stratified_sample(_mani(recs), per_source=2)
    srcs = {r.source for r in got}
    assert srcs == {"web", "feeder"}  # 每源都有代表
    assert sum(1 for r in got if r.source == "web") == 2  # 每源封顶 per_source


def test_stratified_sample_deterministic() -> None:
    recs = [_rec("web", "bird") for _ in range(10)]
    recs = [
        DetImageRecord(
            path=f"web/{i}.jpg",
            split="train",
            width=200,
            height=100,
            boxes=[DetBox(bbox=[1, 1, 2, 2], category_id=0)],
            source="web",
        )
        for i in range(10)
    ]
    a = [r.path for r in stratified_sample(_mani(recs), per_source=3)]
    b = [r.path for r in stratified_sample(_mani(recs), per_source=3)]
    assert a == b and len(a) == 3  # 可复现
