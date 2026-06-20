"""检测管线承重(#13):COCO 目录 → DetectionManifest(免 FiftyOne,合成数据)。"""

from __future__ import annotations

import json

from edge_cam.data.detection_ingest import build_detection_manifest


def _write_split(d, split_dir: str, n_imgs: int) -> None:
    p = d / split_dir
    (p / "data").mkdir(parents=True)
    coco = {
        "images": [
            {"id": i, "file_name": f"img{i}.jpg", "width": 100, "height": 80} for i in range(n_imgs)
        ],
        "annotations": [
            {"id": i, "image_id": i, "category_id": 1, "bbox": [1, 2, 3, 4]} for i in range(n_imgs)
        ],
        "categories": [{"id": 1, "name": "bird"}, {"id": 2, "name": "cat"}],
    }
    (p / "labels.json").write_text(json.dumps(coco), encoding="utf-8")


def test_build_manifest_from_coco_dir(tmp_path) -> None:
    _write_split(tmp_path, "train", 3)
    _write_split(tmp_path, "validation", 2)
    m = build_detection_manifest(tmp_path, name="feeder", source="coco", license="mixed")
    assert m.counts_by_split() == {"train": 3, "val": 2, "test": 0}  # validation→val
    assert m.categories == {"bird": 0, "cat": 1}
    # 路径补了 split 子目录 + data/,可经 root 解析
    r = m.records[0]
    assert r.path.startswith("train/data/") and r.split == "train"
    assert (tmp_path / r.path).parent.name == "data"
    assert m.provenance() == (["coco"], ["mixed"])
    # 承重闭环:manifest → write_nanodet_labels(NanoDet 消费)
    out = m.write_nanodet_labels("train", tmp_path / "derived_labels.json")
    assert len(json.loads(out.read_text())["images"]) == 3
