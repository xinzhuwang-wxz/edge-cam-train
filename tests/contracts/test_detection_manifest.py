"""检测 manifest 数据脊柱（[[ADR-0003]] C5）：provenance 共享 + to_coco + 蒸馏字段。"""

from __future__ import annotations

import pytest

from edge_cam.contracts.schemas.dataset import SampleRecord, provenance_summary
from edge_cam.contracts.schemas.detection_manifest import (
    DetBox,
    DetectionManifest,
    DetImageRecord,
)


def _manifest() -> DetectionManifest:
    return DetectionManifest(
        name="detection_feeder",
        version="v0",
        categories={"bird": 0, "cat": 1},
        records=[
            DetImageRecord(
                path="train/a.jpg",
                split="train",
                width=640,
                height=480,
                boxes=[DetBox(bbox=[10, 10, 50, 60], category_id=0)],
                source="coco-2017",
                license="CC-BY-4.0",
            ),
            DetImageRecord(
                path="val/b.jpg",
                split="val",
                width=320,
                height=240,
                boxes=[DetBox(bbox=[5, 5, 20, 20], category_id=1)],
                source="oiv7",
            ),
        ],
    )


def test_validator_rejects_unknown_category() -> None:
    with pytest.raises(ValueError, match="category_id"):
        DetectionManifest(
            name="x",
            version="v0",
            categories={"bird": 0},
            records=[
                DetImageRecord(
                    path="p",
                    split="train",
                    width=1,
                    height=1,
                    boxes=[DetBox(bbox=[0, 0, 1, 1], category_id=9)],
                )
            ],
        )


def test_to_coco_split_and_1indexed() -> None:
    coco = _manifest().to_coco("train")
    assert [c["name"] for c in coco["categories"]] == ["bird", "cat"]
    assert coco["categories"][0]["id"] == 1  # COCO 1-indexed
    assert len(coco["images"]) == 1  # 只 train split
    assert coco["annotations"][0]["category_id"] == 1  # bird(0)+1
    assert coco["annotations"][0]["bbox"] == [10, 10, 50, 60]


def test_shared_provenance_summary() -> None:
    ds, lic = _manifest().provenance()
    assert ds == ["coco-2017", "oiv7"]
    assert lic == ["CC-BY-4.0"]  # unknown 被忽略
    assert _manifest().counts_by_split() == {"train": 1, "val": 1, "test": 0}


def test_save_load_roundtrip(tmp_path) -> None:
    p = tmp_path / "det.json"
    _manifest().save(p)
    assert DetectionManifest.load(p).num_classes == 2


def test_distillation_soft_label_field() -> None:
    # 两族 Provenanced 都带可选 soft_label（蒸馏挂同一 schema,默认 None 不影响现有数据）
    r = SampleRecord(path="p", label="bird", split="train", soft_label=[0.1, 0.9])
    assert r.soft_label == [0.1, 0.9]
    assert SampleRecord(path="p", label="bird", split="train").soft_label is None
    di = DetImageRecord(path="p", split="train", width=1, height=1, soft_label=[0.2])
    assert di.soft_label == [0.2]


def test_provenance_summary_ignores_unknown() -> None:
    recs = [SampleRecord(path="p", label="x", split="train")]  # 默认 source/license=unknown
    assert provenance_summary(recs) == ([], [])
