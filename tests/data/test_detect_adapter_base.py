"""检测 DatasetAdapter 抽象（[[ADR-0003]]/[[ADR-0004]]）：映射/负样本/split/组装/路由（fake）。"""

from __future__ import annotations

import pytest

from edge_cam.data.adapters.detect import (
    FEEDER5_CATEGORIES,
    DatasetSpec,
    DetectionDatasetAdapter,
    RawSample,
    assemble,
)
from edge_cam.data.adapters.detect.base import _split_of, available_adapters, register_adapter


class _FakeAdapter(DetectionDatasetAdapter):
    def __init__(self, spec, samples):
        super().__init__(spec)
        self._samples = samples

    def load_raw(self):
        return self._samples


def _spec(**kw):
    base = dict(
        name="fake",
        raw_format="coco_json",
        label_map={"Bird": "bird", "Raccoon": "other_animal"},
        license="CDLA",
        commercial_safe=True,
    )
    base.update(kw)
    return DatasetSpec(**base)


def test_spec_rejects_non5_target() -> None:
    with pytest.raises(ValueError, match="非 5 类"):
        DatasetSpec(
            name="x",
            raw_format="coco_json",
            label_map={"a": "elephant"},
            license="L",
            commercial_safe=True,
        )


def test_build_records_maps_and_drops_unmapped() -> None:
    a = _FakeAdapter(
        _spec(),
        [
            RawSample("img/a.jpg", 100, 80, [("Bird", [1, 2, 3, 4]), ("Cow", [0, 0, 5, 5])]),
        ],
    )
    recs = a.build_records()
    assert len(recs) == 1
    assert len(recs[0].boxes) == 1  # Cow 未映射→丢
    assert recs[0].boxes[0].category_id == FEEDER5_CATEGORIES["bird"]
    assert recs[0].source == "fake" and recs[0].license == "CDLA"


def test_exhaustive_keeps_negative_but_nonexhaustive_drops() -> None:
    only_unmapped = [RawSample("n.jpg", 10, 10, [("Cow", [0, 0, 1, 1])])]
    # 穷尽源:只剩未映射 → 当负样本(0 框)保留
    rec_ex = _FakeAdapter(_spec(exhaustive=True), only_unmapped).build_records()
    assert len(rec_ex) == 1 and rec_ex[0].boxes == []
    # 非穷尽源:可能漏标真目标 → 丢弃,不当负样本
    rec_non = _FakeAdapter(_spec(exhaustive=False), only_unmapped).build_records()
    assert rec_non == []
    # 显式负样本:无论穷尽与否都保留
    neg = [RawSample("e.jpg", 10, 10, [], is_negative=True)]
    assert len(_FakeAdapter(_spec(exhaustive=False), neg).build_records()) == 1


def test_split_deterministic_by_group() -> None:
    # 同 group_key → 同 split(防泄漏);确定性可复现
    assert _split_of("loc-7", "ena24") == _split_of("loc-7", "ena24")
    assert _split_of("loc-7", "ena24") in ("train", "val", "test")
    a = _FakeAdapter(
        _spec(split_unit="location"),
        [
            RawSample(f"img{i}.jpg", 10, 10, [("Bird", [0, 0, 2, 2])], group_key="loc-7")
            for i in range(5)
        ],
    )
    splits = {r.split for r in a.build_records()}
    assert len(splits) == 1  # 同 group 全进同一 split


def test_assemble_routes_by_role_and_commercial() -> None:
    train_ad = _FakeAdapter(
        _spec(name="oiv7", commercial_safe=True, role="train"),
        [
            RawSample(f"t{i}.jpg", 10, 10, [("Bird", [0, 0, 2, 2])], group_key=f"g{i}")
            for i in range(20)
        ],
    )
    feas_ad = _FakeAdapter(
        _spec(name="coco", commercial_safe=False, role="eval_only"),
        [
            RawSample("c.jpg", 10, 10, [("Bird", [0, 0, 2, 2])]),
        ],
    )
    out = assemble([train_ad, feas_ad])
    assert set(out) == {"train", "test", "eval_feasibility"}
    assert out["eval_feasibility"].records[0].source == "coco"  # feasibility 仅入评估
    assert all(r.split != "test" for r in out["train"].records)
    assert all(r.split == "test" for r in out["test"].records)
    assert out["train"].categories == FEEDER5_CATEGORIES


def test_assemble_rejects_noncommercial_train() -> None:
    bad = _FakeAdapter(_spec(name="coco", commercial_safe=False, role="train"), [])
    with pytest.raises(ValueError, match="不可商用数据不得进训练"):
        assemble([bad])


def test_registry() -> None:
    register_adapter("fake_reg", _FakeAdapter)
    assert "fake_reg" in available_adapters()
