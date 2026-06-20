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
    # negative_quota=None 隔离"留/丢"判定，不掺限额（限额另测）
    only_unmapped = [RawSample("n.jpg", 10, 10, [("Cow", [0, 0, 1, 1])])]
    build = lambda samples, **kw: _FakeAdapter(  # noqa: E731
        _spec(negative_quota=None, **kw), samples
    ).build_records()
    # 穷尽源:只剩未映射 → 当负样本(0 框)保留
    rec_ex = build(only_unmapped, exhaustive=True)
    assert len(rec_ex) == 1 and rec_ex[0].boxes == []
    # 非穷尽源:可能漏标真目标 → 丢弃,不当负样本
    assert build(only_unmapped, exhaustive=False) == []
    # 显式负样本:无论穷尽与否都保留
    neg = [RawSample("e.jpg", 10, 10, [], is_negative=True)]
    assert len(build(neg, exhaustive=False)) == 1


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


def test_negative_quota_caps_negatives_deterministically() -> None:
    negs = [RawSample(f"n{i}.jpg", 10, 10, [], is_negative=True) for i in range(10)]
    # quota=0 → 不留；None → 全留；N → 留前 N（确定性可复现）
    assert _FakeAdapter(_spec(exhaustive=True, negative_quota=0), negs).build_records() == []
    all_kept = _FakeAdapter(_spec(exhaustive=True, negative_quota=None), negs).build_records()
    assert len(all_kept) == 10
    kept1 = {r.path for r in _FakeAdapter(_spec(negative_quota=3), negs).build_records()}
    kept2 = {r.path for r in _FakeAdapter(_spec(negative_quota=3), negs).build_records()}
    assert len(kept1) == 3 and kept1 == kept2


def test_max_per_class_caps_positive_images() -> None:
    birds = [RawSample(f"b{i}.jpg", 10, 10, [("Bird", [0, 0, 2, 2])]) for i in range(8)]
    recs = _FakeAdapter(_spec(max_per_class=3), birds).build_records()
    assert len(recs) == 3  # bird 类封顶 3 张
    # 多类图：任一类未满即留（Raccoon→other_animal 与 bird 独立计数）
    mixed = [RawSample("m.jpg", 10, 10, [("Bird", [0, 0, 2, 2]), ("Raccoon", [0, 0, 2, 2])])]
    assert len(_FakeAdapter(_spec(max_per_class=0), mixed).build_records()) == 0


def test_registry() -> None:
    register_adapter("fake_reg", _FakeAdapter)
    assert "fake_reg" in available_adapters()
