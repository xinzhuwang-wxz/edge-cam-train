"""细分类 ClassifyDatasetAdapter 抽象：license 过滤/taxonomy/限额/split/组装（fake，快）。"""

from __future__ import annotations

import pytest

from edge_cam.data.adapters.classify import (
    ClassifyDatasetAdapter,
    ClassifyRawSample,
    ClassifySpec,
    assemble,
    available_adapters,
    normalize_license,
    register_adapter,
)
from edge_cam.data.taxonomy import EbirdTaxonomy

_TAX = EbirdTaxonomy({"House Sparrow": "houspa", "American Robin": "amerob"}, version="t")


class _FakeAdapter(ClassifyDatasetAdapter):
    def __init__(self, spec, samples):
        super().__init__(spec)
        self._samples = samples

    def load_raw(self):
        return self._samples


def _spec(**kw):
    base = dict(name="fake", source="fake", raw_format="x", taxonomy=_TAX)
    base.update(kw)
    return ClassifySpec(**base)


def test_normalize_license() -> None:
    assert normalize_license("CC0") == "CC0"
    assert normalize_license("http://creativecommons.org/publicdomain/zero/1.0/") == "CC0"
    assert normalize_license("CC-BY 4.0") == "CC-BY"
    assert normalize_license("CC_BY_4_0") == "CC-BY"
    assert normalize_license("http://creativecommons.org/licenses/by-nc/4.0/") == "CC-BY-NC"
    assert normalize_license("CC-BY-NC-SA") == "CC-BY-NC-SA"
    assert normalize_license("All rights reserved") == "other"


def test_build_filters_license_and_maps_taxon() -> None:
    samples = [
        ClassifyRawSample("a.jpg", "House Sparrow", "CC-BY", group_key="o1"),
        ClassifyRawSample("b.jpg", "House Sparrow", "CC-BY-NC", group_key="o2"),  # NC → 丢
        ClassifyRawSample("c.jpg", "Unknown Bird", "CC0", group_key="o3"),  # 未映射 → 丢
        ClassifyRawSample("d.jpg", "American Robin", "CC0", group_key="o4"),
    ]
    recs = _FakeAdapter(_spec(), samples).build_records()
    assert {r.label for r in recs} == {"houspa", "amerob"}  # 仅 CC0/CC-BY + 已映射
    assert all(r.taxon_key == r.label and r.source == "fake" for r in recs)
    assert {r.license for r in recs} == {"CC-BY", "CC0"}


def test_split_deterministic_by_group() -> None:
    # 同 observation → 同 split（防泄漏）
    samples = [
        ClassifyRawSample(f"x{i}.jpg", "House Sparrow", "CC0", group_key="obs-7") for i in range(6)
    ]
    splits = {
        r.split for r in _FakeAdapter(_spec(split_unit="observation"), samples).build_records()
    }
    assert len(splits) == 1


def test_max_per_class_caps() -> None:
    samples = [
        ClassifyRawSample(f"s{i}.jpg", "House Sparrow", "CC0", group_key=f"o{i}") for i in range(10)
    ]
    recs = _FakeAdapter(_spec(max_per_class=4), samples).build_records()
    assert len(recs) == 4 and all(r.label == "houspa" for r in recs)


def test_assemble_builds_manifest_from_taxon_keys() -> None:
    a = _FakeAdapter(
        _spec(name="s1", source="s1"),
        [
            ClassifyRawSample("a.jpg", "House Sparrow", "CC0", group_key="o1"),
            ClassifyRawSample("b.jpg", "American Robin", "CC-BY", group_key="o2"),
        ],
    )
    eval_only = _FakeAdapter(
        _spec(name="s2", source="s2", role="eval_only"),
        [
            ClassifyRawSample("z.jpg", "House Sparrow", "CC0", group_key="o9"),
        ],
    )
    m = assemble([a, eval_only])
    assert set(m.class_to_idx) == {"amerob", "houspa"}  # eval_only 不进 train manifest
    assert all(r.source == "s1" for r in m.records)
    assert m.num_classes == 2


def test_spec_rejects_bad_role_split() -> None:
    with pytest.raises(ValueError, match="role"):
        _spec(role="bogus")
    with pytest.raises(ValueError, match="split_unit"):
        _spec(split_unit="bogus")


def test_registry() -> None:
    register_adapter("fake_cls", _FakeAdapter)
    assert "fake_cls" in available_adapters()
