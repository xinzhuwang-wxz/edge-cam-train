"""stratified_split：确定性、顺序无关、分层、小类兜底、完备互斥。"""

from __future__ import annotations

import pytest

from edge_cam.data.split import stratified_split


def _items(per_class: dict[str, int]) -> list[tuple[str, str]]:
    return [(f"{label}/{i}.jpg", label) for label, n in per_class.items() for i in range(n)]


def test_empty() -> None:
    assert stratified_split([]) == {}


def test_partition_complete_and_disjoint() -> None:
    items = _items({"a": 20, "b": 20})
    out = stratified_split(items, seed=0)
    assert set(out) == {k for k, _ in items}  # 覆盖全部
    assert set(out.values()) <= {"train", "val", "test"}


def test_deterministic() -> None:
    items = _items({"a": 30, "b": 15})
    assert stratified_split(items, seed=7) == stratified_split(items, seed=7)


def test_order_independent() -> None:
    items = _items({"a": 30, "b": 15})
    shuffled = list(reversed(items))
    assert stratified_split(items, seed=1) == stratified_split(shuffled, seed=1)


def test_seed_changes_assignment() -> None:
    items = _items({"a": 40})
    assert stratified_split(items, seed=1) != stratified_split(items, seed=2)


def test_ratios_roughly_hold() -> None:
    items = _items({"a": 100})
    out = stratified_split(items, ratios=(0.7, 0.15, 0.15), seed=0)
    counts = {s: sum(1 for v in out.values() if v == s) for s in ("train", "val", "test")}
    assert counts["train"] == 70
    assert counts["val"] == 15
    assert counts["test"] == 15


def test_min_train_per_class_guaranteed() -> None:
    # 每个类都至少 1 张进 train，即使是单样本类
    items = _items({"a": 1, "b": 2, "c": 5})
    out = stratified_split(items, seed=0)
    for label in ("a", "b", "c"):
        train_n = sum(1 for k, v in out.items() if k.startswith(f"{label}/") and v == "train")
        assert train_n >= 1


def test_stratified_each_class_present_in_train() -> None:
    items = _items({f"c{i}": 10 for i in range(20)})
    out = stratified_split(items, seed=3)
    train_labels = {k.split("/")[0] for k, v in out.items() if v == "train"}
    assert train_labels == {f"c{i}" for i in range(20)}


def test_duplicate_keys_rejected() -> None:
    with pytest.raises(ValueError, match="唯一"):
        stratified_split([("x.jpg", "a"), ("x.jpg", "b")])


def test_zero_ratios_rejected() -> None:
    with pytest.raises(ValueError, match="ratios"):
        stratified_split(_items({"a": 3}), ratios=(0, 0, 0))
