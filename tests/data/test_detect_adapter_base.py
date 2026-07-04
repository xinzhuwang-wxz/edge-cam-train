"""检测 DatasetAdapter 抽象（[[ADR-0003]]/[[ADR-0004]]）：映射/负样本/split/组装/路由（fake）。"""

from __future__ import annotations

import hashlib
import json

import pytest

from edge_cam.data.adapters.detect import (
    FEEDER5_CATEGORIES,
    AcquireSpec,
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


def test_max_per_class_dict_caps_only_named_class() -> None:
    # dict 上限：只压 other_animal，bird 不限 → bird 全留、other_animal 封顶
    birds = [RawSample(f"b{i}.jpg", 10, 10, [("Bird", [0, 0, 2, 2])]) for i in range(8)]
    others = [RawSample(f"o{i}.jpg", 10, 10, [("Raccoon", [0, 0, 2, 2])]) for i in range(8)]
    recs = _FakeAdapter(_spec(max_per_class={"other_animal": 3}), birds + others).build_records()
    by_cls = {c: 0 for c in ("bird", "other_animal")}
    for r in recs:
        for b in r.boxes:
            if b.category_id == FEEDER5_CATEGORIES["bird"]:
                by_cls["bird"] += 1
            elif b.category_id == FEEDER5_CATEGORIES["other_animal"]:
                by_cls["other_animal"] += 1
    assert by_cls["bird"] == 8 and by_cls["other_animal"] == 3  # bird 不限、other_animal 封顶 3


def test_max_per_class_dict_rejects_bad_key() -> None:
    with pytest.raises(ValueError, match="非 5 类键"):
        DatasetSpec(
            name="x",
            raw_format="c",
            label_map={"Bird": "bird"},
            license="L",
            commercial_safe=True,
            max_per_class={"elephant": 5},
        )


def test_split_ratios_configurable() -> None:
    s = [
        RawSample(f"r{i}.jpg", 10, 10, [("Bird", [0, 0, 2, 2])], group_key=f"g{i}")
        for i in range(20)
    ]
    all_train = _FakeAdapter(_spec(split_ratios=(1.0, 0.0, 0.0)), s).build_records()
    assert all(r.split == "train" for r in all_train)
    all_test = _FakeAdapter(_spec(split_ratios=(0.0, 0.0, 1.0)), s).build_records()
    assert all(r.split == "test" for r in all_test)


def test_registry() -> None:
    register_adapter("fake_reg", _FakeAdapter)
    assert "fake_reg" in available_adapters()


def test_attribution_flows_to_record_and_boxes() -> None:
    """逐样本 attribution + 框级 label_provenance 从 RawSample 流到 DetImageRecord/DetBox
    （ADR-0006 D4/D7：兑现 CC-BY 逐图署名 + 框来源透明）。"""
    a = _FakeAdapter(
        _spec(),
        [
            RawSample(
                "img/a.jpg",
                100,
                80,
                [("Bird", [1, 2, 3, 4])],
                author="Jane Doe",
                original_url="https://inat.example/photo/9",
                source_media_id="obs-42",
                asset_sha256="deadbeef",
                label_provenance="md_human_verified",
            ),
        ],
    )
    r = a.build_records()[0]
    assert r.author == "Jane Doe"
    assert r.original_url == "https://inat.example/photo/9"
    assert r.source_media_id == "obs-42"
    assert r.asset_sha256 == "deadbeef"
    assert r.boxes[0].label_provenance == "md_human_verified"


def test_attribution_defaults_backward_compat() -> None:
    """不给 attribution → 字段默认空 / 框 label_provenance=gt（向后兼容，旧 RawSample 不破）。"""
    r = _FakeAdapter(
        _spec(), [RawSample("b.jpg", 10, 10, [("Bird", [0, 0, 2, 2])])]
    ).build_records()[0]
    assert r.author is None and r.original_url is None
    assert r.source_media_id is None and r.asset_sha256 is None
    assert r.boxes[0].label_provenance == "gt"


# --- acquire() seam（ADR-0006 D2/D3）---


def test_acquire_spec_rejects_bad_method() -> None:
    with pytest.raises(ValueError, match="method 非法"):
        AcquireSpec(method="ftp")


def test_acquire_no_spec_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="未声明 acquire"):
        _FakeAdapter(_spec(), []).acquire(tmp_path, now="t")


def test_acquire_manual_verifies_and_writes_receipt(tmp_path) -> None:
    """manual：raw 就位 + sha256 匹配 → 落 _acquire.json 收据（可复现可审计）。"""
    dest = tmp_path / "commercial" / "fake"  # commercial（commercial_safe=True）/<name>
    dest.mkdir(parents=True)
    (dest / "data.zip").write_bytes(b"payload")
    sha = hashlib.sha256(b"payload").hexdigest()
    a = _FakeAdapter(
        _spec(
            acquire=AcquireSpec(
                method="manual",
                urls=["http://x/data.zip"],
                version="v1",
                archive_sha256={"data.zip": sha},
            )
        ),
        [],
    )
    receipt = a.acquire(tmp_path, now="2026-07-04T00:00:00")
    assert receipt.source == "fake" and receipt.method == "manual"
    got = json.loads((dest / "_acquire.json").read_text(encoding="utf-8"))
    assert got["archive_sha256"]["data.zip"] == sha
    assert got["downloaded_at"] == "2026-07-04T00:00:00"
    assert got["version"] == "v1"


def test_acquire_manual_missing_raises_actionable(tmp_path) -> None:
    """manual 源缺文件 → 抛可执行错误（含下载 URL），不静默放行。"""
    a = _FakeAdapter(
        _spec(
            acquire=AcquireSpec(
                method="manual", urls=["http://x/z.zip"], archive_sha256={"z.zip": "abc"}
            )
        ),
        [],
    )
    with pytest.raises(FileNotFoundError, match="请获取"):
        a.acquire(tmp_path, now="t")


def test_acquire_manual_checksum_mismatch_raises(tmp_path) -> None:
    dest = tmp_path / "commercial" / "fake"
    dest.mkdir(parents=True)
    (dest / "data.zip").write_bytes(b"tampered")
    a = _FakeAdapter(
        _spec(acquire=AcquireSpec(method="manual", archive_sha256={"data.zip": "0" * 64})),
        [],
    )
    with pytest.raises(ValueError, match="sha256 不符"):
        a.acquire(tmp_path, now="t")


def test_acquire_nonmanual_without_fetch_override_raises(tmp_path) -> None:
    """非 manual 源、raw 未就位 → 走 _fetch，base 未覆写即报错（提示 adapter 需实现下载）。"""
    a = _FakeAdapter(
        _spec(
            acquire=AcquireSpec(
                method="s3_direct", urls=["s3://x"], archive_sha256={"x.zip": "abc"}
            )
        ),
        [],
    )
    with pytest.raises(NotImplementedError, match="_fetch"):
        a.acquire(tmp_path, now="t")


def test_acquire_empty_checksum_still_fetches(tmp_path) -> None:
    """修 bug：动态源（s3/roboflow/inat）archive_sha256 为空 → **必 _fetch**（此前空真 _checksums_ok
    致误跳、0 图）。"""
    calls = []

    class _FetchAdapter(_FakeAdapter):
        def _fetch(self, dest):
            calls.append(dest)

    a = _FetchAdapter(
        _spec(acquire=AcquireSpec(method="s3_direct", urls=["s3://x"], archive_sha256={})), []
    )
    a.acquire(tmp_path, now="t")
    assert len(calls) == 1  # _fetch 被调（不再空真跳过）


def test_acquire_valid_checksum_skips_fetch(tmp_path) -> None:
    """有声明 checksum 且全通过 → 幂等跳 _fetch（不重下）。"""
    dest = tmp_path / "commercial" / "fake"
    dest.mkdir(parents=True)
    (dest / "f.bin").write_bytes(b"x")
    sha = hashlib.sha256(b"x").hexdigest()
    calls = []

    class _FetchAdapter(_FakeAdapter):
        def _fetch(self, dest):
            calls.append(dest)

    a = _FetchAdapter(
        _spec(acquire=AcquireSpec(method="s3_direct", archive_sha256={"f.bin": sha})), []
    )
    a.acquire(tmp_path, now="t")
    assert not calls  # checksum 通过 → 跳 fetch
