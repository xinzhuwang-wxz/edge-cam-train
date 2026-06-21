"""classify build：多源 config → 单 manifest（路径按源加前缀、跨源按学名合并、署名清册）。"""

from __future__ import annotations

import csv
import json

from edge_cam.contracts.schemas.dataset import DatasetManifest
from edge_cam.data.adapters.classify.build import ClassifyBuildConfig, SourceEntry, build

_COLS = ["path", "scientific_name", "license", "group_key", "lat", "lon", "observed_at"]


def _write_source(raw_root, name, rows):
    d = raw_root / name
    d.mkdir(parents=True)
    with (d / "index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _COLS})


def _row(path, sci, lic, grp):
    return {"path": path, "scientific_name": sci, "license": lic, "group_key": grp}


def test_build_merges_sources_prefixes_paths_and_filters(tmp_path):
    raw = tmp_path / "classify_raw"
    # 两源共享一个物种（Passer domesticus）→ 应合并为同一类；各含一张 NC（应被过滤）。
    _write_source(
        raw,
        "naturgucker",
        [
            _row("images/1/a.jpg", "Passer domesticus", "CC_BY_4_0", "o1"),
            _row("images/2/b.jpg", "Turdus merula", "CC0_1_0", "o2"),
            _row("images/3/c.jpg", "Corvus corone", "CC_BY_NC_4_0", "o3"),  # NC → 丢
        ],
    )
    _write_source(
        raw,
        "arter",
        [
            _row("images/9/x.jpg", "Passer domesticus", "CC_BY_4_0", "o9"),
            _row("images/8/y.jpg", "Cyanistes caeruleus", "CC_BY_4_0", "o8"),
        ],
    )
    out = raw / "processed"
    cfg = ClassifyBuildConfig(
        raw_root=str(raw),
        out_dir=str(out),
        sources=[SourceEntry(root="naturgucker"), SourceEntry(root="arter")],
    )
    build(cfg)

    m = DatasetManifest.load(out / "manifest.json")
    # NC 丢 → 共 4 张；3 个物种类（Passer 跨源合并）
    assert m.num_samples == 4
    assert m.num_classes == 3
    assert "passer domesticus" in m.class_to_idx
    # 路径按源加了前缀（可移植：root=raw_root + record.path）
    paths = {r.path for r in m.records}
    assert "naturgucker/images/1/a.jpg" in paths
    assert "arter/images/9/x.jpg" in paths
    assert m.root == str(raw)
    # Passer 两源各一张 → 合并类下 2 张
    assert m.class_counts()["passer domesticus"] == 2
    # 无 NC 残留
    assert all(r.license in ("CC-BY", "CC0") for r in m.records)

    # 署名清册逐图 path/source/license
    with (out / "license_manifest.csv").open(encoding="utf-8") as fh:
        lic_rows = list(csv.DictReader(fh))
    assert len(lic_rows) == 4
    assert {r["source"] for r in lic_rows} == {"naturgucker", "arter"}

    # summary 计数
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_samples"] == 4
    assert summary["by_source"] == {"naturgucker": 2, "arter": 2}
    assert summary["by_license"]["CC-BY"] == 3
