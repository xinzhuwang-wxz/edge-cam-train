"""GbifBirdsAdapter：读合成 index.csv → license 过滤/taxonomy/防泄漏（不联网）。"""

from __future__ import annotations

import csv

from edge_cam.data.adapters.classify import available_adapters, build_adapter


def _write_index(tmp_path, rows):
    p = tmp_path / "index.csv"
    cols = ["path", "scientific_name", "license", "group_key", "lat", "lon", "observed_at"]
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return p


def test_gbif_adapter_reads_index_filters_license(tmp_path) -> None:
    _write_index(
        tmp_path,
        [
            {
                "path": "images/1/a.jpg",
                "scientific_name": "Passer domesticus",
                "license": "CC_BY_4_0",
                "group_key": "obsr1",
                "lat": "1.0",
                "lon": "2.0",
                "observed_at": "2024-05-01",
            },
            {
                "path": "images/1/b.jpg",
                "scientific_name": "Passer domesticus",
                "license": "CC_BY_NC_4_0",
                "group_key": "obsr2",
                "lat": "",
                "lon": "",
                "observed_at": "",
            },  # NC → 丢
            {
                "path": "images/2/c.jpg",
                "scientific_name": "Turdus migratorius",
                "license": "CC0_1_0",
                "group_key": "obsr3",
                "lat": "",
                "lon": "",
                "observed_at": "",
            },
        ],
    )
    ad = build_adapter("gbif_birds", str(tmp_path), source="naturgucker")
    recs = ad.build_records()
    # IdentityTaxonomy：taxon_key = 学名归一（小写）；NC 丢
    assert {r.label for r in recs} == {"passer domesticus", "turdus migratorius"}
    assert {r.license for r in recs} == {"CC-BY", "CC0"}
    assert all(r.source == "naturgucker" for r in recs)


def test_gbif_adapter_max_per_class_and_split(tmp_path) -> None:
    rows = [
        {
            "path": f"images/1/{i}.jpg",
            "scientific_name": "Passer domesticus",
            "license": "CC0_1_0",
            "group_key": f"o{i}",
            "lat": "",
            "lon": "",
            "observed_at": "",
        }
        for i in range(10)
    ]
    _write_index(tmp_path, rows)
    recs = build_adapter("gbif_birds", str(tmp_path), max_per_class=4).build_records()
    assert len(recs) == 4
    assert build_adapter("gbif_birds", str(tmp_path)).spec.split_unit == "observer"


def test_registered() -> None:
    assert "gbif_birds" in available_adapters()
