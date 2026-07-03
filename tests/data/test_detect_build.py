"""检测组装入口 build：config → adapter → assemble → manifest/labels/署名清册（合成 raw）。"""

from __future__ import annotations

import json

from edge_cam.contracts.schemas.detection_manifest import DetectionManifest
from edge_cam.data.adapters.detect.build import DetectBuildConfig, build


def _write(path, coco):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco), encoding="utf-8")


def _seed_raw(raw_root) -> None:
    ena = {
        "images": [
            {"id": i, "file_name": f"e{i}.jpg", "width": 99, "height": 99} for i in range(12)
        ],
        "annotations": [
            {"id": i, "image_id": i, "category_id": 0, "bbox": [1, 1, 2, 2]} for i in range(12)
        ],
        "categories": [{"id": 0, "name": "Bird"}],
    }
    _write(raw_root / "commercial/ena24/ena24_public.json", ena)
    coco = {
        "images": [{"id": 1, "file_name": "c.jpg", "width": 9, "height": 9}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 16, "bbox": [1, 1, 2, 2]}],
        "categories": [{"id": 16, "name": "bird"}],
    }
    _write(raw_root / "feasibility/coco/annotations/instances_val2017.json", coco)


def test_build_writes_manifests_labels_and_license(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    out = tmp_path / "processed"
    _seed_raw(raw_root)
    cfg = DetectBuildConfig(
        raw_root=str(raw_root),
        out_dir=str(out),
        datasets={"ena24": {"negative_quota": 0}, "coco2017": {}},
    )
    build(cfg)

    # 三份 manifest 落盘(JSONL + .meta.json sidecar，ADR-0006 D5)且可加载
    for key in ("train", "test", "eval_feasibility"):
        assert (out / f"manifest_{key}.jsonl").exists()
        assert (out / f"manifest_{key}.meta.json").exists()
        m = DetectionManifest.load(out / f"manifest_{key}.jsonl")
        assert m.root == str(raw_root)
    # ena24 进 train/test（可商用），coco2017 仅 eval_feasibility
    feas = DetectionManifest.load(out / "manifest_eval_feasibility.jsonl")
    assert feas.records and all(r.source == "coco2017" for r in feas.records)
    train = DetectionManifest.load(out / "manifest_train.jsonl")
    assert all(r.source == "ena24" for r in train.records)

    # NanoDet labels 至少有 train split（COCO dict 结构）
    train_labels = json.loads((out / "labels" / "train_train.json").read_text())
    assert {"images", "annotations", "categories"} <= set(train_labels)

    # 署名清册含训练源、不含可行性源
    csv_text = (out / "license_manifest.csv").read_text(encoding="utf-8")
    assert "ena24" in csv_text and "CDLA-Permissive" in csv_text
    assert "coco2017" not in csv_text  # 可行性不进训练 → 不入署名清册
    # ADR-0006 D4：署名清册扩到逐图 author/URL/media_id/sha256（兑现 CC-BY，§4）
    header = csv_text.splitlines()[0]
    assert header == "path,source,license,author,original_url,source_media_id,asset_sha256"

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) == {"train", "test", "eval_feasibility"}


def test_config_from_yaml_roundtrip(tmp_path) -> None:
    import yaml

    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.safe_dump(
            {"raw_root": "r", "out_dir": "o", "datasets": {"ena24": {"negative_quota": 0}}}
        ),
        encoding="utf-8",
    )
    cfg = DetectBuildConfig.from_yaml(p)
    assert cfg.raw_root == "r" and cfg.datasets["ena24"]["negative_quota"] == 0
