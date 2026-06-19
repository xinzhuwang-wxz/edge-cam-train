"""可行性包络 CLI 端到端（slow）：random 权重 + 地域 json → report.json/md + gate。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from edge_cam.data.prep import DataPrepConfig, prepare
from edge_cam.eval.run_envelope import run


@pytest.mark.slow
def test_run_envelope_produces_report(flat_imagefolder: Path, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = prepare(
        DataPrepConfig(name="t", root=str(flat_imagefolder), out_path=str(manifest_path))
    )

    # 地域清单：保留前两个类的 taxon_key
    keys = [r.taxon_key for r in manifest.records][:1]
    region_json = tmp_path / "region.json"
    region_json.write_text(json.dumps(keys), encoding="utf-8")

    cfg = OmegaConf.create(
        {
            "manifest": str(manifest_path),
            "ckpt": None,
            "model_name": "efficientnet_lite0",
            "fp32_onnx": None,
            "regional_json": str(region_json),
            "input_size": 64,
            "batch_size": 2,
            "output_dir": str(tmp_path / "env_out"),
            "gate": {"min_fp32_top1": None},
        }
    )
    json_path, md_path = run(cfg)
    assert json_path.exists() and md_path.exists()
    assert "可行性包络" in md_path.read_text(encoding="utf-8")
    assert "Gate" in md_path.read_text(encoding="utf-8")
