"""PTQ 校准集：抽样、退化、dataset.txt 格式（slow，依赖 torch/PIL 变换）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from edge_cam.data.calib import build_calib_set
from edge_cam.data.prep import DataPrepConfig, build_manifest


@pytest.mark.slow
def test_build_calib_set(flat_imagefolder: Path, tmp_path: Path) -> None:
    manifest = build_manifest(DataPrepConfig(name="t", root=str(flat_imagefolder), seed=0))
    dataset_txt = build_calib_set(
        manifest, tmp_path / "calib_out", n=5, input_size=32, degradation_ratio=0.5, seed=0
    )
    assert dataset_txt.exists()
    lines = dataset_txt.read_text(encoding="utf-8").strip().splitlines()
    assert 1 <= len(lines) <= 5
    assert all(line.endswith(" 0") and line.startswith("calib/") for line in lines)
    # 每行对应一张落盘的图
    for line in lines:
        rel = line.split()[0]
        assert (tmp_path / "calib_out" / rel).exists()
