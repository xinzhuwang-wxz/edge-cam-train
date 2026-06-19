"""full_eval seam（架构审查 B）：单一编排点产四级包络，fp32_onnx 给定时含 INT8 级。"""

from __future__ import annotations

from pathlib import Path

import pytest

from edge_cam.data.prep import DataPrepConfig, build_manifest
from edge_cam.eval.full_eval import run_full_eval
from edge_cam.train.classify.export import export_onnx
from edge_cam.train.classify.module import Classifier


@pytest.mark.slow
def test_full_eval_with_quant_adds_int8_level(flat_imagefolder: Path, tmp_path: Path) -> None:
    manifest = build_manifest(DataPrepConfig(name="t", root=str(flat_imagefolder), seed=0))
    model = Classifier("efficientnet_lite0", num_classes=manifest.num_classes, pretrained=False)
    fp32 = export_onnx(model, tmp_path / "m.onnx", input_size=64, simplify=False)

    report = run_full_eval(
        model,
        manifest,
        input_size=64,
        batch_size=2,
        num_workers=0,
        fp32_onnx=fp32,  # → full_eval 内部量化出 INT8 级
        output_dir=tmp_path,
    )
    names = {lv.name for lv in report.levels}
    assert "int8_sim" in names  # 量化级由 seam 统一编排出来
    assert {"fp32_val", "field"} <= names


def test_full_eval_without_quant_skips_int8(flat_imagefolder: Path) -> None:
    """不给 fp32_onnx → 无 INT8 级（不触 onnxruntime，快）。"""
    manifest = build_manifest(DataPrepConfig(name="t", root=str(flat_imagefolder), seed=0))
    model = Classifier("efficientnet_lite0", num_classes=manifest.num_classes, pretrained=False)
    report = run_full_eval(model, manifest, input_size=32, batch_size=2, num_workers=0)
    names = {lv.name for lv in report.levels}
    assert "int8_sim" not in names
    assert "fp32_val" in names
