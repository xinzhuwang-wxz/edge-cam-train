"""端到端可行性包络（slow）：训练好的小模型 → INT8 量化 → 四级报告。

证明评估机制本地 CPU 跑通；真实数字待 GPU 真训后产出。"""

from __future__ import annotations

from pathlib import Path

import pytest

from edge_cam.data.prep import DataPrepConfig, build_manifest
from edge_cam.eval.envelope import build_envelope
from edge_cam.eval.regional import RegionalMask
from edge_cam.train.classify.export import export_onnx
from edge_cam.train.classify.module import Classifier


@pytest.mark.slow
def test_envelope_all_levels(flat_imagefolder: Path, tmp_path: Path) -> None:
    manifest = build_manifest(DataPrepConfig(name="t", root=str(flat_imagefolder), seed=0))
    model = Classifier("efficientnet_lite0", num_classes=manifest.num_classes, pretrained=False)

    # FP32 ONNX → INT8 量化（ORT-QDQ）
    from edge_cam.eval.quant_estimate import quantize_int8
    from edge_cam.train.classify.data import ClassifyDataModule

    fp32 = export_onnx(model, tmp_path / "m.onnx", input_size=64, simplify=False)
    dm = ClassifyDataModule(manifest, input_size=64, batch_size=2, num_workers=0)
    int8 = quantize_int8(fp32, dm.train_dataloader(), tmp_path / "m.int8.onnx", max_calib_samples=6)

    # 地域 mask：只留前 2 类
    keep = set(list(manifest.class_to_idx.values())[:2])
    mask = RegionalMask(keep, num_classes=manifest.num_classes)

    report = build_envelope(
        model,
        manifest,
        input_size=64,
        batch_size=2,
        num_workers=0,
        int8_onnx=int8,
        regional_mask=mask,
    )
    names = {lv.name for lv in report.levels}
    assert names == {"fp32_val", "int8_sim", "field", "regional"}
    assert all(0.0 <= lv.top1 <= 1.0 for lv in report.levels)
    assert "可行性包络" in report.to_markdown()
