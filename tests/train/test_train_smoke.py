"""端到端训练 smoke（slow）：fast_dev_run + CPU + pretrained=false + 导出 ONNX。

证明「框架在本地 CPU 能跑通」——真训只是把规模/epochs/加速器调大（AutoDL）。"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from edge_cam.data.prep import DataPrepConfig, prepare
from edge_cam.train.classify.module import Classifier
from edge_cam.train.classify.train import run


def _smoke_cfg(manifest_path: Path, output_dir: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "seed": 0,
            "output_dir": str(output_dir),
            "data": {
                "manifest": str(manifest_path),
                "input_size": 64,
                "batch_size": 2,
                "num_workers": 0,
                "degradation_strength": 1.0,
            },
            "model": {"name": "efficientnet_lite0", "pretrained": False},
            "optim": {"lr": 1e-3, "weight_decay": 1e-4, "label_smoothing": 0.1},
            "trainer": {
                "max_epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "fast_dev_run": True,
                "limit_train_batches": 1.0,
                "limit_val_batches": 1.0,
                "log_every_n_steps": 1,
            },
            "export": {"enabled": False, "opset": 13},
        }
    )


@pytest.mark.slow
def test_train_smoke_runs(flat_imagefolder: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    prepare(DataPrepConfig(name="t", root=str(flat_imagefolder), out_path=str(manifest)))
    model = run(_smoke_cfg(manifest, tmp_path / "out"))
    assert isinstance(model, Classifier)


@pytest.mark.slow
def test_train_smoke_exports_onnx(flat_imagefolder: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    prepare(DataPrepConfig(name="t", root=str(flat_imagefolder), out_path=str(manifest)))
    cfg = _smoke_cfg(manifest, tmp_path / "out")
    cfg.export.enabled = True
    cfg.data.input_size = 64
    run(cfg)
    assert (tmp_path / "out" / "efficientnet_lite0_fp32.onnx").exists()
