"""消融 harness：网格展开（纯）+ 结果导出（纯）+ runner 端到端（slow）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from edge_cam.eval.ablation.grid import expand_grid, label_for
from edge_cam.eval.ablation.runner import write_results


def test_expand_empty() -> None:
    assert expand_grid({}) == [{}]


def test_expand_cartesian_product_ordered() -> None:
    grid = expand_grid({"model.name": ["a", "b"], "data.input_size": [192, 224]})
    assert grid == [
        {"model.name": "a", "data.input_size": 192},
        {"model.name": "a", "data.input_size": 224},
        {"model.name": "b", "data.input_size": 192},
        {"model.name": "b", "data.input_size": 224},
    ]


def test_label_for() -> None:
    assert label_for({}) == "baseline"
    assert label_for({"model.name": "x", "data.input_size": 224}) == "name=x input_size=224"


def test_write_results(tmp_path: Path) -> None:
    rows = [
        {"label": "name=a", "model.name": "a", "fp32_val_top1": 0.9},
        {"label": "name=b", "model.name": "b", "fp32_val_top1": 0.8, "field_top1": 0.6},
    ]
    csv_path, md_path = write_results(rows, tmp_path)
    assert csv_path.exists() and md_path.exists()
    md = md_path.read_text(encoding="utf-8")
    assert "消融实验总表" in md
    assert "field_top1" in md  # 列并集（第二行独有列也在表头）


@pytest.mark.slow
def test_run_ablation_smoke(flat_imagefolder: Path, tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    from edge_cam.data.prep import DataPrepConfig, prepare
    from edge_cam.eval.ablation.runner import run_ablation

    manifest_path = tmp_path / "manifest.json"
    manifest = prepare(
        DataPrepConfig(name="t", root=str(flat_imagefolder), out_path=str(manifest_path))
    )
    base = OmegaConf.create(
        {
            "seed": 0,
            "output_dir": str(tmp_path / "out"),
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
    rows = run_ablation(base, {"data.input_size": [64]}, manifest)
    assert len(rows) == 1
    assert "fp32_val_top1" in rows[0]
    assert "field_top1" in rows[0]
