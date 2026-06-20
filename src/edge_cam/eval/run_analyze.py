"""深度分析入口（Hydra）：选定 ckpt → clean test 的 per-class + 混淆对（plan §B.1）。

消融选出最优骨干后跑：
    python -m edge_cam.eval.run_analyze manifest=... ckpt=... model_name=efficientnet_lite0 \
        input_size=224 output_dir=outputs/analysis +data.data_root=/path/to/raw
产物：analysis.json + analysis.md（最差类表 + 易混淆对表）。
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from edge_cam.contracts.schemas.dataset import DatasetManifest
from edge_cam.eval.analyze import deep_analyze, write_analysis
from edge_cam.train.classify.data import ClassifyDataModule
from edge_cam.train.classify.module import Classifier

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs" / "eval")


def run(cfg: DictConfig) -> tuple[Path, Path]:
    manifest = DatasetManifest.load(cfg.manifest)
    if cfg.get("ckpt"):
        model = Classifier.load_from_checkpoint(cfg.ckpt, map_location="cpu")
    else:
        model = Classifier(cfg.model_name, num_classes=manifest.num_classes, pretrained=False)

    dm = ClassifyDataModule(
        manifest,
        input_size=cfg.input_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.get("num_workers", 4),
        data_root=cfg.data.get("data_root", None) if cfg.get("data") else None,
    )
    idx_to_class = {v: k for k, v in manifest.class_to_idx.items()}
    da = deep_analyze(model, dm.test_dataloader(), idx_to_class)
    json_path, md_path = write_analysis(da, cfg.output_dir, model_name=cfg.get("model_name", ""))
    print(md_path.read_text(encoding="utf-8"))
    return json_path, md_path


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="envelope")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
