"""分类器微调入口（Hydra 驱动；engineering §2/§7）。

读 manifest → ClassifyDataModule → Classifier → Lightning fit → 导 FP32 ONNX。
- 本地 CPU smoke：trainer.fast_dev_run=true（或 limit_*_batches 小数），accelerator=cpu。
- GPU 真训（AutoDL）：accelerator=gpu，pretrained=true，epochs=80。
- Hydra multirun 跑消融：`-m model.name=efficientnet_lite0,mobilenetv3_large_100 ...`

CLI:
    PYTHONPATH=src python -m edge_cam.train.classify.train \\
        data.manifest=data/processed/birds525/manifest.json trainer.fast_dev_run=true
"""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from edge_cam.contracts.schemas.dataset import DatasetManifest
from edge_cam.train.classify.data import ClassifyDataModule
from edge_cam.train.classify.export import export_onnx, verify_onnx
from edge_cam.train.classify.module import Classifier

CONFIG_DIR = str(Path(__file__).resolve().parents[4] / "configs" / "train" / "classify")


def run(cfg: DictConfig) -> Classifier:
    """执行一次训练（+ 可选导出）；返回训练好的 module。供测试直接调用。"""
    L.seed_everything(cfg.seed, workers=True)
    manifest = DatasetManifest.load(cfg.data.manifest)

    datamodule = ClassifyDataModule(
        manifest=manifest,
        input_size=cfg.data.input_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        degradation_strength=cfg.data.degradation_strength,
    )
    model = Classifier(
        model_name=cfg.model.name,
        num_classes=manifest.num_classes,
        pretrained=cfg.model.pretrained,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        label_smoothing=cfg.optim.label_smoothing,
        max_epochs=cfg.trainer.max_epochs,
    )

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        fast_dev_run=cfg.trainer.fast_dev_run,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        default_root_dir=cfg.output_dir,
    )
    trainer.fit(model, datamodule)

    if cfg.export.enabled:
        onnx_path = Path(cfg.output_dir) / f"{cfg.model.name}_fp32.onnx"
        export_onnx(model, onnx_path, input_size=cfg.data.input_size, opset=cfg.export.opset)
        ok = verify_onnx(onnx_path, model, input_size=cfg.data.input_size)
        print(f"[export] {onnx_path} (onnxruntime 对齐: {ok})")
    return model


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":
    main()
