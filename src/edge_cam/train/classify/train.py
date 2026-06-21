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
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from edge_cam.contracts.schemas.dataset import DatasetManifest
from edge_cam.train.classify.data import ClassifyDataModule, inverse_freq_class_weights
from edge_cam.train.classify.export import export_onnx, verify_onnx
from edge_cam.train.classify.module import Classifier

CONFIG_DIR = str(Path(__file__).resolve().parents[4] / "configs" / "train" / "classify")


def build_logger(cfg: DictConfig):
    """可选 aim 实验追踪（track.aim=true 且 aim 可用时）；否则 None → Lightning 用默认。

    惰性 import：未装 [track]（aim）也不报错，只是不追踪。消融对比的可追溯性靠它。
    """
    track = cfg.get("track")
    if not (track and track.get("aim")):
        return None
    try:
        from aim.pytorch_lightning import AimLogger
    except ImportError:
        print("[track] aim 未安装（pip install -e '.[track]'）→ 跳过 aim 追踪")
        return None
    # log_system_params=False：关掉系统信息采集（在该 GPU 云上会 hang 住训练启动，实跑踩到）
    return AimLogger(experiment=cfg.model.name, log_system_params=False)


def build_callbacks(cfg: DictConfig) -> list[Callback]:
    """best-on-val checkpoint + early-stop（按 val 指标，避免导出过拟合末轮）。

    `monitor` 默认 val_top1（max）。fast_dev_run 下 Lightning 自动跳过 checkpoint，无副作用。
    `early_stop.enabled=false` 关早停；`save_top_k` 默认 1（只留最优轮）。"""
    ckpt = cfg.get("checkpoint") or {}
    monitor = ckpt.get("monitor", "val_top1")
    mode = ckpt.get("mode", "max")
    callbacks: list[Callback] = [
        ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            save_top_k=ckpt.get("save_top_k", 1),
            filename="best-{epoch}-{val_top1:.4f}",
        )
    ]
    es = cfg.get("early_stop") or {}
    if es.get("enabled", True):
        callbacks.append(EarlyStopping(monitor=monitor, mode=mode, patience=es.get("patience", 8)))
    return callbacks


def best_checkpoint_path(callbacks: list[Callback]) -> str | None:
    """从 callbacks 取 best ckpt 路径（无/未触发则 None）。"""
    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
            return cb.best_model_path
    return None


def run(cfg: DictConfig) -> Classifier:
    """执行一次训练，返回训练好的 module（**只训练，不导出**，架构审查 C）。

    导出 ONNX 属发布路职责，移到 export_classifier；消融 harness 只要模型对象、不必每格导出。
    训完加载 **best-on-val** 权重返回（而非过拟合末轮），保证 export/评估用最优轮。
    """
    L.seed_everything(cfg.seed, workers=True)
    manifest = DatasetManifest.load(cfg.data.manifest)

    datamodule = ClassifyDataModule(
        manifest=manifest,
        input_size=cfg.data.input_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        degradation_strength=cfg.data.degradation_strength,
        # 换机时覆盖：data.data_root=/path/to/uploaded/raw（留空=用 manifest 记录的 root）
        data_root=cfg.data.get("data_root", None),
        # 检测式裁框增强（优化 A1）：默认 0.7/温和；级联鲁棒训练用 data.crop_scale_min=0.4 等覆盖
        crop_scale_min=cfg.data.get("crop_scale_min", 0.7),
        crop_ratio=tuple(cfg.data.get("crop_ratio", (0.75, 1.333))),
        # 类平衡过采样（治长尾，data.balanced_sampler=true）；与 optim.class_weighted 二选一
        balanced_sampler=cfg.data.get("balanced_sampler", False),
    )
    # 类不均衡 → 反频类权重喂 CE（optim.class_weighted=true）。与 balanced_sampler 二选一即可。
    class_weights = (
        inverse_freq_class_weights(manifest, "train")
        if cfg.optim.get("class_weighted", False)
        else None
    )
    model = Classifier(
        model_name=cfg.model.name,
        num_classes=manifest.num_classes,
        pretrained=cfg.model.pretrained,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        label_smoothing=cfg.optim.label_smoothing,
        max_epochs=cfg.trainer.max_epochs,
        class_weights=class_weights,
    )

    callbacks = build_callbacks(cfg)
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        fast_dev_run=cfg.trainer.fast_dev_run,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=build_logger(cfg),  # 可选 aim 追踪（track.aim=true）
        callbacks=callbacks,
        default_root_dir=cfg.output_dir,
    )
    trainer.fit(model, datamodule)
    # 导出/评估用 best-on-val（非过拟合末轮）：有 best ckpt 则加载回来
    best_cb = next((cb for cb in callbacks if isinstance(cb, ModelCheckpoint)), None)
    if best_cb and best_cb.best_model_path:
        print(f"[best] 加载最优轮权重：{best_cb.best_model_path} (best={best_cb.best_model_score})")
        model = Classifier.load_from_checkpoint(best_cb.best_model_path)
    return model


def export_classifier(model: Classifier, cfg: DictConfig) -> Path | None:
    """发布路：导 FP32 ONNX + onnxruntime 对齐校验（铁律）。cfg.export.enabled=False 则跳过。"""
    if not cfg.export.enabled:
        return None
    onnx_path = Path(cfg.output_dir) / f"{cfg.model.name}_fp32.onnx"
    export_onnx(model, onnx_path, input_size=cfg.data.input_size, opset=cfg.export.opset)
    ok = verify_onnx(onnx_path, model, input_size=cfg.data.input_size)
    print(f"[export] {onnx_path} (onnxruntime 对齐: {ok})")
    return onnx_path


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    model = run(cfg)
    export_classifier(model, cfg)


if __name__ == "__main__":
    main()
