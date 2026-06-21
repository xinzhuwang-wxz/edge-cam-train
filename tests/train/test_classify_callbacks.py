"""classify 训练回调：best-on-val checkpoint + early-stop（按 config 装配）。"""

from __future__ import annotations

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import OmegaConf

from edge_cam.train.classify.train import best_checkpoint_path, build_callbacks


def test_build_callbacks_default_has_checkpoint_and_earlystop():
    cfg = OmegaConf.create(
        {
            "checkpoint": {"monitor": "val_top1", "mode": "max", "save_top_k": 1},
            "early_stop": {"enabled": True, "patience": 8},
        }
    )
    cbs = build_callbacks(cfg)
    ckpt = next(c for c in cbs if isinstance(c, ModelCheckpoint))
    es = next(c for c in cbs if isinstance(c, EarlyStopping))
    assert ckpt.monitor == "val_top1" and ckpt.mode == "max" and ckpt.save_top_k == 1
    assert es.monitor == "val_top1" and es.patience == 8


def test_early_stop_can_be_disabled():
    cfg = OmegaConf.create({"early_stop": {"enabled": False}})
    cbs = build_callbacks(cfg)
    assert any(isinstance(c, ModelCheckpoint) for c in cbs)
    assert not any(isinstance(c, EarlyStopping) for c in cbs)


def test_build_callbacks_defaults_when_section_missing():
    # 旧 config 无 checkpoint/early_stop 段 → 用默认（仍装 best ckpt + early-stop）
    cbs = build_callbacks(OmegaConf.create({}))
    ckpt = next(c for c in cbs if isinstance(c, ModelCheckpoint))
    assert ckpt.monitor == "val_top1"
    assert any(isinstance(c, EarlyStopping) for c in cbs)


def test_best_checkpoint_path_none_when_not_triggered():
    cbs = build_callbacks(OmegaConf.create({}))
    # 未 fit → best_model_path 为空 → None
    assert best_checkpoint_path(cbs) is None
