"""检测预处理单一真值源（round2 打磨）：训练配置 normalize == 推理 == 规范，不漂移。"""

from __future__ import annotations

from edge_cam.contracts.schemas.detect_preprocess import NANODET_PREPROCESS
from edge_cam.train.detect.nanodet_config import patch_nanodet_config


def _base():
    """最小 NanoDet 模板（含 patch 会碰的字段）；normalize 故意设错值，验证被真值源覆盖。"""
    return {
        "model": {"arch": {"head": {"num_classes": 80}, "aux_head": {"num_classes": 80}}},
        "data": {
            "train": {"pipeline": {"normalize": [[1, 2, 3], [4, 5, 6]]}},  # 错值，应被覆盖
            "val": {"pipeline": {"normalize": [[1, 2, 3], [4, 5, 6]]}},
        },
        "device": {},
        "schedule": {"lr_schedule": {"T_max": 300}},
    }


def test_patch_writes_normalize_from_single_source():
    """训练配置的 normalize 由 NANODET_PREPROCESS 写入（覆盖模板错值）。"""
    cfg = patch_nanodet_config(
        _base(),
        num_classes=5,
        class_names=["bird", "squirrel", "cat", "person", "other_animal"],
        train_img="i",
        train_ann="a",
        val_img="i",
        val_ann="a",
        save_dir="s",
        input_size=416,
    )
    want = [list(NANODET_PREPROCESS.mean_bgr), list(NANODET_PREPROCESS.std_bgr)]
    assert cfg["data"]["train"]["pipeline"]["normalize"] == want
    assert cfg["data"]["val"]["pipeline"]["normalize"] == want
    assert cfg["data"]["train"]["keep_ratio"] == NANODET_PREPROCESS.keep_ratio


def test_inference_reads_same_source():
    """推理侧 _DET_MEAN/_DET_STD 与 OnnxDetector 默认 input_size 均源自 NANODET_PREPROCESS。"""
    import pytest

    from edge_cam.cascade import adapters

    assert adapters._DET_MEAN.tolist() == pytest.approx(list(NANODET_PREPROCESS.mean_bgr))
    assert adapters._DET_STD.tolist() == pytest.approx(list(NANODET_PREPROCESS.std_bgr))
    # OnnxDetector 默认 input_size = 规范（不再散写 416）
    import inspect

    sig = inspect.signature(adapters.OnnxDetector.__init__)
    assert sig.parameters["input_size"].default == NANODET_PREPROCESS.input_size


def test_preprocess_frozen_canonical():
    """规范是 NanoDet 口径且冻结（改预处理只改一处）。"""
    assert NANODET_PREPROCESS.mean_bgr == (103.53, 116.28, 123.675)
    assert NANODET_PREPROCESS.std_bgr == (57.375, 57.12, 58.395)
    assert NANODET_PREPROCESS.keep_ratio is False and NANODET_PREPROCESS.to_bgr is True
