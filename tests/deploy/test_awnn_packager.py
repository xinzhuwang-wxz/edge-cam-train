"""AWNN 打包器测试——锁 VS861 量化策略（[[ADR-0009]] / docs/detect/04 §4.1）。

只测纯函数 `gen_config`/`write_config`（策略编码）；`pack` 的 docker 编译不在单测内。
"""

from __future__ import annotations

import yaml

from edge_cam.deploy.packager.awnn_packager import (
    FEEDER_MEAN_BGR,
    NANODET_HEAD_FP32,
    AwnnPackager,
    gen_config,
    write_config,
)
from edge_cam.deploy.packager.base import PackagerBackend


def test_default_strategy_is_head_fp32_mixed_precision():
    """VS861 策略核心：检测头 gfl_cls 保 fp32（防 INT8 裁 cls 峰值→丢检出）。"""
    cfg = gen_config("main_416_fp32_logits.onnx", "/data/calib/calibration_datasets.txt")
    wl = cfg["build_conf"]["hybrid_quantization_conf"]["white_list"]
    assert wl == NANODET_HEAD_FP32
    assert len(wl) == 4 and all("gfl_cls" in n for n in wl)


def test_bgr_normalization_baked_into_preprocess():
    """归一化＝BGR mean/norm（board 由 use_npu_preprocess 折进 NPU，喂 0-255 BGR）。"""
    cfg = gen_config("m.onnx", "/data/calib/x.txt")
    pp = cfg["build_conf"]["dataset_conf"][0]["preprocess_conf"][0]
    assert pp["color_space"] == "BGR"
    assert pp["mean"] == FEEDER_MEAN_BGR
    assert pp["tensor_layout"] == "NCHW" and pp["shape"] == [1, 3, 416, 416]
    assert cfg["build_conf"]["use_npu_preprocess"] is True


def test_int8_per_channel_and_output_path():
    cfg = gen_config("feeder.onnx", "/data/calib/x.txt", output_dir="/data/build_out")
    q = cfg["build_conf"]["quantize_conf"]
    assert q["quantized_dtype"] == "symmetric_i8"
    assert q["quantized_method"] == "per-channel"
    assert cfg["general_conf"]["model_names"] == ["feeder.onnx"]
    assert cfg["general_conf"]["output"] == "/data/build_out/feeder"


def test_head_fp32_can_be_disabled_falls_back_to_full_int8():
    """head_fp32_layers=[] → 关混合精度（对照/消融用）。"""
    cfg = gen_config("m.onnx", "/data/calib/x.txt", head_fp32_layers=[])
    assert "hybrid_quantization_conf" not in cfg["build_conf"]


def test_written_config_has_no_tabs(tmp_path):
    """AWNN 硬要求：config 禁用 Tab。"""
    cfg = gen_config("m.onnx", "/data/calib/x.txt")
    p = write_config(cfg, tmp_path / "c.yml")
    text = p.read_text()
    assert "\t" not in text
    assert yaml.safe_load(text) == cfg  # 可回读


def test_packager_conforms_to_backend_protocol():
    assert isinstance(AwnnPackager(), PackagerBackend)
