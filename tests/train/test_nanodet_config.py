"""NanoDet config patcher：覆盖 num_classes/数据路径/class_names/档位（纯函数）。"""

from __future__ import annotations

from pathlib import Path

import yaml

from edge_cam.train.detect.nanodet_config import generate_nanodet_config, patch_nanodet_config

# 精简的官方模板骨架（结构对齐 nanodet-plus-m_320.yml）
_BASE = {
    "save_dir": "workspace/x",
    "model": {
        "arch": {
            "name": "NanoDetPlus",
            "head": {"name": "NanoDetPlusHead", "num_classes": 80},
            "aux_head": {"name": "SimpleConvHead", "num_classes": 80},
        }
    },
    "data": {
        "train": {
            "name": "CocoDataset",
            "img_path": "coco/train2017",
            "ann_path": "x.json",
            "input_size": [320, 320],
        },
        "val": {
            "name": "CocoDataset",
            "img_path": "coco/val2017",
            "ann_path": "y.json",
            "input_size": [320, 320],
        },
    },
    "device": {"gpu_ids": [0], "batchsize_per_gpu": 96, "workers_per_gpu": 10},
    "schedule": {"total_epochs": 300, "lr_schedule": {"name": "CosineAnnealingLR", "T_max": 300}},
    "class_names": ["person", "bird"],
}


def test_patch_overrides_num_classes_both_heads() -> None:
    cfg = patch_nanodet_config(
        _BASE,
        num_classes=11,
        class_names=["bird", "squirrel"],
        train_img="d/train/data",
        train_ann="d/train/labels.json",
        val_img="d/validation/data",
        val_ann="d/validation/labels.json",
        save_dir="out/detect",
        input_size=416,
        epochs=120,
        load_model="weights/nanodet/coco_pretrained.ckpt",
    )
    assert cfg["model"]["arch"]["head"]["num_classes"] == 11
    assert cfg["model"]["arch"]["aux_head"]["num_classes"] == 11
    assert cfg["data"]["train"]["ann_path"] == "d/train/labels.json"
    assert cfg["data"]["val"]["input_size"] == [416, 416]
    assert cfg["schedule"]["total_epochs"] == 120
    assert cfg["schedule"]["lr_schedule"]["T_max"] == 120
    assert cfg["schedule"]["load_model"] == "weights/nanodet/coco_pretrained.ckpt"  # 微调起点
    assert cfg["class_names"] == ["bird", "squirrel"]


def test_patch_does_not_mutate_input() -> None:
    patch_nanodet_config(
        _BASE,
        num_classes=11,
        class_names=["bird"],
        train_img="a",
        train_ann="b",
        val_img="c",
        val_ann="d",
        save_dir="e",
    )
    assert _BASE["model"]["arch"]["head"]["num_classes"] == 80  # 原 dict 不变


def test_generate_from_template_file(tmp_path: Path) -> None:
    repo = tmp_path / "nanodet"
    (repo / "config").mkdir(parents=True)
    (repo / "config" / "nanodet-plus-m_320.yml").write_text(yaml.safe_dump(_BASE), encoding="utf-8")
    out = generate_nanodet_config(
        repo,
        "data/raw/detect",
        "data/processed/detect/labels",
        ["bird", "cat", "dog"],
        tmp_path / "gen.yml",
    )
    cfg = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert cfg["model"]["arch"]["head"]["num_classes"] == 3
    # img_path=raw_root（file_name 自带子路径）；ann 指向 manifest 派生 labels
    assert cfg["data"]["train"]["img_path"] == "data/raw/detect"
    assert cfg["data"]["train"]["ann_path"].endswith("labels/train_train.json")
    assert cfg["data"]["val"]["ann_path"].endswith("labels/train_val.json")
