"""NanoDet-Plus 配置生成（engineering §2「fork 锁版包一层」）。

设计边界（engineering §2）：NanoDet 跑在**自己 pin 的 checkout + 自己的环境**（其旧版
pytorch-lightning 与本仓 lightning 2.6 冲突），我们只**生成它的训练 config**（指向本仓
FiftyOne 导出的 COCO-json）并**消费它导出的 ONNX**。不 fork 改其源码。

策略：patch 官方模板（third_party/nanodet/config/nanodet-plus-m_320.yml）而非从零生成
→ 鲁棒。只覆盖 num_classes / class_names / 数据路径 / save_dir / 档位。"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

# 锁定的 NanoDet commit（engineering §2：release 停 2023-03，fork 锁版）
NANODET_PINNED_COMMIT = "be9b4a9001d7f9b6fc89c2df31ae8d428e35b4f0"


def patch_nanodet_config(
    base: dict,
    *,
    num_classes: int,
    class_names: list[str],
    train_img: str,
    train_ann: str,
    val_img: str,
    val_ann: str,
    save_dir: str,
    input_size: int = 320,
    epochs: int = 300,
    gpu_ids: tuple[int, ...] = (0,),
    batch_size: int = 96,
    workers: int = 10,
) -> dict:
    """在官方模板基础上覆盖我们需要的字段，返回新 config dict（不改入参）。"""
    cfg = copy.deepcopy(base)
    cfg["save_dir"] = save_dir

    arch = cfg["model"]["arch"]
    for key in ("head", "aux_head"):  # NanoDetPlus 主头 + 辅助头都要改类数
        if isinstance(arch.get(key), dict) and "num_classes" in arch[key]:
            arch[key]["num_classes"] = num_classes

    for split, img, ann in (("train", train_img, train_ann), ("val", val_img, val_ann)):
        cfg["data"][split]["img_path"] = img
        cfg["data"][split]["ann_path"] = ann
        cfg["data"][split]["input_size"] = [input_size, input_size]

    cfg["device"]["gpu_ids"] = list(gpu_ids)
    cfg["device"]["batchsize_per_gpu"] = batch_size
    cfg["device"]["workers_per_gpu"] = workers

    cfg["schedule"]["total_epochs"] = epochs
    if "lr_schedule" in cfg["schedule"]:
        cfg["schedule"]["lr_schedule"]["T_max"] = epochs

    cfg["class_names"] = list(class_names)
    return cfg


def generate_nanodet_config(
    nanodet_repo: str | Path,
    raw_root: str | Path,
    labels_dir: str | Path,
    class_names: list[str],
    out_path: str | Path,
    *,
    base_config: str = "config/nanodet-plus-m_320.yml",
    input_size: int = 320,
    epochs: int = 300,
    train_labels: str = "train_train.json",
    val_labels: str = "train_val.json",
    **kwargs,
) -> Path:
    """加载 NanoDet 官方模板 → patch 指向**新 manifest 派生 labels** → 存盘。

    数据布局（DatasetAdapter build 产物，取代旧 FiftyOne 导出）：
      - img_path = `raw_root`（labels 的 file_name 已含相对子路径，NanoDet 拼 img_path/file_name）。
      - ann_path = `labels_dir/{train_labels,val_labels}`（write_nanodet_labels 派生的 COCO）。
    """
    nanodet_repo, labels = Path(nanodet_repo), Path(labels_dir)
    base = yaml.safe_load((nanodet_repo / base_config).read_text(encoding="utf-8"))
    cfg = patch_nanodet_config(
        base,
        num_classes=len(class_names),
        class_names=class_names,
        train_img=str(raw_root),
        train_ann=str(labels / train_labels),
        val_img=str(raw_root),
        val_ann=str(labels / val_labels),
        save_dir=str(Path("outputs") / "detect" / "nanodet-plus-m"),
        input_size=input_size,
        epochs=epochs,
        **kwargs,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_path
