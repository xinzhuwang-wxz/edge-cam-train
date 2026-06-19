"""FiftyOne 拉 COCO + OIV7 feeder-cam 子集 → 统一大类 → 导 COCO-json（NanoDet 吃）。

engineering §5.5 流水 1。fiftyone 惰性导入（框架不强依赖它）。

⚠️ 许可（plan §C.1）：OIV7 逐图 CC-BY（强制署名）；COCO 图片版权杂 → 当前**可行性优先、
不进商用权重**；署名清册随产物为后续（TODO）。OIV7 非穷尽标注 → 漏标当背景会污染负样本
（plan §5.1）；当前先保留 target 检测，ignore-region 精化留后续。"""

from __future__ import annotations

import json
from pathlib import Path

from edge_cam.data.detection_classes import (
    DetectionDataConfig,
    class_index,
    coco_to_unified,
    oiv7_to_unified,
    source_labels,
)


def _detections_field(sample) -> str | None:
    """找样本里第一个 Detections 字段名（COCO 用 ground_truth，OIV7 用 detections）。"""
    import fiftyone as fo

    for name, value in sample.iter_fields():
        if isinstance(value, fo.Detections):
            return name
    return None


def _relabel_to_unified(
    dataset, mapping: dict[str, str], target_field: str = "ground_truth"
) -> None:
    """把检测标签 remap 到统一大类、丢弃非 target，落到统一字段 target_field。"""
    import fiftyone as fo

    for sample in dataset.iter_samples(autosave=True, progress=True):
        field = _detections_field(sample)
        dets = sample[field] if field else None
        kept = []
        if dets is not None:
            for d in dets.detections:
                unified = mapping.get(d.label)
                if unified is not None:
                    d.label = unified
                    kept.append(d)
        sample[target_field] = fo.Detections(detections=kept)


def build_detection_dataset(config: DetectionDataConfig) -> Path:
    """下载 → 合并 → 导 COCO-json，每个 split 一个目录；返回输出根目录。"""
    import fiftyone as fo
    import fiftyone.zoo as foz

    selected = config.selected()
    coco_map = coco_to_unified(selected)
    oiv7_map = oiv7_to_unified(selected)
    coco_labels = source_labels("coco", selected)
    oiv7_labels = source_labels("oiv7", selected)
    unified_classes = list(class_index(selected).keys())

    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}

    for split in config.splits:
        merged = fo.Dataset()
        if coco_labels:
            ds = foz.load_zoo_dataset(
                "coco-2017",
                split=split,
                label_types=["detections"],
                classes=coco_labels,
                max_samples=config.max_per_class * len(coco_labels),
            )
            _relabel_to_unified(ds, coco_map)
            merged.add_samples(ds)
        if oiv7_labels:
            ds = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                label_types=["detections"],
                classes=oiv7_labels,
                max_samples=config.max_per_class * len(oiv7_labels),
            )
            _relabel_to_unified(ds, oiv7_map)
            merged.add_samples(ds)

        merged.export(
            export_dir=str(out / split),
            dataset_type=fo.types.COCODetectionDataset,
            label_field="ground_truth",
            classes=unified_classes,
        )
        summary[split] = merged.count()
        merged.delete()

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[detection] exported → {out} : {summary}")
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="拉取并合并 feeder-cam 检测数据集")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    build_detection_dataset(DetectionDataConfig(**raw))


if __name__ == "__main__":
    main()
