"""跨集合并映射（engineering §6 data/merge_map；plan §C.9 显式映射 + §5.1 非穷尽标注）。

把 COCO / OIV7 的原始标签经**显式映射表**归一到统一大类（映射定义在 detection_classes，
此处 re-export 作为合并入口），并处理两个合并坑：
1. **跨源去重**（同一图在多源/多次拉取中重复）；
2. **OIV7 非穷尽标注**：OIV7 不保证标全图中所有目标实例 → 未标区域**不能当背景负样本**
   （会把真鸟当负例，污染训练，§5.1）。COCO 对其类是穷尽标注，OIV7 不是 → 用 exhaustive
   标志区分，下游负样本挖掘据此跳过非穷尽样本。

纯函数、不依赖 fiftyone（轻量 dataclass），本地可单测；detection_ingest 的 fiftyone 流水复用它。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# re-export：统一大类映射的单一事实来源（plan §C.9）
from edge_cam.data.detection_classes import (
    CoarseClass,
    class_index,
    coco_to_unified,
    oiv7_to_unified,
    source_labels,
)

__all__ = [
    "CoarseClass",
    "class_index",
    "coco_to_unified",
    "oiv7_to_unified",
    "source_labels",
    "UnifiedDetection",
    "MergedSample",
    "is_exhaustive",
    "unified_label",
    "dedup_by_image_id",
    "apply_ignore_policy",
]

# 各源是否对其类目「穷尽标注」（决定能否拿背景当负样本，§5.1）
_EXHAUSTIVE: dict[str, bool] = {
    "coco": True,  # COCO 标全图中其 80 类的所有实例
    "oiv7": False,  # OIV7 非穷尽 → 未标区域不可信为背景
}


@dataclass
class UnifiedDetection:
    """归一后的单框：统一大类标签 + 像素框 + 来源。"""

    label: str
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    source: str = ""


@dataclass
class MergedSample:
    """合并后的单图：图 id + 来源 + 归一检测框 + 是否穷尽标注。"""

    image_id: str
    source: str
    detections: list[UnifiedDetection] = field(default_factory=list)
    exhaustive: bool = True  # False → 下游不从此图挖背景负样本（§5.1）


def is_exhaustive(source: str) -> bool:
    """该源对其类目是否穷尽标注（未知源保守视为非穷尽）。"""
    return _EXHAUSTIVE.get(source, False)


def unified_label(
    source_label: str, source: str, classes: list[CoarseClass] | None = None
) -> str | None:
    """单个源标签 → 统一大类名；不在映射表内返回 None（应丢弃）。"""
    mapping = coco_to_unified(classes) if source == "coco" else oiv7_to_unified(classes)
    return mapping.get(source_label)


def dedup_by_image_id(samples: list[MergedSample]) -> list[MergedSample]:
    """按 (source, image_id) 去重，保留首次出现（确定性、保序）。"""
    seen: set[tuple[str, str]] = set()
    out: list[MergedSample] = []
    for s in samples:
        key = (s.source, s.image_id)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def apply_ignore_policy(sample: MergedSample) -> MergedSample:
    """据来源设置 exhaustive 标志（OIV7 → False，非穷尽，禁背景负样本）。"""
    sample.exhaustive = is_exhaustive(sample.source)
    return sample
