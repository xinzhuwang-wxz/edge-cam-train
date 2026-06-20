"""数据 ingest adapter 注册表（[[ADR-0003]] #6）：源 → manifest,注册一次 config 切换。

与 TrainerBackend/Quantizer 同款"注册-工厂-config 派发"。每种数据源一个 ingest adapter
(产 DatasetManifest 或 DetectionManifest),新源 register_ingest 即可,下游 caller 不改。

内置两个真 adapter(≥2 = 真 seam):
- "imagefolder"    → prep.prepare → DatasetManifest(分类)
- "coco_detection" → detection_ingest.build_detection_manifest → DetectionManifest(检测)

惰性导入 adapter:取用时才 import,避免拉重依赖(如 fiftyone/torch)。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# name → 惰性工厂(返回 ingest callable)。值是"取 adapter 的函数",非 adapter 本身,以惰性导入。
_INGESTERS: dict[str, Callable[[], Callable[..., Any]]] = {}


def register_ingest(name: str, loader: Callable[[], Callable[..., Any]]) -> None:
    """注册一个 ingest adapter 的惰性 loader。loader() 返回真正的 ingest callable(源→manifest)。"""
    _INGESTERS[name] = loader


def get_ingest(name: str) -> Callable[..., Any]:
    """按名取 ingest adapter(源→manifest)。未知名抛 ValueError(含可选清单)。"""
    try:
        return _INGESTERS[name]()
    except KeyError:
        raise ValueError(
            f"未知 ingest 源 {name!r}；可选：{sorted(_INGESTERS)}（新源 register_ingest 注册）"
        ) from None


def available_ingests() -> list[str]:
    return sorted(_INGESTERS)


def _imagefolder():
    from edge_cam.data.prep import prepare  # DataPrepConfig → DatasetManifest

    return prepare


def _coco_detection():
    from edge_cam.data.detection_ingest import (
        build_detection_manifest,  # COCO 目录 → DetectionManifest
    )

    return build_detection_manifest


register_ingest("imagefolder", _imagefolder)
register_ingest("coco_detection", _coco_detection)
