"""细分类数据集 adapter（[[ADR-0002]] taxonomy / [[ADR-0005]] 许可）。各源 adapter 注册于此。"""

from edge_cam.data.adapters.classify.base import (
    ClassifyDatasetAdapter,
    ClassifyRawSample,
    ClassifySpec,
    assemble,
    available_adapters,
    build_adapter,
    get_adapter_cls,
    normalize_license,
    register_adapter,
)

__all__ = [
    "ClassifySpec",
    "ClassifyRawSample",
    "ClassifyDatasetAdapter",
    "assemble",
    "normalize_license",
    "register_adapter",
    "get_adapter_cls",
    "available_adapters",
    "build_adapter",
]

# 具体数据源 adapter：import 即注册（registry 副作用）。加新源 = 写模块 + 在此 import。
from edge_cam.data.adapters.classify import gbif  # noqa: E402,F401
