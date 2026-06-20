"""检测数据集 adapter（[[ADR-0003]]/[[ADR-0004]]）。各源 adapter 注册于此。"""

from edge_cam.data.adapters.detect.base import (
    FEEDER5_CATEGORIES,
    DatasetSpec,
    DetectionDatasetAdapter,
    RawSample,
    assemble,
    available_adapters,
    get_adapter_cls,
    register_adapter,
)
from edge_cam.data.adapters.detect.coco_json import CocoJsonAdapter

__all__ = [
    "FEEDER5_CATEGORIES",
    "DatasetSpec",
    "RawSample",
    "DetectionDatasetAdapter",
    "CocoJsonAdapter",
    "assemble",
    "register_adapter",
    "get_adapter_cls",
    "available_adapters",
]
