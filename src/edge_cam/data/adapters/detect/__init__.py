"""检测数据集 adapter（[[ADR-0003]]/[[ADR-0004]]）。各源 adapter 注册于此。"""

from edge_cam.data.adapters.detect.base import (
    FEEDER5_CATEGORIES,
    AcquireReceipt,
    AcquireSpec,
    DatasetSpec,
    DetectionDatasetAdapter,
    RawSample,
    assemble,
    available_adapters,
    build_adapter,
    get_adapter_cls,
    register_adapter,
)
from edge_cam.data.adapters.detect.coco_json import CocoJsonAdapter

__all__ = [
    "FEEDER5_CATEGORIES",
    "AcquireSpec",
    "AcquireReceipt",
    "DatasetSpec",
    "RawSample",
    "DetectionDatasetAdapter",
    "CocoJsonAdapter",
    "assemble",
    "register_adapter",
    "get_adapter_cls",
    "available_adapters",
    "build_adapter",
]

# 具体数据集 adapter：import 即注册（registry 副作用）。加新源 = 写一个模块 + 在此 import。
# 放在 base/__all__ 之后（避免 isort 上移到 base 之前）。
from edge_cam.data.adapters.detect import (  # noqa: E402,F401
    caltech_ct,
    coco2017,
    ena24,
    fiftyone_oiv7,
    oiv7_direct,
)
