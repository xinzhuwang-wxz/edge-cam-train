"""级联（[[ADR-0003]] C2）：粗检测 → crop → 细分类,把两个模型族合成产品本体。

脊柱之上的组合层(不属任何单族)。检测的 decode 在 `Detector` seam 之后(端侧 A7 / 离线评估
可各自实现),分类经 ONNX;crop 复用 data/crop.py。收编此前散在 results/scripts 的级联逻辑。"""

from edge_cam.cascade.pipeline import (
    CascadePipeline,
    CascadeReport,
    CascadeResult,
    Classifier,
    Detection,
    Detector,
)

__all__ = [
    "CascadePipeline",
    "CascadeResult",
    "CascadeReport",
    "Detector",
    "Classifier",
    "Detection",
]
