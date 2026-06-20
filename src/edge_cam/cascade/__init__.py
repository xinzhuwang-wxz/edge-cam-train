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


def __getattr__(name: str):  # 惰性暴露真 adapter(避免顶层 import onnxruntime)
    if name in ("OnnxClassifier", "OnnxDetector", "decode_nanodet"):
        from edge_cam.cascade import adapters

        return getattr(adapters, name)
    raise AttributeError(name)


__all__ = [
    "CascadePipeline",
    "CascadeResult",
    "CascadeReport",
    "Detector",
    "Classifier",
    "Detection",
    "OnnxClassifier",
    "OnnxDetector",
    "decode_nanodet",
]
