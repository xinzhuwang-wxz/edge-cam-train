"""FP32 ONNX 导出 + onnxruntime 对齐校验（slow）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from edge_cam.train.classify.export import export_onnx, verify_onnx
from edge_cam.train.classify.module import Classifier


@pytest.mark.slow
def test_export_and_verify(tmp_path: Path) -> None:
    model = Classifier(model_name="efficientnet_lite0", num_classes=5, pretrained=False)
    out = export_onnx(model, tmp_path / "m.onnx", input_size=224, simplify=False)
    assert out.exists()
    assert verify_onnx(out, model, input_size=224)
