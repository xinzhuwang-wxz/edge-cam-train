"""检测器预处理**单一真值源**——train↔export↔inference 口径不漂移。

背景（round2 打磨）：归一化此前**两处各写一份**——训练在 NanoDet YAML、推理在 `cascade/adapters.py`
硬编码 `_DET_MEAN/_DET_STD`；`input_size` 默认散在 patch/OnnxDetector/decode 三处。改一处漏另一处 →
train/inference 不一致（同 [[ADR-0007]] sigmoid 那类缝上 bug）。

本模块收成**一个 `DetectorPreprocess`**：训练配置生成写它进 YAML、推理读它 → 一个源，改一处全同步。
NanoDet 口径：BGR、mean/std 减除、**stretch** resize（`decode` 的 `sx=w/input_size` 假设 stretch）。
"""

from __future__ import annotations

from pydantic import BaseModel


class DetectorPreprocess(BaseModel):
    """检测器预处理规范（NanoDet-Plus 口径）。train + inference 共读，消除硬编码漂移。"""

    model_config = {"frozen": True}

    input_size: int = 416
    mean_bgr: tuple[float, float, float] = (103.53, 116.28, 123.675)
    std_bgr: tuple[float, float, float] = (57.375, 57.12, 58.395)
    keep_ratio: bool = False  # stretch；decode 的 sx=w/input_size 假设 stretch
    to_bgr: bool = True  # RGB→BGR（NanoDet）


# 规范实例：全仓 train/inference 共用这一个（改预处理只改这里）
NANODET_PREPROCESS = DetectorPreprocess()
