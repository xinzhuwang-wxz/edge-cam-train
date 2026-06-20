"""检测评估器（[[ADR-0003]] C3）：DetectionMetrics（各级）→ 统一 EnvelopeReport。

让检测和分类走**同一条**发布链(EvalReport → metrics_from_report → ModelCard → registry → OTA)。
检测级指标进 LevelResult.metrics(map_50/map_5095/bird_recall_50),primary=map_5095(掉点按它算);
分类专属的 regional 级不适用检测(bird 是检测层单一类),故检测分级 = fp32 / int8_sim / field。
"""

from __future__ import annotations

from edge_cam.contracts.schemas.eval_report import EnvelopeReport, LevelResult
from edge_cam.eval.detect_metrics import DetectionMetrics

_PRIMARY = "map_5095"


def _level(name: str, dm: DetectionMetrics, note: str = "") -> LevelResult:
    metrics = {"map_50": dm.map_50, "map_5095": dm.map_5095}
    if dm.bird_recall_50 is not None:
        metrics["bird_recall_50"] = dm.bird_recall_50
    return LevelResult(name=name, metrics=metrics, primary=_PRIMARY, note=note)


def build_detection_report(
    levels: dict[str, DetectionMetrics],
    *,
    model_name: str,
    num_classes: int,
    manifest: str,
) -> EnvelopeReport:
    """各级 DetectionMetrics → EnvelopeReport。

    levels 顺序即报告顺序;首级作掉点 baseline(惯例 "fp32")。例:
        {"fp32": dm_fp32, "int8_sim": dm_int8, "field": dm_field}
    """
    notes = {
        "fp32": "FP32 ONNX",
        "int8_sim": "ORT-QDQ 模拟，非板子实测",
        "field": "退化代理，≠真现场",
    }
    return EnvelopeReport(
        model_name=model_name,
        num_classes=num_classes,
        manifest=manifest,
        levels=[_level(name, dm, notes.get(name, "")) for name, dm in levels.items()],
    )
