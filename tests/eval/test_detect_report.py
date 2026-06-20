"""检测走统一发布链（[[ADR-0003]] C3）：DetectionMetrics → EnvelopeReport → ModelCard + gate。"""

from __future__ import annotations

from edge_cam.eval.detect_metrics import DetectionMetrics
from edge_cam.eval.evaluators.detect import build_detection_report
from edge_cam.eval.gates.gate import GateThresholds, evaluate_gate
from edge_cam.registry.promotion import build_model_card, metrics_from_report


def _report():
    return build_detection_report(
        {
            "fp32": DetectionMetrics(map_50=0.72, map_5095=0.59, bird_recall_50=0.645),
            "int8_sim": DetectionMetrics(map_50=0.70, map_5095=0.58, bird_recall_50=0.63),
        },
        model_name="nanodet_plus_m",
        num_classes=11,
        manifest="detection_feeder v0",
    )


def test_detection_report_metrics_flatten() -> None:
    m = metrics_from_report(_report())
    assert m["fp32_map_5095"] == 0.59
    assert m["fp32_map_50"] == 0.72
    assert m["fp32_bird_recall_50"] == 0.645
    # int8 掉点按 primary(map_5095) 相对首级 fp32
    assert m["int8_sim_drop"] == round(0.59 - 0.58, 4)


def test_detection_model_card_publishable() -> None:
    gate = evaluate_gate(_report(), GateThresholds())  # 无阈值 → pass
    card = build_model_card(
        _report(),
        gate,
        name="nanodet_plus_m",
        version="v0",
        backbone="shufflenetv2",
        num_classes=11,
        input_size=416,
        task="detect",  # 检测一等公民
    )
    assert card.task == "detect"
    assert card.gate_pass is True
    assert card.metrics["fp32_map_5095"] == 0.59


def test_detection_gate_named_thresholds() -> None:
    rep = _report()
    # mAP 下限 + int8 掉点上限：满足 → pass
    ok = evaluate_gate(rep, GateThresholds(min_map_5095=0.55, max_int8_map_drop=0.05))
    assert ok.passed
    # mAP 下限过高 → fail
    bad = evaluate_gate(rep, GateThresholds(min_map_5095=0.65))
    assert not bad.passed


def test_classify_gate_unaffected() -> None:
    # 分类报告 + 分类阈值仍按 top1 工作(向后兼容)
    from edge_cam.contracts.schemas.eval_report import EnvelopeReport, LevelResult

    rep = EnvelopeReport(
        model_name="eff_lite0",
        num_classes=525,
        manifest="birds525",
        levels=[
            LevelResult(name="fp32_val", top1=0.93, top5=0.98, n=100),
            LevelResult(name="int8_sim", top1=0.92, top5=0.97, n=100),
        ],
    )
    res = evaluate_gate(rep, GateThresholds(min_fp32_top1=0.9, max_int8_drop=0.05))
    assert res.passed
    assert metrics_from_report(rep)["fp32_val_top1"] == 0.93
