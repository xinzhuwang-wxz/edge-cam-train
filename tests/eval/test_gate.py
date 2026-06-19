"""质量门：无阈值放行、阈值 AND、掉点上限。"""

from __future__ import annotations

from edge_cam.contracts.schemas.eval_report import EnvelopeReport, LevelResult
from edge_cam.eval.gates.gate import GateThresholds, evaluate_gate


def _report(fp32=0.9, int8=0.86, field=0.7, regional=0.94) -> EnvelopeReport:
    return EnvelopeReport(
        model_name="m",
        num_classes=525,
        manifest="birds525 v0",
        levels=[
            LevelResult(name="fp32_val", top1=fp32, top5=0.99, n=100),
            LevelResult(name="int8_sim", top1=int8, top5=0.99, n=100),
            LevelResult(name="field", top1=field, top5=0.9, n=100),
            LevelResult(name="regional", top1=regional, top5=0.99, n=100),
        ],
    )


def test_no_thresholds_passes() -> None:
    result = evaluate_gate(_report(), GateThresholds())
    assert result.passed
    assert result.checks == []


def test_min_regional_pass_fail() -> None:
    assert evaluate_gate(_report(regional=0.94), GateThresholds(min_regional_top1=0.9)).passed
    assert not evaluate_gate(_report(regional=0.80), GateThresholds(min_regional_top1=0.9)).passed


def test_max_int8_drop() -> None:
    # fp32 0.9 → int8 0.86，掉点 0.04
    assert evaluate_gate(_report(), GateThresholds(max_int8_drop=0.05)).passed
    assert not evaluate_gate(_report(), GateThresholds(max_int8_drop=0.02)).passed


def test_and_of_checks() -> None:
    thr = GateThresholds(min_fp32_top1=0.85, max_field_drop=0.15)
    # field drop = 0.9-0.7 = 0.2 > 0.15 → 整体 fail 即使 fp32 过
    assert not evaluate_gate(_report(), thr).passed
