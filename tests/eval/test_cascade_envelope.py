"""级联折（检测+分类）发布骨架（[[ADR-0003]] C2 → 统一发布链）。

CascadeReport → EnvelopeReport（cascade_top1/bird_hit_rate/fallback_rate）→ gate →
cascade_ablation.csv → publish(task="cascade")。与检测折 run_detect_envelope 对等。
"""

from __future__ import annotations

import pytest

from edge_cam.cascade.pipeline import CascadeReport
from edge_cam.eval.evaluators.cascade import build_cascade_report
from edge_cam.eval.gates.gate import GateThresholds
from edge_cam.eval.run_cascade_envelope import cascade_envelope_markdown, run_cascade_envelope
from edge_cam.registry.store import ModelRegistry


def _levels() -> dict[str, CascadeReport]:
    return {
        "fp32": CascadeReport(n=1000, bird_hit_rate=0.90, cascade_top1=0.80, fallback_rate=0.10),
        "int8_sim": CascadeReport(
            n=1000, bird_hit_rate=0.88, cascade_top1=0.76, fallback_rate=0.14
        ),
    }


def test_build_cascade_report_maps_metrics() -> None:
    rep = build_cascade_report(
        _levels(), model_name="nanodet_320+lite0", num_classes=525, manifest="feeder v1"
    )
    fp32 = rep.get("fp32")
    assert fp32 is not None
    assert fp32.primary == "cascade_top1"
    assert fp32.value("cascade_top1") == 0.80
    assert fp32.value("bird_hit_rate") == 0.90
    assert fp32.value("fallback_rate") == 0.10
    # 掉点按 primary（cascade_top1）：0.80 - 0.76
    assert rep.drop_from_baseline("int8_sim", baseline="fp32") == pytest.approx(0.04)


def test_cascade_markdown_uses_cascade_cols_not_top1() -> None:
    rep = build_cascade_report(_levels(), model_name="m", num_classes=525, manifest="x")
    md = cascade_envelope_markdown(rep)
    assert "级联 top-1" in md
    assert "bird 检出率" in md
    assert "top-5" not in md  # 不是分类废表
    assert "0.800" in md  # fp32 cascade_top1


def test_run_cascade_envelope_writes_and_publishes(tmp_path) -> None:
    idx = tmp_path / "models.yaml"
    _rep, gate, jp = run_cascade_envelope(
        _levels(),
        model_name="nanodet_320+lite0",
        num_classes=525,
        manifest="feeder v1",
        label="cascade_v1",
        output_dir=tmp_path,
        gate=GateThresholds(
            min_cascade_top1=0.7, min_bird_hit_rate=0.85, max_int8_cascade_drop=0.06
        ),
        register={
            "name": "cascade_feeder",
            "version": "v1",
            "backbone": "nanodet_320+lite0",
            "input_size": 224,
            "index": str(idx),
            "promote": True,
        },
    )
    assert jp.exists() and (tmp_path / "report.md").exists()
    csv = tmp_path / "cascade_ablation.csv"
    assert csv.exists()
    rows = csv.read_text(encoding="utf-8")
    assert "cascade_v1_fp32" in rows and "cascade_v1_int8_sim" in rows
    assert gate.passed

    card = ModelRegistry(idx).get("cascade_feeder", version="v1")
    assert card is not None
    assert card.task == "cascade"
    assert card.metrics["fp32_cascade_top1"] == 0.80
    assert card.gate_pass and card.channel == "stable"


def test_run_cascade_envelope_gate_fail_no_promote(tmp_path) -> None:
    idx = tmp_path / "models.yaml"
    _rep, gate, _jp = run_cascade_envelope(
        _levels(),
        model_name="m",
        num_classes=525,
        manifest="x",
        label="c",
        output_dir=tmp_path,
        gate=GateThresholds(min_cascade_top1=0.9),  # 0.80 < 0.9 → fail
        register={
            "name": "c",
            "version": "v0",
            "backbone": "b",
            "input_size": 224,
            "index": str(idx),
            "promote": True,
        },
    )
    assert not gate.passed
    card = ModelRegistry(idx).get("c", version="v0")
    assert card is not None
    assert card.gate_pass is False and card.channel == "candidate"


def test_run_cascade_envelope_no_register_skips_publish(tmp_path) -> None:
    _r, _g, _j = run_cascade_envelope(
        _levels(),
        model_name="m",
        num_classes=525,
        manifest="x",
        label="c",
        output_dir=tmp_path,
    )
    assert not (tmp_path / "models.yaml").exists()
