"""检测可行性包络入口（对等分类 run_envelope；ADR-0003 C3）。

box 的 NanoDet env 跑出 fp32/int8 的 COCOeval 数字 → 本入口纯结构化组装成
EnvelopeReport + gate + 检测版逐级掉点表 + detect_ablation.csv 总表（不依赖 nanodet）。
"""

from __future__ import annotations

from edge_cam.contracts.schemas.eval_report import EnvelopeReport
from edge_cam.eval.detect_metrics import DetectionMetrics
from edge_cam.eval.evaluators.detect import build_detection_report
from edge_cam.eval.gates.gate import GateThresholds
from edge_cam.eval.run_detect_envelope import (
    detect_envelope_markdown,
    metrics_from_eval_dict,
    run_detect_envelope,
)


def _levels() -> dict[str, DetectionMetrics]:
    return {
        "fp32": DetectionMetrics(
            map_50=0.679, map_5095=0.459, bird_recall_50=0.62, per_class_ap={"bird": 0.341}
        ),
        "int8_sim": DetectionMetrics(map_50=0.66, map_5095=0.45, bird_recall_50=0.60),
    }


def test_metrics_from_eval_dict_with_defaults() -> None:
    m = metrics_from_eval_dict(
        {
            "map_50": 0.679,
            "map_5095": 0.459,
            "bird_recall_50": 0.62,
            "per_class_ap": {"bird": 0.341},
        }
    )
    assert m.map_5095 == 0.459
    assert m.bird_recall_50 == 0.62
    assert m.per_class_ap["bird"] == 0.341
    # 缺省字段容错：bird_recall/per_class 可缺
    m2 = metrics_from_eval_dict({"map_50": 0.6, "map_5095": 0.5})
    assert m2.bird_recall_50 is None
    assert m2.per_class_ap == {}


def test_detect_markdown_uses_map_not_top1() -> None:
    rep = build_detection_report(
        _levels(), model_name="nanodet_320", num_classes=5, manifest="feeder v1"
    )
    md = detect_envelope_markdown(rep)
    assert "mAP@.5:.95" in md
    assert "top-1" not in md and "top-5" not in md  # 不是分类废表
    assert "0.459" in md  # fp32 mAP
    assert "0.009" in md  # int8 相对 fp32 的 mAP 掉点 (0.459-0.45)


def test_run_detect_envelope_writes_artifacts(tmp_path) -> None:
    report, gate, jp = run_detect_envelope(
        _levels(),
        model_name="nanodet_320",
        num_classes=5,
        manifest="feeder v1",
        label="feeder_320",
        output_dir=tmp_path,
        gate=GateThresholds(min_map_5095=0.4, max_int8_map_drop=0.05),
    )
    assert jp.exists() and (tmp_path / "report.md").exists()
    csv = tmp_path / "detect_ablation.csv"
    assert csv.exists()
    rows = csv.read_text(encoding="utf-8")
    assert "feeder_320_fp32" in rows and "feeder_320_int8_sim" in rows
    # gate：min_map_5095=0.4 满足(0.459)、int8 掉点 0.009<0.05 → pass
    assert gate.passed
    # 持久化可回读
    rep2 = EnvelopeReport.load(jp)
    assert rep2.get("fp32").value("map_5095") == 0.459


def test_run_detect_envelope_gate_fail(tmp_path) -> None:
    _r, gate, _j = run_detect_envelope(
        _levels(),
        model_name="m",
        num_classes=5,
        manifest="x",
        label="t",
        output_dir=tmp_path,
        gate=GateThresholds(min_map_5095=0.6),  # 0.459 < 0.6 → fail
    )
    assert not gate.passed


def test_run_detect_envelope_publishes_card(tmp_path) -> None:
    """检测接发布链：register 给定 → build_model_card(task=detect) → registry → promote。"""
    from edge_cam.registry.store import ModelRegistry

    idx = tmp_path / "models.yaml"
    _report, _gate, _jp = run_detect_envelope(
        _levels(),
        model_name="nanodet_320",
        num_classes=5,
        manifest="feeder v1",
        label="feeder_320",
        output_dir=tmp_path,
        gate=GateThresholds(min_map_5095=0.4),  # 0.459 ≥ 0.4 → pass
        register={
            "name": "detector_feeder",
            "version": "v1",
            "backbone": "nanodet_320",
            "input_size": 320,
            "index": str(idx),
            "promote": True,
        },
    )
    card = ModelRegistry(idx).get("detector_feeder", version="v1")
    assert card is not None
    assert card.task == "detect"
    assert card.metrics["fp32_map_5095"] == 0.459
    assert card.gate_pass and card.channel == "stable"  # 过门 + promote


def test_run_detect_envelope_no_register_skips_publish(tmp_path) -> None:
    """不给 register → 只出 report/csv，不碰 registry（向后兼容）。"""
    _r, _g, _j = run_detect_envelope(
        _levels(),
        model_name="m",
        num_classes=5,
        manifest="x",
        label="t",
        output_dir=tmp_path,
    )
    assert not (tmp_path / "models.yaml").exists()
