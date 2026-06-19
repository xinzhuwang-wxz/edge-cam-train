"""promotion（架构审查 A）：report+gate→ModelCard→registry，gate_pass 真正驱动 promote。"""

from __future__ import annotations

from pathlib import Path

import pytest

from edge_cam.contracts.schemas.dataset import DatasetManifest, SampleRecord
from edge_cam.contracts.schemas.eval_report import EnvelopeReport, LevelResult
from edge_cam.eval.gates.gate import GateThresholds, evaluate_gate
from edge_cam.registry.promotion import (
    build_model_card,
    metrics_from_report,
    provenance_from_manifest,
    publish,
)
from edge_cam.registry.store import ModelRegistry


def _report(fp32=0.9, int8=0.86) -> EnvelopeReport:
    return EnvelopeReport(
        model_name="efficientnet_lite0",
        num_classes=3,
        manifest="t v0",
        levels=[
            LevelResult(name="fp32_val", top1=fp32, top5=0.99, n=100),
            LevelResult(name="int8_sim", top1=int8, top5=0.98, n=100),
        ],
    )


def _manifest() -> DatasetManifest:
    return DatasetManifest(
        name="t",
        version="v0",
        seed=0,
        class_to_idx={"a": 0, "b": 1, "c": 2},
        records=[
            SampleRecord(
                path="a/0.jpg", label="a", split="train", source="kaggle", license="cc-by"
            ),
            SampleRecord(path="b/0.jpg", label="b", split="train", source="oiv7", license="cc-by"),
        ],
    )


def test_metrics_and_provenance_extracted() -> None:
    m = metrics_from_report(_report())
    assert m["fp32_val_top1"] == 0.9
    assert m["int8_sim_top1"] == 0.86
    assert m["int8_sim_drop"] == pytest.approx(0.04)  # 0.9-0.86
    prov = provenance_from_manifest(_manifest())
    assert prov.datasets == ["kaggle", "oiv7"]
    assert prov.licenses == ["cc-by"]
    assert prov.commercial_safe is False


def test_card_gate_pass_reflects_gate() -> None:
    """接通断链的关键：gate.passed 写进 ModelCard.gate_pass。"""
    gate_fail = evaluate_gate(_report(), GateThresholds(max_int8_drop=0.02))  # 0.04>0.02 → fail
    card = build_model_card(
        _report(), gate_fail, name="m", version="v0", backbone="x", num_classes=3, input_size=224
    )
    assert card.gate_pass is False

    gate_ok = evaluate_gate(_report(), GateThresholds(max_int8_drop=0.05))  # 0.04<=0.05 → pass
    card_ok = build_model_card(
        _report(), gate_ok, name="m", version="v0", backbone="x", num_classes=3, input_size=224
    )
    assert card_ok.gate_pass is True


def test_publish_chain_gate_drives_promote(tmp_path: Path) -> None:
    """端到端：过门 → register+promote 到 stable；未过门 → promote 被拒。"""
    registry = ModelRegistry(tmp_path / "models.yaml")

    ok = build_model_card(
        _report(),
        evaluate_gate(_report(), GateThresholds(max_int8_drop=0.05)),
        name="good",
        version="v0",
        backbone="x",
        num_classes=3,
        input_size=224,
        platform="v85x",
    )
    published = publish(registry, ok, promote=True)
    assert published.channel == "stable"  # gate 过 → 升 stable

    bad = build_model_card(
        _report(),
        evaluate_gate(_report(), GateThresholds(max_int8_drop=0.02)),
        name="bad",
        version="v0",
        backbone="x",
        num_classes=3,
        input_size=224,
        platform="v85x",
    )
    with pytest.raises(ValueError, match="未过 gate"):
        publish(registry, bad, promote=True)  # gate 未过 → promote 拒
    # 但 register 已发生：仍在 candidate
    assert registry.get("bad", channel="candidate") is not None
