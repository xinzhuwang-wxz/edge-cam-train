"""发布编排：EnvelopeReport + GateResult + 训练产物 → ModelCard → registry（架构审查 A）。

此前断链：run_envelope 算出 GateResult.passed 却没人写进 ModelCard.gate_pass，于是
registry.promote 的 gate 门恒拒、右半边（registry/OTA）够不着。本模块是那条缺失的桥：
把评估结论 + 溯源固化成一张 ModelCard，register，并据 gate_pass 决定能否 promote 到 stable。
"""

from __future__ import annotations

from edge_cam.contracts.schemas.dataset import DatasetManifest
from edge_cam.contracts.schemas.eval_report import EnvelopeReport
from edge_cam.contracts.schemas.model_card import ModelCard, Platform, Precision, Provenance, Task
from edge_cam.eval.gates.gate import GateResult
from edge_cam.registry.store import ModelRegistry


def provenance_from_manifest(
    manifest: DatasetManifest, *, commercial_safe: bool = False
) -> Provenance:
    """从 manifest 逐样本溯源汇出 Provenance（数据集 + 许可，去重保序）。"""
    datasets = sorted({r.source for r in manifest.records if r.source})
    licenses = sorted({r.license for r in manifest.records if r.license})
    return Provenance(datasets=datasets, licenses=licenses, commercial_safe=commercial_safe)


def metrics_from_report(report: EnvelopeReport) -> dict[str, float]:
    """把四级包络摊平成 ModelCard.metrics（各级 top1/5 + int8/field 掉点）。"""
    metrics: dict[str, float] = {}
    for lv in report.levels:
        metrics[f"{lv.name}_top1"] = round(lv.top1, 4)
        metrics[f"{lv.name}_top5"] = round(lv.top5, 4)
    for level in ("int8_sim", "field"):
        drop = report.drop_from_baseline(level)
        if drop is not None:
            metrics[f"{level}_drop"] = round(drop, 4)
    return metrics


def build_model_card(
    report: EnvelopeReport,
    gate: GateResult,
    *,
    name: str,
    version: str,
    backbone: str,
    num_classes: int,
    input_size: int,
    task: Task = "classify",
    precision: Precision = "fp32",
    platform: Platform = "dev",
    artifact_path: str = "",
    provenance: Provenance | None = None,
) -> ModelCard:
    """评估结论固化成 ModelCard：**gate_pass = gate.passed**（接通断链的关键一行）。"""
    return ModelCard(
        name=name,
        task=task,
        backbone=backbone,
        num_classes=num_classes,
        input_size=input_size,
        precision=precision,
        platform=platform,
        artifact_path=artifact_path,
        version=version,
        provenance=provenance or Provenance(),
        metrics=metrics_from_report(report),
        gate_pass=gate.passed,
    )


def publish(registry: ModelRegistry, card: ModelCard, *, promote: bool = False) -> ModelCard:
    """register 卡片；promote=True 且过门则升 stable（未过门 registry.promote 会拒）。"""
    card = registry.register(card)
    if promote:
        card = registry.promote(card.name, card.version)
    return card
