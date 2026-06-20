"""发布编排：EnvelopeReport + GateResult + 训练产物 → ModelCard → registry（架构审查 A）。

此前断链：run_envelope 算出 GateResult.passed 却没人写进 ModelCard.gate_pass，于是
registry.promote 的 gate 门恒拒、右半边（registry/OTA）够不着。本模块是那条缺失的桥：
把评估结论 + 溯源固化成一张 ModelCard，register，并据 gate_pass 决定能否 promote 到 stable。
"""

from __future__ import annotations

from edge_cam.contracts.schemas.dataset import DatasetManifest, provenance_summary
from edge_cam.contracts.schemas.detection_manifest import DetectionManifest
from edge_cam.contracts.schemas.eval_report import EnvelopeReport
from edge_cam.contracts.schemas.model_card import ModelCard, Platform, Precision, Provenance, Task
from edge_cam.eval.gates.gate import GateResult
from edge_cam.registry.store import ModelRegistry


def provenance_from_manifest(
    manifest: DatasetManifest | DetectionManifest, *, commercial_safe: bool = False
) -> Provenance:
    """逐样本溯源汇出 Provenance（数据集+许可）。分类/检测 manifest 通用([[ADR-0003]] C5)。"""
    datasets, licenses = provenance_summary(manifest.records)
    return Provenance(datasets=datasets, licenses=licenses, commercial_safe=commercial_safe)


def metrics_from_report(report: EnvelopeReport) -> dict[str, float]:
    """把包络各级摊平成 ModelCard.metrics（[[ADR-0003]] C3：dict 化，两族通用）。

    分类级 metrics={top1,top5}（validator 自镜像）→ 输出 `{级}_top1/top5` 同旧；
    检测级 metrics={map_50,map_5095,bird_recall}→ 输出 `{级}_map_50` 等。掉点按各级 primary 算。"""
    metrics: dict[str, float] = {}
    for lv in report.levels:
        for mname, mval in lv.metrics.items():
            metrics[f"{lv.name}_{mname}"] = round(mval, 4)
    # 掉点：相对首级（分类 fp32_val / 检测 fp32），按各级 primary 指标
    if report.levels:
        baseline = report.levels[0].name
        for lv in report.levels[1:]:
            drop = report.drop_from_baseline(lv.name, baseline=baseline)
            if drop is not None:
                metrics[f"{lv.name}_drop"] = round(drop, 4)
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
