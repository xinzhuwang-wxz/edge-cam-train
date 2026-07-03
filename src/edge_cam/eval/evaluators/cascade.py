"""级联评估器（[[ADR-0003]] C2）：CascadeReport（各级）→ 统一 EnvelopeReport。

让级联（检测+分类组合）走**同一条**发布链（EvalReport → metrics_from_report → ModelCard →
registry → OTA），与检测/分类两族对等。级联级指标进 LevelResult.metrics
（cascade_top1/bird_hit_rate/fallback_rate），primary=cascade_top1（掉点/门控按它算）。

级名惯例与检测一致：fp32 / int8_sim（fp32=检测+分类均 fp32；int8_sim=两段均 ORT-QDQ 模拟 INT8
的**组合**掉点——级联特有关注点，量化对 crop→分类的链式放大）。
"""

from __future__ import annotations

import csv
from pathlib import Path

from edge_cam.cascade.pipeline import CascadeReport
from edge_cam.contracts.schemas.eval_report import EnvelopeReport, LevelResult

_PRIMARY = "cascade_top1"
_NOTES = {
    "fp32": "FP32 检测+分类组合",
    "int8_sim": "INT8 组合(ORT-QDQ 模拟，非板子实测)",
}


def _level(name: str, cr: CascadeReport) -> LevelResult:
    return LevelResult(
        name=name,
        metrics={
            "cascade_top1": cr.cascade_top1,
            "bird_hit_rate": cr.bird_hit_rate,
            "fallback_rate": cr.fallback_rate,
        },
        primary=_PRIMARY,
        n=cr.n,
        note=_NOTES.get(name, ""),
    )


def build_cascade_report(
    levels: dict[str, CascadeReport],
    *,
    model_name: str,
    num_classes: int,
    manifest: str,
) -> EnvelopeReport:
    """各级 CascadeReport → EnvelopeReport。

    levels 顺序即报告顺序；首级作掉点 baseline（惯例 "fp32"）。例：
        {"fp32": cr_fp32, "int8_sim": cr_int8}
    """
    return EnvelopeReport(
        model_name=model_name,
        num_classes=num_classes,
        manifest=manifest,
        levels=[_level(name, cr) for name, cr in levels.items()],
    )


_CASC_FIELDS = ["label", "cascade_top1", "bird_hit_rate", "fallback_rate"]


def append_cascade_row(cr: CascadeReport, label: str, out_dir: str | Path) -> Path:
    """把一行级联结果写入 cascade_ablation.csv（与检测 detect_ablation / 分类 ablation 分开）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cascade_ablation.csv"
    rows: list[dict] = []
    if csv_path.exists():
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    rows = [r for r in rows if r.get("label") != label]  # 同 label 覆盖
    rows.append(
        {
            "label": label,
            "cascade_top1": cr.cascade_top1,
            "bird_hit_rate": cr.bird_hit_rate,
            "fallback_rate": cr.fallback_rate,
        }
    )
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CASC_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return csv_path
