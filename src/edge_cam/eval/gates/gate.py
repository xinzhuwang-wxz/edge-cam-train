"""质量门（plan §B.7 硬门）。

阈值默认全 None（不设硬门）——呼应 ADR-0001「先看包络再定数」。需要时从 config 填阈值，
让 CI / promote 流程据 gate_pass 决策。所有设了的检查取 AND。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from edge_cam.contracts.schemas.eval_report import EnvelopeReport

# from_yaml 只认这些键；多余键报错防配置漂移。分类(top1/regional)+ 检测(map/recall)两族
_THRESHOLD_KEYS = {
    "min_fp32_top1",  # 分类
    "min_regional_top1",
    "max_int8_drop",
    "max_field_drop",
    "min_map_5095",  # 检测（[[ADR-0003]] C3：命名指标阈值）
    "min_map_50",
    "min_bird_recall_50",
    "max_int8_map_drop",
}


@dataclass
class GateThresholds:
    # 分类
    min_fp32_top1: float | None = None
    min_regional_top1: float | None = None
    max_int8_drop: float | None = None  # int8_sim 相对 fp32 的 top-1 掉点上限
    max_field_drop: float | None = None  # 类现场相对 fp32 的 top-1 掉点上限
    # 检测（fp32 级 map/recall 下限 + int8 的 mAP 掉点上限）
    min_map_5095: float | None = None
    min_map_50: float | None = None
    min_bird_recall_50: float | None = None
    max_int8_map_drop: float | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> GateThresholds:
        """从 yaml 加载阈值（缺省键 = None = 不设该维硬门，呼应 ADR-0001）。"""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(raw) - _THRESHOLD_KEYS
        if unknown:
            raise ValueError(
                f"gate 配置含未知键 {sorted(unknown)}（允许：{sorted(_THRESHOLD_KEYS)}）"
            )
        return cls(**{k: raw[k] for k in _THRESHOLD_KEYS if k in raw})


@dataclass
class GateResult:
    passed: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"**Gate: {'PASS ✅' if self.passed else 'FAIL ❌'}**", ""]
        for name, ok, detail in self.checks:
            lines.append(f"- {'✅' if ok else '❌'} {name}: {detail}")
        if not self.checks:
            lines.append("- （未设阈值，不设硬门 —— 先看包络再定，ADR-0001）")
        return "\n".join(lines)


def evaluate_gate(report: EnvelopeReport, thr: GateThresholds) -> GateResult:
    checks: list[tuple[str, bool, str]] = []

    def check_min(name: str, value: float | None, floor: float | None) -> None:
        if floor is None:
            return
        ok = value is not None and value >= floor
        checks.append((name, ok, f"{value if value is not None else 'n/a'} ≥ {floor}"))

    def check_max_drop(name: str, drop: float | None, ceil: float | None) -> None:
        if ceil is None:
            return
        ok = drop is not None and drop <= ceil
        checks.append((name, ok, f"掉点 {drop if drop is not None else 'n/a'} ≤ {ceil}"))

    # 分类：fp32_val/regional 的 top-1 下限 + int8/field 掉点上限
    fp32 = report.get("fp32_val")
    regional = report.get("regional")
    check_min("fp32_top1", fp32.top1 if fp32 else None, thr.min_fp32_top1)
    # regional 改 in-region on 口径(issue#11);兼容旧 top1
    reg_val = None
    if regional is not None:
        reg_val = regional.value("in_region_top1_on") or regional.top1
    check_min("regional_top1", reg_val, thr.min_regional_top1)
    check_max_drop("int8_drop", report.drop_from_baseline("int8_sim"), thr.max_int8_drop)
    check_max_drop("field_drop", report.drop_from_baseline("field"), thr.max_field_drop)

    # 检测：fp32 级 map/recall 下限 + int8 的 mAP@.5:.95 掉点上限
    fp32_det = report.get("fp32")
    check_min("map_5095", fp32_det.value("map_5095") if fp32_det else None, thr.min_map_5095)
    check_min("map_50", fp32_det.value("map_50") if fp32_det else None, thr.min_map_50)
    check_min(
        "bird_recall_50",
        fp32_det.value("bird_recall_50") if fp32_det else None,
        thr.min_bird_recall_50,
    )
    check_max_drop(
        "int8_map_drop",
        report.drop_from_baseline("int8_sim", baseline="fp32", metric="map_5095"),
        thr.max_int8_map_drop,
    )

    passed = all(ok for _, ok, _ in checks) if checks else True
    return GateResult(passed=passed, checks=checks)
