"""质量门（plan §B.7 硬门）。

阈值默认全 None（不设硬门）——呼应 ADR-0001「先看包络再定数」。需要时从 config 填阈值，
让 CI / promote 流程据 gate_pass 决策。所有设了的检查取 AND。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from edge_cam.contracts.schemas.eval_report import EnvelopeReport

# from_yaml 只认这些键（4 维阈值门，plan §B.1）；多余键报错防配置漂移
_THRESHOLD_KEYS = {
    "min_fp32_top1",
    "min_regional_top1",
    "max_int8_drop",
    "max_field_drop",
}


@dataclass
class GateThresholds:
    min_fp32_top1: float | None = None
    min_regional_top1: float | None = None
    max_int8_drop: float | None = None  # int8_sim 相对 fp32 的 top-1 掉点上限
    max_field_drop: float | None = None  # 类现场相对 fp32 的 top-1 掉点上限

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

    fp32 = report.get("fp32_val")
    regional = report.get("regional")
    check_min("fp32_top1", fp32.top1 if fp32 else None, thr.min_fp32_top1)
    check_min("regional_top1", regional.top1 if regional else None, thr.min_regional_top1)
    check_max_drop("int8_drop", report.drop_from_baseline("int8_sim"), thr.max_int8_drop)
    check_max_drop("field_drop", report.drop_from_baseline("field"), thr.max_field_drop)

    passed = all(ok for _, ok, _ in checks) if checks else True
    return GateResult(passed=passed, checks=checks)
