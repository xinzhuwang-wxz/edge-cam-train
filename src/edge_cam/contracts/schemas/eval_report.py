"""可行性包络报告（CONTEXT.md 的 ①→④ 逐级掉点链；plan §8 三层口径 + §B.6 一页纸）。

持久化 + 渲染 Markdown 表给 stakeholder。诚实标注「类现场 ≠ 真现场」。"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class LevelResult(BaseModel):
    """包络中一级的结果。"""

    name: str  # fp32_val / int8_sim / field / regional
    top1: float
    top5: float
    n: int
    note: str = ""


class EnvelopeReport(BaseModel):
    """逐级掉点包络：FP32验证集 → 模拟INT8 → 类现场 → +地域过滤。"""

    model_name: str
    num_classes: int
    manifest: str
    levels: list[LevelResult] = Field(default_factory=list)

    def get(self, name: str) -> LevelResult | None:
        return next((lv for lv in self.levels if lv.name == name), None)

    def drop_from_baseline(self, name: str, baseline: str = "fp32_val") -> float | None:
        """某级相对 baseline 的 top-1 掉点（正=掉、负=涨）。"""
        base, lvl = self.get(baseline), self.get(name)
        if base is None or lvl is None:
            return None
        return base.top1 - lvl.top1

    def to_markdown(self) -> str:
        rows = [
            "| 级 | top-1 | top-5 | n | vs FP32 (top-1) | 备注 |",
            "|---|---|---|---|---|---|",
        ]
        for lv in self.levels:
            drop = self.drop_from_baseline(lv.name)
            drop_s = "—" if drop is None else f"{-drop:+.3f}"
            rows.append(
                f"| {lv.name} | {lv.top1:.3f} | {lv.top5:.3f} | {lv.n} | {drop_s} | {lv.note} |"
            )
        header = f"### 可行性包络 · {self.model_name} ({self.num_classes} 类) · {self.manifest}\n"
        return header + "\n".join(rows)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> EnvelopeReport:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
