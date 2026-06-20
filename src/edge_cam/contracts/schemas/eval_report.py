"""可行性包络报告（CONTEXT.md 的 ①→④ 逐级掉点链；plan §8 三层口径 + §B.6 一页纸）。

持久化 + 渲染 Markdown 表给 stakeholder。诚实标注「类现场 ≠ 真现场」。"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class LevelResult(BaseModel):
    """包络中一级的结果（[[ADR-0003]] C3：dict 化指标,容纳分类 top-k 与检测 mAP）。

    `metrics` 是规范指标存储(dict);`primary` 是该级主指标名(掉点/门控按它算)。
    分类历史字段 top1/top5 保留向后兼容,validator 自动镜像进 metrics → gate/promotion
    统一读 metrics,两族(top1 / map_50…)同一条路。"""

    name: str  # 分类: fp32_val/int8_sim/field/regional;检测: fp32/int8_sim/field
    top1: float = 0.0  # 分类便利字段(检测留默认)
    top5: float = 0.0
    n: int = 0
    note: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)  # 规范指标:{top1,top5} 或 {map_50,…}
    primary: str = "top1"  # 主指标名;检测置 "map_5095" 等

    @model_validator(mode="after")
    def _mirror_topk(self) -> LevelResult:
        """metrics 空且有 top1 → 镜像 {top1,top5}(分类向后兼容);保证 metrics 恒为规范源。"""
        if not self.metrics and (self.top1 or self.top5):
            self.metrics = {"top1": self.top1, "top5": self.top5}
        return self

    def value(self, metric: str | None = None) -> float | None:
        """取指标值(默认 primary)。"""
        return self.metrics.get(metric or self.primary)


class EnvelopeReport(BaseModel):
    """逐级掉点包络：FP32验证集 → 模拟INT8 → 类现场 → +地域过滤。"""

    model_name: str
    num_classes: int
    manifest: str
    levels: list[LevelResult] = Field(default_factory=list)

    def get(self, name: str) -> LevelResult | None:
        return next((lv for lv in self.levels if lv.name == name), None)

    def drop_from_baseline(
        self, name: str, baseline: str = "fp32_val", metric: str | None = None
    ) -> float | None:
        """某级相对 baseline 的掉点（正=掉、负=涨）。

        metric 默认取该级 primary（分类=top1、检测=map_5095…）→ 两族同一逻辑。"""
        base, lvl = self.get(baseline), self.get(name)
        if base is None or lvl is None:
            return None
        m = metric or lvl.primary
        bv, lv = base.value(m), lvl.value(m)
        if bv is None or lv is None:
            return None
        return bv - lv

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
