"""EnvelopeReport：逐级掉点计算、markdown 渲染、存盘往返。"""

from __future__ import annotations

from pathlib import Path

from edge_cam.contracts.schemas.eval_report import EnvelopeReport, LevelResult


def _report() -> EnvelopeReport:
    return EnvelopeReport(
        model_name="efficientnet_lite0",
        num_classes=525,
        manifest="birds525 v0",
        levels=[
            LevelResult(name="fp32_val", top1=0.90, top5=0.98, n=2500),
            LevelResult(name="int8_sim", top1=0.86, top5=0.97, n=2500, note="ORT 模拟"),
            LevelResult(name="field", top1=0.70, top5=0.88, n=2500, note="退化代理"),
            LevelResult(name="regional", top1=0.94, top5=0.99, n=2500, note="区域 12%"),
        ],
    )


def test_drop_from_baseline() -> None:
    r = _report()
    assert r.drop_from_baseline("int8_sim") == 0.90 - 0.86
    assert r.drop_from_baseline("regional") < 0  # 地域过滤反而涨点
    assert r.drop_from_baseline("missing") is None


def test_markdown_contains_levels() -> None:
    md = _report().to_markdown()
    for name in ("fp32_val", "int8_sim", "field", "regional"):
        assert name in md
    assert "可行性包络" in md


def test_save_load_roundtrip(tmp_path: Path) -> None:
    r = _report()
    out = tmp_path / "envelope.json"
    r.save(out)
    assert EnvelopeReport.load(out) == r
