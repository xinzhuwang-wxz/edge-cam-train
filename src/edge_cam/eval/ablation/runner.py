"""消融 harness（plan §B：受控网格 → 训练 + 包络 → 一张实验总表）。

对网格每个单元跑 train.run → build_envelope，把各级 top-1/5 汇成一行；导 CSV + Markdown
（plan §B.6 给 stakeholder 的一页纸）。单变量受控：grid 只改 backbone/输入/quant 档之一。

真跑在 GPU（AutoDL）；本地 CPU 可 smoke（tiny grid + fast_dev_run）。"""

from __future__ import annotations

import csv
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from edge_cam.contracts.schemas.dataset import DatasetManifest
from edge_cam.eval.ablation.grid import expand_grid, label_for
from edge_cam.eval.envelope import build_envelope
from edge_cam.eval.regional import RegionalMask
from edge_cam.train.classify.train import run as run_train


def run_ablation(
    base_cfg: DictConfig,
    grid_spec: dict[str, list],
    manifest: DatasetManifest,
    *,
    regional_mask: RegionalMask | None = None,
) -> list[dict]:
    """跑整个网格，返回结果行列表（每行 = overrides + 各级 top-1/5）。"""
    rows: list[dict] = []
    for overrides in expand_grid(grid_spec):
        dotlist = [f"{k}={v}" for k, v in overrides.items()]
        cfg = OmegaConf.merge(base_cfg, OmegaConf.from_dotlist(dotlist))
        model = run_train(cfg)  # type: ignore[arg-type]
        report = build_envelope(
            model,
            manifest,
            input_size=cfg.data.input_size,
            batch_size=cfg.data.batch_size,
            num_workers=0,
            regional_mask=regional_mask,
        )
        row: dict = {"label": label_for(overrides), **overrides}
        for lv in report.levels:
            row[f"{lv.name}_top1"] = round(lv.top1, 4)
            row[f"{lv.name}_top5"] = round(lv.top5, 4)
        rows.append(row)
    return rows


def write_results(rows: list[dict], out_dir: str | Path) -> tuple[Path, Path]:
    """导 CSV + Markdown 表，返回两者路径。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    csv_path = out_dir / "ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in rows:
        md_lines.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    md_path = out_dir / "ablation.md"
    md_path.write_text("### 消融实验总表\n\n" + "\n".join(md_lines) + "\n", encoding="utf-8")
    return csv_path, md_path
