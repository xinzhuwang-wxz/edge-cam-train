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
from edge_cam.eval.full_eval import run_full_eval
from edge_cam.eval.regional import RegionalMask
from edge_cam.train.backends import get_backend


def run_ablation(
    base_cfg: DictConfig,
    grid_spec: dict[str, list],
    manifest: DatasetManifest,
    *,
    regional_mask: RegionalMask | None = None,
    quant: bool = False,
) -> list[dict]:
    """跑整个网格，返回结果行列表（每行 = overrides + 各级 top-1/5）。

    quant=True 时每格导出 FP32 ONNX 并加 INT8 模拟级（plan §B.4 量化掉点列）；默认关（更快）。
    经 run_full_eval 统一编排，与 run_envelope 同一条评估路（架构审查 B）。
    """
    rows: list[dict] = []
    backend = get_backend(base_cfg.get("backend", "classify"))  # 经训练 seam（[[ADR-0003]] C1）
    for i, overrides in enumerate(expand_grid(grid_spec)):
        dotlist = [f"{k}={v}" for k, v in overrides.items()]
        cfg = OmegaConf.merge(base_cfg, OmegaConf.from_dotlist(dotlist))
        model = backend.train(cfg)  # type: ignore[arg-type]

        fp32_onnx = None
        if quant:  # 导出本格模型 → run_full_eval 据此出 INT8 级（过 onnx_artifact 契约门）
            fp32_onnx = backend.export_fp32_onnx(
                model,
                Path(cfg.output_dir) / f"cell{i}_{cfg.model.name}_fp32.onnx",
                cfg.data.input_size,
            )

        report = run_full_eval(
            model,
            manifest,
            input_size=cfg.data.input_size,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.get("num_workers", 4),
            fp32_onnx=fp32_onnx,
            output_dir=cfg.output_dir,
            regional_mask=regional_mask,
            data_root=cfg.data.get("data_root", None),
            val_only=not quant,  # 消融默认只 val 选型不碰 test（§B.0）；quant 模式才跑全包络
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
