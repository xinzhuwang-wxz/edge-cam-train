"""级联可行性包络入口（对等分类 run_envelope / 检测 run_detect_envelope；[[ADR-0003]] C2）。

级联=产品本体（检测→crop→分类，贯穿置信门控+层级回退）。CascadePipeline.evaluate 产出各级
CascadeReport（检出率/级联 top-1/回退率）→ 本入口**纯结构化**组装成 EnvelopeReport + gate +
级联版逐级掉点表 + cascade_ablation.csv 总表，并可选发布到 registry（task="cascade"）。

**级联卡语义**：检测器固定、分类器才是 OTA 换/扩单元；故级联卡是**产品级评估记录**（追踪组合
精度是否达标），非 OTA 投递单元本身。backbone/num_classes/input_size 由调用方传（记录用，
惯例 backbone="<det>+<clf>"、num_classes=种数、input_size=分类器输入）；provenance 建议为检测+
分类两族的并集（许可红线随卡披露，§4）。要收紧此语义再走 ADR。

CLI:
    python -m edge_cam.eval.run_cascade_envelope --eval-json casc_eval.json \\
        --model-name "nanodet_320+lite0" --num-classes 525 --manifest "feeder v1" \\
        --label cascade_v1 --out outputs/cascade/v1 [--gate gate.yaml] \\
        [--register cascade_feeder --input-size 224 --promote]

eval-json 形态：{"fp32": {"n":..,"bird_hit_rate":..,"cascade_top1":..,"fallback_rate":..},
                 "int8_sim": {...}}（首键作掉点 baseline）。
"""

from __future__ import annotations

from pathlib import Path

from edge_cam.cascade.pipeline import CascadeReport
from edge_cam.contracts.schemas.eval_report import EnvelopeReport
from edge_cam.contracts.schemas.model_card import Provenance
from edge_cam.eval.evaluators.cascade import append_cascade_row, build_cascade_report
from edge_cam.eval.gates.gate import GateResult, GateThresholds, evaluate_gate

_BASELINE = "fp32"  # 级联首级名（掉点基线）；evaluators/cascade 惯例


def report_from_eval_dict(d: dict) -> CascadeReport:
    """一级评估 dict → CascadeReport。"""
    return CascadeReport(
        n=int(d.get("n", 0)),
        bird_hit_rate=float(d["bird_hit_rate"]),
        cascade_top1=float(d["cascade_top1"]),
        fallback_rate=float(d.get("fallback_rate", 0.0)),
    )


def cascade_envelope_markdown(report: EnvelopeReport) -> str:
    """级联版逐级掉点表（级联 top-1 / bird 检出率 / 回退率 + 相对 fp32 的级联 top-1 掉点）。

    分类 `EnvelopeReport.to_markdown` 渲染 top-1/5、检测渲 map；级联指标另成一列，否则出废表。"""
    rows = [
        "| 级 | 级联 top-1 | bird 检出率 | 回退率 | vs fp32 (top-1) | 备注 |",
        "|---|---|---|---|---|---|",
    ]
    for lv in report.levels:
        if lv.name == _BASELINE:
            drop_s = "—"
        else:
            drop = report.drop_from_baseline(lv.name, baseline=_BASELINE, metric="cascade_top1")
            drop_s = "—" if drop is None else f"{-drop:+.3f}"  # 正=涨、负=掉
        rows.append(
            f"| {lv.name} | {lv.value('cascade_top1'):.3f} | {lv.value('bird_hit_rate'):.3f} "
            f"| {lv.value('fallback_rate'):.3f} | {drop_s} | {lv.note} |"
        )
    header = (
        f"### 级联可行性包络 · {report.model_name} ({report.num_classes} 种) · {report.manifest}\n"
    )
    return header + "\n".join(rows)


def run_cascade_envelope(
    levels: dict[str, CascadeReport],
    *,
    model_name: str,
    num_classes: int,
    manifest: str,
    label: str,
    output_dir: str | Path,
    gate: GateThresholds | None = None,
    register: dict | None = None,
    provenance: Provenance | None = None,
) -> tuple[EnvelopeReport, GateResult, Path]:
    """组装级联包络 + gate，落盘 report.json/md + cascade_ablation.csv，可选发布到 registry。

    返回 (report, gate, json_path)。levels 首级作掉点 baseline（惯例 "fp32"）。
    register 给定则 build_model_card(task="cascade") → registry →（promote 且过门）升 stable；
    键同检测：name/version/backbone/input_size/index/promote/platform/artifact_path。"""
    report = build_cascade_report(
        levels, model_name=model_name, num_classes=num_classes, manifest=manifest
    )
    gate_res = evaluate_gate(report, gate or GateThresholds())

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report.json"
    report.save(json_path)
    md = cascade_envelope_markdown(report) + "\n\n" + gate_res.to_markdown() + "\n"
    (out / "report.md").write_text(md, encoding="utf-8")

    for name, cr in levels.items():
        append_cascade_row(cr, f"{label}_{name}", out)

    if register is not None:
        _publish_cascade(report, gate_res, register, num_classes, model_name, out, provenance)

    return report, gate_res, json_path


def _publish_cascade(
    report: EnvelopeReport,
    gate: GateResult,
    reg: dict,
    num_classes: int,
    model_name: str,
    out_dir: Path,
    provenance: Provenance | None,
) -> None:
    """接发布链（task="cascade"）。惰性 import 免测试拉 registry 依赖。"""
    from edge_cam.registry.promotion import publish_report

    card = publish_report(
        report,
        gate,
        registry_index=reg.get("index", str(out_dir / "models.yaml")),
        name=reg.get("name", model_name),
        version=reg.get("version", "v0"),
        backbone=reg.get("backbone", model_name),
        num_classes=num_classes,
        input_size=int(reg.get("input_size", 224)),
        task="cascade",
        platform=reg.get("platform", "dev"),
        artifact_path=reg.get("artifact_path", ""),
        provenance=provenance,
        promote=bool(reg.get("promote", False)),
    )
    print(
        f"[publish] {card.name} v{card.version} → channel={card.channel} gate_pass={card.gate_pass}"
    )


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    p = argparse.ArgumentParser(description="级联可行性包络入口（ADR-0003 C2）")
    p.add_argument(
        "--eval-json", required=True, help="各级评估 json（CascadePipeline.evaluate 产出）"
    )
    p.add_argument("--model-name", required=True, help='惯例 "<det>+<clf>"')
    p.add_argument("--num-classes", type=int, required=True, help="种数（分类器输出）")
    p.add_argument("--manifest", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gate", default=None, help="gate 阈值 yaml(可选)")
    p.add_argument("--register", default=None, help="发布卡名（给定即接 registry）")
    p.add_argument("--reg-version", default="v0")
    p.add_argument("--backbone", default=None, help="缺省=model-name")
    p.add_argument("--input-size", type=int, default=224, help="分类器输入（级联终端决策段）")
    p.add_argument(
        "--registry-index", default=None, help="models.yaml 路径（缺省=out/models.yaml）"
    )
    p.add_argument("--platform", default="dev", choices=["dev", "v85x"])
    p.add_argument("--artifact", default="")
    p.add_argument("--promote", action="store_true", help="过门则升 stable")
    args = p.parse_args(argv)

    raw = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    levels = {k: report_from_eval_dict(v) for k, v in raw.items()}
    gate = GateThresholds.from_yaml(args.gate) if args.gate else None

    register = None
    if args.register:
        register = {
            "name": args.register,
            "version": args.reg_version,
            "backbone": args.backbone or args.model_name,
            "input_size": args.input_size,
            "platform": args.platform,
            "artifact_path": args.artifact,
            "promote": args.promote,
        }
        if args.registry_index:
            register["index"] = args.registry_index

    report, gate_res, jp = run_cascade_envelope(
        levels,
        model_name=args.model_name,
        num_classes=args.num_classes,
        manifest=args.manifest,
        label=args.label,
        output_dir=args.out,
        gate=gate,
        register=register,
    )
    print(cascade_envelope_markdown(report))
    print()
    print(gate_res.to_markdown())
    print(f"\n[cascade-envelope] → {jp}")


if __name__ == "__main__":
    main()
