"""检测可行性包络入口（对等分类 run_envelope.py；[[ADR-0003]] C3 检测走统一发布链）。

box 的 NanoDet env 跑出 fp32/int8 的 COCOeval 数字（map_50/map_5095/per-class/bird_recall），
本入口**纯结构化**组装：DetectionMetrics → build_detection_report → gate → 检测版逐级掉点表
+ detect_ablation.csv 总表。不依赖 nanodet/pycocotools，本地可跑可测。

为何另写：分类 run_envelope 把 fp32/int8 评估也包进来（torch/ORT），但检测的评估依赖
NanoDet env（dataset 预处理 + head.post_process + COCOeval），注定在 box 跑；本入口只接
box 产出的数字做结构化/门控/总表——这正是此前缺的「检测端到端入口」。
`EnvelopeReport.to_markdown` 是分类(top-1/5)专用，检测在此另渲染 map 列。

CLI:
    python -m edge_cam.eval.run_detect_envelope --eval-json box_eval.json \\
        --model-name nanodet_320 --num-classes 5 --manifest "feeder v1" \\
        --label feeder_320 --out outputs/detect/feeder_320/quant [--gate gate.yaml]

eval-json 形态：{"fp32": {"map_50":..,"map_5095":..,"bird_recall_50":?,"per_class_ap":{}},
                 "int8_sim": {...}}（首键作掉点 baseline）。
"""

from __future__ import annotations

from pathlib import Path

from edge_cam.contracts.schemas.eval_report import EnvelopeReport
from edge_cam.eval.detect_metrics import DetectionMetrics, append_detection_row
from edge_cam.eval.evaluators.detect import build_detection_report
from edge_cam.eval.gates.gate import GateResult, GateThresholds, evaluate_gate

_BASELINE = "fp32"  # 检测首级名（掉点基线）；evaluators/detect 惯例


def metrics_from_eval_dict(d: dict) -> DetectionMetrics:
    """一级评估 dict → DetectionMetrics（bird_recall/per_class 缺省容错）。"""
    bird = d.get("bird_recall_50")
    return DetectionMetrics(
        map_50=float(d["map_50"]),
        map_5095=float(d["map_5095"]),
        bird_recall_50=None if bird is None else float(bird),
        per_class_ap=dict(d.get("per_class_ap") or {}),
    )


def detect_envelope_markdown(report: EnvelopeReport) -> str:
    """检测版逐级掉点表（map 列 + 相对 fp32 的 mAP 掉点）。

    分类 `EnvelopeReport.to_markdown` 渲染 top-1/5；检测指标是 map_50/map_5095(+bird_recall)，
    故另渲染——否则检测报告会出一张 top1=0.000 的废表。"""
    rows = [
        "| 级 | mAP@.5:.95 | mAP@.5 | bird_recall@.5 | vs fp32 (mAP) | 备注 |",
        "|---|---|---|---|---|---|",
    ]
    for lv in report.levels:
        if lv.name == _BASELINE:
            drop_s = "—"
        else:
            drop = report.drop_from_baseline(lv.name, baseline=_BASELINE, metric="map_5095")
            drop_s = "—" if drop is None else f"{-drop:+.3f}"  # 正=涨、负=掉
        br = lv.value("bird_recall_50")
        br_s = "—" if br is None else f"{br:.3f}"
        rows.append(
            f"| {lv.name} | {lv.value('map_5095'):.3f} | {lv.value('map_50'):.3f} "
            f"| {br_s} | {drop_s} | {lv.note} |"
        )
    header = (
        f"### 检测可行性包络 · {report.model_name} ({report.num_classes} 类) · {report.manifest}\n"
    )
    return header + "\n".join(rows)


def run_detect_envelope(
    levels: dict[str, DetectionMetrics],
    *,
    model_name: str,
    num_classes: int,
    manifest: str,
    label: str,
    output_dir: str | Path,
    gate: GateThresholds | None = None,
) -> tuple[EnvelopeReport, GateResult, Path]:
    """组装检测包络 + gate，落盘 report.json/md + detect_ablation.csv。

    返回 (report, gate, json_path)。
    levels 形如 {"fp32": DetectionMetrics, "int8_sim": DetectionMetrics}；首级作掉点 baseline。
    每级写一行进 detect_ablation.csv（label 加级后缀区分），与分类消融总表对等。"""
    report = build_detection_report(
        levels, model_name=model_name, num_classes=num_classes, manifest=manifest
    )
    gate_res = evaluate_gate(report, gate or GateThresholds())

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report.json"
    report.save(json_path)
    md = detect_envelope_markdown(report) + "\n\n" + gate_res.to_markdown() + "\n"
    (out / "report.md").write_text(md, encoding="utf-8")

    for name, dm in levels.items():
        append_detection_row(dm, f"{label}_{name}", out)

    return report, gate_res, json_path


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    p = argparse.ArgumentParser(description="检测可行性包络入口（ADR-0003 C3）")
    p.add_argument("--eval-json", required=True, help="box 产出的各级评估 json")
    p.add_argument("--model-name", required=True)
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gate", default=None, help="gate 阈值 yaml(可选)")
    args = p.parse_args(argv)

    raw = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    levels = {k: metrics_from_eval_dict(v) for k, v in raw.items()}
    gate = GateThresholds.from_yaml(args.gate) if args.gate else None
    report, gate_res, jp = run_detect_envelope(
        levels,
        model_name=args.model_name,
        num_classes=args.num_classes,
        manifest=args.manifest,
        label=args.label,
        output_dir=args.out,
        gate=gate,
    )
    print(detect_envelope_markdown(report))
    print()
    print(gate_res.to_markdown())
    print(f"\n[detect-envelope] → {jp}")


if __name__ == "__main__":
    main()
