"""可行性包络入口（Hydra）：训练好的 checkpoint → 四级包络报告 + gate。

GPU 真训后的闭环最后一步：
    python -m edge_cam.eval.run_envelope manifest=... ckpt=outputs/.../xxx.ckpt \\
        fp32_onnx=outputs/.../efficientnet_lite0_fp32.onnx regional_json=regions/us.json

产物：report.json + report.md（逐级掉点表）+ gate 判定。INT8 级需给 fp32_onnx（就地量化）。
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from edge_cam.contracts.schemas.dataset import DatasetManifest
from edge_cam.eval.full_eval import run_full_eval
from edge_cam.eval.gates.gate import GateThresholds, evaluate_gate
from edge_cam.eval.regional import RegionalMask
from edge_cam.train.classify.module import Classifier

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs" / "eval")


def run(cfg: DictConfig) -> tuple[Path, Path]:
    """跑包络 + gate，落盘 report.json/md，返回两者路径。"""
    manifest = DatasetManifest.load(cfg.manifest)
    if cfg.get("ckpt"):
        model = Classifier.load_from_checkpoint(cfg.ckpt, map_location="cpu")
    else:  # 无 checkpoint（如 CI smoke）：随机权重，仅验证机制
        model = Classifier(cfg.model_name, num_classes=manifest.num_classes, pretrained=False)

    mask = None
    if cfg.get("regional_json"):
        taxon_of = {r.label: r.taxon_key for r in manifest.records if r.taxon_key}
        mask = RegionalMask.from_json(cfg.regional_json, manifest.class_to_idx, taxon_of)

    report = run_full_eval(
        model,
        manifest,
        input_size=cfg.input_size,
        batch_size=cfg.batch_size,
        num_workers=0,
        fp32_onnx=cfg.get("fp32_onnx"),
        output_dir=cfg.output_dir,
        regional_mask=mask,
        data_root=cfg.data.get("data_root", None) if cfg.get("data") else None,
    )
    gate = evaluate_gate(report, GateThresholds(**dict(cfg.get("gate", {}))))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / "report.json", out_dir / "report.md"
    report.save(json_path)
    md = report.to_markdown() + "\n\n" + gate.to_markdown() + "\n"
    md_path.write_text(md, encoding="utf-8")
    print(md)

    # 接通断链（架构审查 A）：cfg.register 给定则把 report+gate 固化成 ModelCard → registry，
    # gate 过则 promote 到 stable（registry.promote 内部再校验 gate_pass，双保险）。
    if cfg.get("register"):
        _publish_card(cfg, report, gate, manifest)

    return json_path, md_path


def _publish_card(cfg: DictConfig, report, gate, manifest: DatasetManifest) -> None:
    """据 cfg.register 把评估结论发布到 registry（架构审查 A 的接线点）。"""
    from edge_cam.registry.promotion import build_model_card, provenance_from_manifest, publish
    from edge_cam.registry.store import ModelRegistry

    reg = cfg.register
    card = build_model_card(
        report,
        gate,
        name=reg.get("name", cfg.model_name),
        version=reg.get("version", "v0"),
        backbone=cfg.model_name,
        num_classes=manifest.num_classes,
        input_size=cfg.input_size,
        platform=reg.get("platform", "dev"),
        artifact_path=reg.get("artifact_path", cfg.get("fp32_onnx") or ""),
        provenance=provenance_from_manifest(manifest),
    )
    registry = ModelRegistry(reg.get("index", str(Path(cfg.output_dir) / "models.yaml")))
    card = publish(registry, card, promote=bool(reg.get("promote", False)) and gate.passed)
    print(
        f"[publish] {card.name} v{card.version} → channel={card.channel} gate_pass={card.gate_pass}"
    )


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="envelope")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
