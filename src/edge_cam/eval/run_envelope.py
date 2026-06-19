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
from edge_cam.eval.envelope import build_envelope
from edge_cam.eval.gates.gate import GateThresholds, evaluate_gate
from edge_cam.eval.regional import RegionalMask
from edge_cam.train.classify.data import ClassifyDataModule
from edge_cam.train.classify.module import Classifier

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs" / "eval")


def run(cfg: DictConfig) -> tuple[Path, Path]:
    """跑包络 + gate，落盘 report.json/md，返回两者路径。"""
    manifest = DatasetManifest.load(cfg.manifest)
    if cfg.get("ckpt"):
        model = Classifier.load_from_checkpoint(cfg.ckpt, map_location="cpu")
    else:  # 无 checkpoint（如 CI smoke）：随机权重，仅验证机制
        model = Classifier(cfg.model_name, num_classes=manifest.num_classes, pretrained=False)

    int8_onnx = None
    if cfg.get("fp32_onnx"):
        from edge_cam.eval.quant_estimate import quantize_int8

        dm = ClassifyDataModule(manifest, input_size=cfg.input_size, num_workers=0)
        int8_onnx = quantize_int8(
            cfg.fp32_onnx, dm.train_dataloader(), Path(cfg.output_dir) / "model.int8.onnx"
        )

    mask = None
    if cfg.get("regional_json"):
        taxon_of = {r.label: r.taxon_key for r in manifest.records if r.taxon_key}
        mask = RegionalMask.from_json(cfg.regional_json, manifest.class_to_idx, taxon_of)

    report = build_envelope(
        model,
        manifest,
        input_size=cfg.input_size,
        batch_size=cfg.batch_size,
        num_workers=0,
        int8_onnx=int8_onnx,
        regional_mask=mask,
    )
    gate = evaluate_gate(report, GateThresholds(**dict(cfg.get("gate", {}))))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / "report.json", out_dir / "report.md"
    report.save(json_path)
    md = report.to_markdown() + "\n\n" + gate.to_markdown() + "\n"
    md_path.write_text(md, encoding="utf-8")
    print(md)
    return json_path, md_path


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="envelope")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
