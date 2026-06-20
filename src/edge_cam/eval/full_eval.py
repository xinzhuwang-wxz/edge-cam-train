"""完整评估编排 seam（架构审查 B）：把「(可选)量化 INT8 + 四级包络」收成一处。

此前 run_envelope 与 ablation/runner 各自手拼「建 DataModule → quantize_int8 → build_envelope」，
且 ablation 漏了 INT8 级。统一到 run_full_eval：两个调用方传同样的入参、拿同样的 EnvelopeReport。
gate 判定是上层策略，留给调用方（run_envelope 接 gate、ablation 只汇总各级 top-k）。
"""

from __future__ import annotations

from pathlib import Path

from edge_cam.contracts.schemas.dataset import DatasetManifest
from edge_cam.contracts.schemas.eval_report import EnvelopeReport
from edge_cam.eval.envelope import build_envelope
from edge_cam.eval.regional import RegionalMask
from edge_cam.train.classify.data import ClassifyDataModule
from edge_cam.train.classify.module import Classifier


def run_full_eval(
    model: Classifier,
    manifest: DatasetManifest,
    *,
    input_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 0,
    fp32_onnx: str | Path | None = None,
    output_dir: str | Path | None = None,
    regional_mask: RegionalMask | None = None,
    data_root: str | None = None,
    device: str | None = None,
    val_only: bool = False,
    quant_method: str = "ort_qdq_per_channel",
) -> EnvelopeReport:
    """跑完整四级包络：fp32_onnx 给定则先 ORT-QDQ 量化出 INT8 级，再 build_envelope。

    单一编排点 —— run_envelope 与 ablation 都调它，避免重复拼装与 ablation 漏 INT8 级。
    device=None 自动选 GPU（torch 各级评测大幅提速；int8_sim 的 ORT 仍走 CPU）。
    val_only=True：只在 val 上评 fp32（消融选型，不碰 test，plan §B.0）；跳过量化。
    """
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    int8_onnx: Path | None = None
    if fp32_onnx and not val_only:
        from edge_cam.eval.quantizers import get_quantizer  # 量化 seam(config 切换量化法)

        dm = ClassifyDataModule(
            manifest, input_size=input_size, num_workers=num_workers, data_root=data_root
        )
        out_dir = Path(output_dir or ".")
        out_dir.mkdir(parents=True, exist_ok=True)  # 量化产物写盘前确保目录在（实跑发现）
        out = out_dir / "model.int8.onnx"
        int8_onnx = get_quantizer(quant_method).quantize(str(fp32_onnx), dm.train_dataloader(), out)

    return build_envelope(
        model,
        manifest,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        int8_onnx=int8_onnx,
        regional_mask=regional_mask,
        data_root=data_root,
        device=device,
        val_only=val_only,
    )
