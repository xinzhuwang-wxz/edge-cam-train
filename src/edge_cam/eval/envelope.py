"""可行性包络编排（CONTEXT.md ①→④）：把各级评估拼成一张逐级掉点报告。

① FP32 验证集 → ② 模拟 INT8(ORT-QDQ) → ③ 类现场(退化) → ④ +地域过滤。
回答「微调后的模型能不能达成目标」的核心产物。各级可独立开关（缺 ONNX 则跳过 INT8）。"""

from __future__ import annotations

from pathlib import Path

from edge_cam.contracts.schemas.dataset import DatasetManifest
from edge_cam.contracts.schemas.eval_report import EnvelopeReport, LevelResult
from edge_cam.eval.metrics import evaluate_onnx, evaluate_torch
from edge_cam.eval.regional import RegionalMask
from edge_cam.train.classify.data import ClassifyDataModule
from edge_cam.train.classify.module import Classifier


def build_envelope(
    model: Classifier,
    manifest: DatasetManifest,
    *,
    input_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 0,
    int8_onnx: str | Path | None = None,
    regional_mask: RegionalMask | None = None,
    data_root: str | None = None,
    device: str = "cpu",
    val_only: bool = False,
) -> EnvelopeReport:
    """跑各级评估，组装 EnvelopeReport。

    Args:
        model: 训练好的分类器（用于 torch 端 FP32/类现场评估）。
        int8_onnx: 给定时加 INT8 模拟级（ORT 评估，量化后 ONNX 由 quant_estimate 产出）。
        regional_mask: 给定时加「+地域过滤」级（在干净 test 上消融）。
        data_root: 换机时覆盖 manifest 记录的数据根（见 DatasetManifest.resolve_path）。
        val_only: 只跑 fp32_val（val 集）。**消融选型用**——int8/field/regional 都触 test，
            选型阶段不得碰 test（plan §B.0：test 仅最终各跑一次）。final winner 再跑全包络。
    """
    dm = ClassifyDataModule(
        manifest,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        data_root=data_root,
    )
    levels: list[LevelResult] = []

    # ① FP32 验证集（干净口径）
    m = evaluate_torch(model, dm.val_dataloader(), device=device)
    levels.append(LevelResult(name="fp32_val", top1=m.top1, top5=m.top5, n=m.n, note="干净验证集"))

    if val_only:  # 消融选型：只看 val，不碰 test（plan §B.0）
        return EnvelopeReport(
            model_name=model.hparams.model_name,
            num_classes=manifest.num_classes,
            manifest=f"{manifest.name} {manifest.version}",
            levels=levels,
        )

    # ② 模拟 INT8（ORT-QDQ；方向性，非真实 INT8）
    if int8_onnx is not None:
        mi = evaluate_onnx(str(int8_onnx), dm.test_dataloader())
        levels.append(
            LevelResult(
                name="int8_sim", top1=mi.top1, top5=mi.top5, n=mi.n, note="ORT-QDQ 模拟，非板子实测"
            )
        )

    # ③ 类现场（退化增强代理 domain gap）
    mf = evaluate_torch(model, dm.field_dataloader(), device=device)
    levels.append(
        LevelResult(
            name="field", top1=mf.top1, top5=mf.top5, n=mf.n, note="退化代理，≠真现场(plan §8)"
        )
    )

    # ④ +地域过滤（最大杠杆；在干净 test 上消融）
    if regional_mask is not None:
        mr = evaluate_torch(
            model, dm.test_dataloader(), device=device, logit_transform=regional_mask.as_transform()
        )
        levels.append(
            LevelResult(
                name="regional",
                top1=mr.top1,
                top5=mr.top5,
                n=mr.n,
                note=f"区域覆盖 {regional_mask.coverage:.1%} 类",
            )
        )

    return EnvelopeReport(
        model_name=model.hparams.model_name,
        num_classes=manifest.num_classes,
        manifest=f"{manifest.name} {manifest.version}",
        levels=levels,
    )
