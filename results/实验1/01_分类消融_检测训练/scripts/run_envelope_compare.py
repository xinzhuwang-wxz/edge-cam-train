"""两候选 test 全包络对比:efficientnet_lite0@224 vs mobilenetv3_large_100@224。
逐候选: ckpt -> 导 FP32 ONNX -> run_full_eval(fp32_val/int8_sim/field) + 额外 fp32_test。
纯 int8 掉点 = fp32_test - int8_sim (同在 test 集,排除 val->test 分布差)。"""

from pathlib import Path

import torch

from edge_cam.contracts.schemas.dataset import DatasetManifest
from edge_cam.eval.full_eval import run_full_eval
from edge_cam.eval.metrics import evaluate_torch
from edge_cam.train.classify.data import ClassifyDataModule
from edge_cam.train.classify.export import export_onnx
from edge_cam.train.classify.module import Classifier

CANDS = [
    (
        "efficientnet_lite0",
        "/root/autodl-tmp/edge-cam-train/.aim/efficientnet_lite0/9fcbfe0b7c64412ab56a6da2/checkpoints/epoch=79-step=7360.ckpt",
    ),
    (
        "mobilenetv3_large_100",
        "/root/autodl-tmp/edge-cam-train/.aim/mobilenetv3_large_100/ff3eaca950bb4aa59c4a1ae1/checkpoints/epoch=79-step=7360.ckpt",
    ),
]
DATA_ROOT = "/root/autodl-tmp/data/raw/birds525"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
manifest = DatasetManifest.load("data/processed/birds525/manifest.json")
rows = []
for name, ckpt in CANDS:
    print("\n========== %s ==========" % name, flush=True)
    out = Path("/root/autodl-tmp/outputs/envelope_cmp/%s" % name)
    out.mkdir(parents=True, exist_ok=True)
    model = Classifier.load_from_checkpoint(ckpt, map_location="cpu")
    onnx_p = out / ("%s_fp32.onnx" % name)
    export_onnx(model, onnx_p, input_size=224, opset=13)
    rep = run_full_eval(
        model,
        manifest,
        input_size=224,
        batch_size=128,
        num_workers=8,
        fp32_onnx=str(onnx_p),
        output_dir=str(out),
        data_root=DATA_ROOT,
        val_only=False,
    )
    rep.save(out / "report.json")
    lv = {l.name: round(l.top1, 4) for l in rep.levels}
    # 额外: 纯 fp32 on test
    dm = ClassifyDataModule(
        manifest, input_size=224, batch_size=128, num_workers=8, data_root=DATA_ROOT
    )
    model.to(DEV)
    fp32_test = round(evaluate_torch(model, dm.test_dataloader(), device=DEV).top1, 4)
    lv["fp32_test"] = fp32_test
    rows.append((name, lv))
    print("LEVELS:", lv, flush=True)

print("\n\n############ 对比总结 ############", flush=True)
hdr = "%-26s %-11s %-11s %-11s %-11s %-11s %-9s" % (
    "backbone",
    "fp32_val",
    "fp32_test",
    "int8_sim",
    "int8掉点",
    "field",
    "field退化",
)
print(hdr, flush=True)
for name, lv in rows:
    ft = lv.get("fp32_test")
    i8 = lv.get("int8_sim")
    fld = lv.get("field")
    drop = round(ft - i8, 4) if ft is not None and i8 is not None else "-"
    fdeg = round(ft - fld, 4) if ft is not None and fld is not None else "-"
    print(
        "%-26s %-11s %-11s %-11s %-11s %-11s %-9s"
        % (name, lv.get("fp32_val"), ft, i8, drop, fld, fdeg),
        flush=True,
    )
print("=== ENVELOPE COMPARE DONE ===", flush=True)
