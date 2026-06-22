#!/usr/bin/env python
"""地域 mask 消融：V2 模型在各区域 in-region 子集上比 mask on/off（正确口径，regional.py）。

evaluate_regional 只在「真值∈区域」的样本上比 off vs on top-1 → 真实地域增益
（避免"把外地真值压 -inf 必错"的 artifact）。

用法（box，PYTHONPATH=src）：
  python results/classify/regional_eval.py --ckpt <best.ckpt> \
    --manifest /root/autodl-tmp/classify_crops/manifest_crop.json \
    --regions results/classify/regions/DK.json,results/classify/regions/GB.json,results/classify/regions/US.json \
    --out results/classify/regional/regional_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from edge_cam.contracts.schemas.dataset import DatasetManifest
from edge_cam.eval.regional import RegionalMask, evaluate_regional
from edge_cam.train.classify.augment import build_eval_transform
from edge_cam.train.classify.data import ManifestDataset
from edge_cam.train.classify.module import Classifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--regions", required=True, help="逗号分隔的 region json 路径")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    man = DatasetManifest.load(args.manifest)
    taxon_of = {r.label: r.taxon_key for r in man.records if r.taxon_key}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Classifier.load_from_checkpoint(args.ckpt, map_location="cpu")

    ds = ManifestDataset(man, "test", build_eval_transform(args.input_size), args.data_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    rows = []
    for rp in args.regions.split(","):
        rp = rp.strip()
        name = Path(rp).stem
        mask = RegionalMask.from_json(rp, man.class_to_idx, taxon_of)
        res = evaluate_regional(model, loader, mask, device=device)
        res.update(region=name, region_classes=len(mask.allowed_idx), coverage=round(mask.coverage, 3))
        rows.append(res)
        print(
            f"[{name}] 区域类数 {len(mask.allowed_idx)}/{mask.num_classes} "
            f"(coverage {mask.coverage:.2f}) | in_region_n={res['in_region_n']} "
            f"top1 off={res['top1_off']:.4f} on={res['top1_on']:.4f} gain={res['gain']:+.4f}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("done ->", args.out)


if __name__ == "__main__":
    main()
