#!/usr/bin/env python
"""分类 V2 前置：用检测器(teacher)把 classify 数据集每张图裁成「鸟框+15%外扩」。

为什么：原图鸟常只占 3~38%（杂背景/多鸟），整图训练被背景稀释。裁出鸟 → 224 输入聚焦鸟身，
且**训练=级联推理**（推理也是检测器裁框喂分类器），一举消掉 domain gap。

做法：feeder_416 检测器(ORT) → 取最高分 bird 框 → expand_to_square(+15%) → resize 存。
无 bird → 回退整图中心方形（诚实计数）。新 manifest 仅换 root（路径结构不变）→ 复用训练管线。
ORT 只有 CPU provider → **多进程并行**（每进程一个单线程会话）跑满 208 核。

用法（box，PYTHONPATH=src）：
  python results/classify/crop_dataset.py --det outputs/detect/feeder_416.onnx \
    --in-manifest /root/autodl-tmp/classify_raw/processed/manifest.json \
    --crops-root /root/autodl-tmp/classify_crops --size 256 --workers 48
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from edge_cam.cascade.adapters import _DET_MEAN, _DET_STD, decode_nanodet
from edge_cam.contracts.schemas.dataset import DatasetManifest
from edge_cam.data.crop import expand_to_square

PAD = 0.15
_G: dict = {}  # 每进程全局：session / 参数


def _init(det: str, in_root: str, crops_root: str, det_size: int, conf: float, size: int) -> None:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1  # 多进程下避免线程过订阅
    so.inter_op_num_threads = 1
    _G["sess"] = ort.InferenceSession(det, sess_options=so, providers=["CPUExecutionProvider"])
    _G["iname"] = _G["sess"].get_inputs()[0].name
    _G.update(in_root=Path(in_root), crops_root=Path(crops_root), det_size=det_size, conf=conf, size=size)


def _top_bird(im: Image.Image):
    ds = _G["det_size"]
    rgb = im.convert("RGB").resize((ds, ds), Image.BILINEAR)
    arr = np.asarray(rgb, np.float32)[:, :, ::-1]
    x = np.transpose((arr - _DET_MEAN) / _DET_STD, (2, 0, 1))[None].astype(np.float32)
    out = _G["sess"].run(None, {_G["iname"]: x})[0]
    dets = decode_nanodet(out, (im.width, im.height), input_size=ds, num_classes=5, conf_thr=_G["conf"])
    birds = [d for d in dets if d.class_id == 0]
    return max(birds, key=lambda d: d.score).box if birds else None


def _work(rel: str) -> tuple[str, str]:
    """处理一条记录 → (status, rel)。status: crop | fallback | fail。"""
    src = _G["in_root"] / rel
    dst = _G["crops_root"] / rel
    try:
        im = Image.open(src).convert("RGB")
        box = _top_bird(im)
        if box is not None:
            sq = expand_to_square(box, PAD, im.width, im.height)
            status = "crop"
        else:
            w, h = im.size
            s = min(w, h)
            sq = ((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)
            status = "fallback"
        crop = im.crop(sq).resize((_G["size"], _G["size"]), Image.BILINEAR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        crop.save(dst, quality=92)
        return status, rel
    except Exception:  # noqa: BLE001 — 单图失败不中断
        return "fail", rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--in-manifest", required=True)
    ap.add_argument("--crops-root", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--det-size", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    man = DatasetManifest.load(args.in_manifest)
    in_root = args.data_root or man.root
    rels = [r.path for r in man.records]
    n = len(rels)
    print(f"待裁 {n} 张，workers={args.workers}，in_root={in_root}", flush=True)

    counts = {"crop": 0, "fallback": 0, "fail": 0}
    bad: set[str] = set()
    with Pool(
        args.workers,
        initializer=_init,
        initargs=(args.det, in_root, args.crops_root, args.det_size, args.conf, args.size),
    ) as pool:
        for i, (status, rel) in enumerate(pool.imap_unordered(_work, rels, chunksize=32)):
            counts[status] += 1
            if status == "fail":
                bad.add(rel)
            if (i + 1) % 10000 == 0:
                print(f"[{i + 1}/{n}] {counts}", flush=True)

    crops_root = Path(args.crops_root)
    records = [r for r in man.records if r.path not in bad]
    labels = sorted({r.label for r in records})
    out_man = DatasetManifest(
        name=man.name + "_crop",
        version=man.version,
        seed=man.seed,
        root=str(crops_root),
        class_to_idx={lab: i for i, lab in enumerate(labels)},
        records=records,
    )
    out_man.save(crops_root / "manifest_crop.json")
    summary = {
        "total": n, **counts, "kept": len(records), "classes": len(labels),
        "crop_rate": round(counts["crop"] / n, 4),
    }
    (crops_root / "crop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== crop done ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
