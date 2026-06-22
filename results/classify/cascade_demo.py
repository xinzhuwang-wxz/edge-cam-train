#!/usr/bin/env python
"""级联（检测→裁鸟→细分类）联合推理 · 逐步可视化 demo。

链路（每步都标在图上）：
  原图 ─[NanoDet feeder_416 ONNX]→ ① 检测框 + 粗类 + 检测置信
       ─[crop_with_padding +15% 外扩→方形]→ ② 喂分类器的外扩框
       ─[Lite0 360种 ONNX]→ ③ 细分种 + 种置信 + top5（vs GT），含置信门控/层级回退

样本：classify_raw test（多种鸟）+ 可选 --extra-dir（squirrel/cat/person/other_animal 非鸟动物）。

用法（box，PYTHONPATH=src）：
  python results/classify/cascade_demo.py \
    --det outputs/detect/feeder_416.onnx \
    --clf outputs/classify/feeder_lite0_224_v2/efficientnet_lite0_fp32.onnx \
    --manifest /root/autodl-tmp/classify_raw/processed/manifest.json \
    --out results/classify/cascade_examples --n-bird 14 --extra-dir /root/autodl-tmp/nonbird_samples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from edge_cam.cascade.adapters import OnnxClassifier, OnnxDetector
from edge_cam.cascade.pipeline import CascadePipeline
from edge_cam.contracts.schemas.dataset import DatasetManifest
from edge_cam.data.crop import expand_to_square

DET_CLASSES = ["bird", "squirrel", "cat", "person", "other_animal"]
PAD = 0.15  # 与 CascadePipeline 默认一致（crop_with_padding 外扩）


def _font(sz: int):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def pick_bird_samples(manifest, n, data_root):
    """test split 取 n 个不同物种：(abs_path, gt_name)。"""
    seen, out = set(), []
    for r in manifest.records:
        if r.split != "test" or r.label in seen:
            continue
        p = manifest.resolve_path(r, data_root)
        if not Path(p).exists():
            continue
        seen.add(r.label)
        out.append((str(p), r.label))
        if len(out) >= n:
            break
    return out


def draw(img, det, padbox, lines, color):
    """画 ① 检测框（绿/红）+ ② 外扩框（黄）+ 顶部多行步骤标注。"""
    im = img.convert("RGB").copy()
    W = im.width
    fsz = max(15, W // 36)
    f = _font(fsz)
    d = ImageDraw.Draw(im)
    if padbox is not None:  # ② 外扩框（黄，分类器实际输入区域）
        d.rectangle(padbox, outline=(255, 210, 0), width=max(2, W // 320))
    if det is not None:  # ① 检测框（绿=种对/红=种错或回退）
        d.rectangle(det, outline=color, width=max(3, W // 200))
    bar_h = len(lines) * (fsz + 7) + 8  # 顶部黑条放步骤标注
    d.rectangle([0, 0, W, bar_h], fill=(0, 0, 0))
    y = 5
    for txt, col in lines:
        d.text((8, y), txt, fill=col, font=f)
        y += fsz + 7
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--clf", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-bird", type=int, default=14)
    ap.add_argument("--extra-dir", default=None, help="非鸟动物图目录（文件名前缀=类名，可选）")
    ap.add_argument("--det-size", type=int, default=416)
    ap.add_argument("--clf-size", type=int, default=224)
    args = ap.parse_args()

    manifest = DatasetManifest.load(args.manifest)
    idx2name = {i: lab for lab, i in manifest.class_to_idx.items()}
    detector = OnnxDetector(args.det, input_size=args.det_size, num_classes=5, conf_thr=0.3)
    classifier = OnnxClassifier(args.clf, input_size=args.clf_size)
    pipe = CascadePipeline(detector, classifier, bird_class=0, padding=PAD, clf_size=args.clf_size)

    samples = [(p, g, "bird") for p, g in pick_bird_samples(manifest, args.n_bird, args.data_root)]
    if args.extra_dir and Path(args.extra_dir).is_dir():
        for fp in sorted(Path(args.extra_dir).glob("*")):
            if fp.suffix.lower() in (".jpg", ".jpeg", ".png"):
                samples.append((str(fp), "", "nonbird"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, (path, gt, kind) in enumerate(samples):
        img = Image.open(path)
        res = pipe.infer(img)
        # 取最高分检测框（与 infer 内部 best 一致）画出来
        dets = [d for d in detector.detect(img) if d.score >= 0.3]
        det = max(dets, key=lambda d: d.score) if dets else None
        det_box = det.box if det else None
        coarse = DET_CLASSES[det.class_id] if det else "—"

        # ② 外扩框（仅 bird 进分类时画）
        padbox = None
        if res.level == "species" or (res.level == "coarse" and res.coarse_class == 0 and det):
            if det and det.class_id == 0:
                padbox = expand_to_square(det.box, PAD, img.width, img.height)

        sp = int(res.species_idx) if res.species_idx is not None else -1
        pred_name = idx2name.get(sp, "")
        correct = res.level == "species" and pred_name == gt
        color = (40, 210, 80) if correct else (235, 80, 80)

        # 逐步标注行（英文，避免 box 无 CJK 字体）
        lines = []
        if det is None:
            lines = [("(1) DET: nothing -> no report", (235, 80, 80))]
        else:
            lines.append((f"(1) DET: {coarse}  conf={det.score:.2f}", (120, 200, 255)))
            if res.level == "species":
                top5 = ", ".join(idx2name.get(i, str(i)).split()[0] for i in res.top5[:5])
                lines.append(("(2) crop bird +15% pad -> classifier", (255, 210, 0)))
                mark = "OK" if correct else "X"
                lines.append((f"(3) SPECIES: {pred_name}  conf={res.confidence:.2f}  [{mark}]", color))
                lines.append((f"    GT={gt or '?'} | top5: {top5}", (210, 210, 210)))
            elif res.coarse_class == 0:
                lines.append(("(2) crop bird +15% pad -> classifier", (255, 210, 0)))
                lines.append((f"(3) low conf/small box -> fallback to BIRD  conf={res.confidence:.2f}", (255, 180, 60)))
                if gt:
                    lines.append((f"    GT={gt}", (210, 210, 210)))
            else:
                lines.append((f"(2) non-bird -> coarse class '{coarse}' (no fine-grained)", (255, 210, 0)))

        draw(img, det_box, padbox, lines, color).save(out_dir / f"ex{k:02d}_{kind}.png")
        rows.append({
            "file": f"ex{k:02d}_{kind}.png", "kind": kind, "gt": gt,
            "det_coarse": coarse, "det_conf": round(det.score, 3) if det else None,
            "level": res.level, "pred": pred_name if res.level == "species" else f"[{res.level}]",
            "conf": round(res.confidence, 3), "correct": correct,
        })
        print(f"ex{k:02d} {kind:7s} det={coarse:12s} level={res.level:7s} pred={rows[-1]['pred']:28s} gt={gt} {'OK' if correct else ''}")

    (out_dir / "cascade_examples.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    birds = [r for r in rows if r["kind"] == "bird"]
    n_sp = sum(r["level"] == "species" for r in birds)
    n_ok = sum(r["correct"] for r in birds)
    print(f"\n鸟 {len(birds)} 例：报种 {n_sp}、正确 {n_ok}；非鸟 {len(rows) - len(birds)} 例")


if __name__ == "__main__":
    main()
