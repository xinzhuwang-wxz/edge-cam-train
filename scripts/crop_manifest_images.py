"""annotation-first 裁框步：检测器裁「鸟框+外扩」——上一版最大杠杆（整图→裁框 +15pt）。

放在 下载 → **裁框** → build → train 之间（GPU 侧，detector 推理）。机理：① 鸟填满输入消背景
稀释；② 训练输入=级联推理输入（部署也是检测器 crop）消 domain gap。绑 round2 检测器 = train==infer。

多框多鸟（细粒度关键坑）：GBIF 一图=一物种标签，属**主体鸟**（对焦那只，通常最高分）。
  - 取最高分 bird 框（主体）；每图记 `n_bird_boxes` → 下游 Cleanlab/QC 筛多鸟；
  - `--drop-ambiguous`：多框且首框不占优 → 疑似混种（毒标风险）丢；默认保留+打 n_bird_boxes
    （不丢数据、交 QC）。**绝不"每框各出一 crop 打同标签"**（混种批量毒标）。
  - 无 bird → 中心方形回退（诚实计数）。

⚠️ detector 必须是**裸 logits 导出**（round2 `main_416_fp32_logits.onnx`）；op13/已 bake 后处理的
导出喂 decode 会双重 sigmoid → 框数爆炸（本步有退化守卫警告）。

读 download index.csv → 裁 → 写 crops index.csv（+ n_bird_boxes 列）。用法（GPU box）：
  python scripts/crop_manifest_images.py \
    --det results/detect/round2/exports/main_416_fp32_logits.onnx \
    --in-root .../classify_raw/europe --out-root .../classify_crops/europe --pad 0.15 --size 256
"""

from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from edge_cam.cascade.adapters import _DET_MEAN, _DET_STD, decode_nanodet
from edge_cam.data.crop import expand_to_square

_G: dict = {}
CROP_FIELDS = [
    "path",
    "ebird_code",
    "scientific_name",
    "license",
    "group_key",
    "lat",
    "lon",
    "observed_at",
    "n_bird_boxes",
]


def _init(
    det: str,
    in_root: str,
    out_root: str,
    det_size: int,
    conf: float,
    size: int,
    pad: float,
    drop_ambiguous: bool,
    dom: float,
    drop_no_bird: bool,
) -> None:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    _G["sess"] = ort.InferenceSession(det, sess_options=so, providers=["CPUExecutionProvider"])
    _G["iname"] = _G["sess"].get_inputs()[0].name
    _G.update(
        in_root=Path(in_root),
        out_root=Path(out_root),
        det_size=det_size,
        conf=conf,
        size=size,
        pad=pad,
        drop_ambiguous=drop_ambiguous,
        dom=dom,
        drop_no_bird=drop_no_bird,
    )


def _bird_boxes(im: Image.Image) -> list:
    ds = _G["det_size"]
    rgb = im.convert("RGB").resize((ds, ds), Image.BILINEAR)
    arr = np.asarray(rgb, np.float32)[:, :, ::-1]
    x = np.transpose((arr - _DET_MEAN) / _DET_STD, (2, 0, 1))[None].astype(np.float32)
    out = _G["sess"].run(None, {_G["iname"]: x})[0]
    dets = decode_nanodet(
        out, (im.width, im.height), input_size=ds, num_classes=5, conf_thr=_G["conf"]
    )
    birds = [d for d in dets if d.class_id == 0]
    return sorted(birds, key=lambda d: d.score, reverse=True)


def _dominant(birds: list) -> bool:
    """首框是否明显主体（分数或面积压过第二框）——判是否疑似混种。"""
    if len(birds) < 2:
        return True
    a = birds[0]
    b = birds[1]

    def area(box) -> float:
        x1, y1, x2, y2 = box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    return a.score >= b.score + 0.15 or area(a.box) >= _G["dom"] * area(b.box)


def _work(row: dict) -> tuple[str, dict | None]:
    """→ (status, crop_row|None)。status: crop|fallback|drop_ambiguous|fail。"""
    rel = row["path"]
    src = _G["in_root"] / rel
    try:
        im = Image.open(src).convert("RGB")
        birds = _bird_boxes(im)
        n = len(birds)
        if n == 0:
            if _G["drop_no_bird"]:
                return "drop_no_bird", None  # 检测无鸟（空帧/相机陷阱误触发）→ 丢，避免空场景毒标
            w, h = im.size
            s = min(w, h)
            sq = ((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)
            status = "fallback"
        else:
            if n >= 2 and _G["drop_ambiguous"] and not _dominant(birds):
                return "drop_ambiguous", None  # 疑似混种 → 丢（毒标风险）
            sq = expand_to_square(birds[0].box, _G["pad"], im.width, im.height)
            status = "crop"
        crop = im.crop(sq).resize((_G["size"], _G["size"]), Image.BILINEAR)
        dst = _G["out_root"] / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        crop.save(dst, quality=92)
        out = {k: row.get(k, "") for k in CROP_FIELDS if k != "n_bird_boxes"}
        out["n_bird_boxes"] = n
        return status, out
    except Exception:  # noqa: BLE001 — 单图失败不中断
        return "fail", None


def main() -> None:
    ap = argparse.ArgumentParser(description="检测裁框 → crops index.csv（多鸟守卫）")
    ap.add_argument("--det", required=True, help="round2 检测器 onnx（train==inference）")
    ap.add_argument("--in-root", required=True, help="下载图根（含 index.csv）")
    ap.add_argument("--out-root", required=True, help="裁框图 + crops index.csv 落地根")
    ap.add_argument("--index", default="index.csv")
    ap.add_argument("--size", type=int, default=256, help="裁框输出边长（配 backbone 输入）")
    ap.add_argument("--det-size", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument(
        "--pad", type=float, default=0.15, help="框外扩比例（上一版 0.15；可 ablate 0.20）"
    )
    ap.add_argument("--drop-ambiguous", action="store_true", help="多框且首框不占优→丢（疑似混种）")
    ap.add_argument("--dominance", type=float, default=1.8, help="首框面积≥此倍第二框判主体")
    ap.add_argument(
        "--drop-no-bird",
        action="store_true",
        help="检测无鸟的帧丢弃（空帧/相机陷阱误触发→避免空场景毒标；欧洲流推荐开）",
    )
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()

    in_root = Path(a.in_root)
    out_root = Path(a.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with (in_root / a.index).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"裁框 {len(rows)} 图（pad={a.pad} drop_ambiguous={a.drop_ambiguous}）…", flush=True)

    stats: dict[str, int] = {}
    kept: list[dict] = []
    box_hist: dict[int, int] = {}
    init_args = (
        a.det,
        str(in_root),
        str(out_root),
        a.det_size,
        a.conf,
        a.size,
        a.pad,
        a.drop_ambiguous,
        a.dominance,
        a.drop_no_bird,
    )
    with Pool(a.workers, initializer=_init, initargs=init_args) as pool:
        for status, crop_row in pool.imap_unordered(_work, rows, chunksize=32):
            stats[status] = stats.get(status, 0) + 1
            if crop_row is not None:
                kept.append(crop_row)
                nb = int(crop_row["n_bird_boxes"])
                box_hist[nb] = box_hist.get(nb, 0) + 1

    with (out_root / a.index).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CROP_FIELDS)
        w.writeheader()
        w.writerows(kept)
    multi = sum(v for k, v in box_hist.items() if k >= 2)
    print(f"裁框统计: {stats}", flush=True)
    print(f"  多鸟图(n_boxes≥2): {multi}/{len(kept)} → n_boxes 已入 index 供 QC", flush=True)
    if kept:
        mean_boxes = sum(int(r["n_bird_boxes"]) for r in kept) / len(kept)
        if mean_boxes > 50:  # 退化守卫：正常应个位数
            print(
                f"  ⚠️ 平均 n_bird_boxes={mean_boxes:.0f} 异常高——detector 疑非裸 logits 导出"
                "（双重 sigmoid）；请用 main_416_fp32_logits.onnx",
                flush=True,
            )
    print(f"  crops index: {out_root / a.index}", flush=True)


if __name__ == "__main__":
    main()
