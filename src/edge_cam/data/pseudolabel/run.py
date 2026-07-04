"""iNat → MD 伪标注 → 置信分层 → Label Studio 人审：一条命令串起（box/GPU 上跑）。

    python -m edge_cam.data.pseudolabel.run \
        --raw-root /root/autodl-tmp/detect_raw --max-obs 4000 \
        --md-version MDV6-apa-rtdetr --conf-hi 0.7 --conf-lo 0.2 --per-taxon-cap 40

分四步产物落 raw_root/commercial/inat_md/：
  ① images/*.jpg                    iNat 拉图（CC0/CC-BY/research/geo）
  ② inat_md_coco.json               MD 伪标注（保 score）
  ③ inat_md_coco.json(auto,md_pseudo) + review 分层 + previews/*.jpg（可视化教师框）
  ④ label_studio_tasks.json         中置信 → LS 导入任务（人审导出后 `--import-ls` 回收）

`--import-ls <export.json>` 单独跑第 ④ 步的回收：LS 人审导出 → inat_verified_coco.json
（md_human_verified）。分层阈值/许可全经纯函数（可测），下图/MD 是薄 box 步骤。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edge_cam.data.adapters.detect.inat_md import InatObs, select_inat
from edge_cam.data.pseudolabel.inat_fetch import download_inat_photos, fetch_inat_aves_obs
from edge_cam.data.pseudolabel.label_studio import from_ls_export, to_ls_tasks
from edge_cam.data.pseudolabel.md_label import run_pseudolabel
from edge_cam.data.pseudolabel.triage import triage_by_confidence

SUBPATH = "commercial/inat_md"


def render_previews(coco: dict, image_root: Path, out_dir: Path, *, limit: int = 40) -> int:
    """把伪标注框画到图上（绿=高置信 / 橙=中置信），肉眼直评教师打框质量。返回渲染数。"""
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    anns_by: dict[int, list[dict]] = {}
    for a in coco.get("annotations", []):
        anns_by.setdefault(a["image_id"], []).append(a)
    fn = {im["id"]: im["file_name"] for im in coco["images"]}
    n = 0
    for img_id, boxes in list(anns_by.items())[:limit]:
        src = image_root / Path(fn[img_id]).name
        try:
            im = Image.open(src).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        d = ImageDraw.Draw(im)
        for b in boxes:
            x, y, w, h = b["bbox"]
            color = (60, 200, 60) if b.get("score", 0) >= 0.7 else (240, 160, 40)
            d.rectangle([x, y, x + w, y + h], outline=color, width=3)
            d.text((x + 2, y + 2), f"{b.get('score', 0):.2f}", fill=color)
        im.save(out_dir / f"{img_id}.jpg")
        n += 1
    return n


def _pull_and_label(args: argparse.Namespace, base: Path) -> None:
    img_dir = base / "images"
    print(f"[inat] ① API 枚举 Aves(CC0/CC-BY/research/geo) max_obs={args.max_obs} …", flush=True)
    obs: list[InatObs] = fetch_inat_aves_obs(max_obs=args.max_obs, sleep=args.api_sleep)
    kept = select_inat(obs, per_taxon_cap=args.per_taxon_cap)
    print(
        f"[inat] 枚举 {len(obs)} → 过滤后 {len(kept)}（per-taxon≤{args.per_taxon_cap}）", flush=True
    )
    ok = download_inat_photos(kept, img_dir, jobs=args.jobs)
    print(f"[inat] ② 下图 {len(ok)}/{len(kept)} → {img_dir}", flush=True)

    print(f"[md] ③ MegaDetector({args.md_version}) 伪标注（conf≥{args.conf_lo}）…", flush=True)
    coco = run_pseudolabel(
        img_dir,
        raw_root=args.raw_root,
        out_json=base / "inat_md_raw_coco.json",
        version=args.md_version,
        conf=args.conf_lo,
    )
    tri = triage_by_confidence(coco, conf_hi=args.conf_hi, conf_lo=args.conf_lo)
    print(f"[triage] {tri.stats}", flush=True)

    # auto（md_pseudo）直接落 inat_md_coco.json（adapter 默认读它）
    (base / "inat_md_coco.json").write_text(json.dumps(tri.auto, ensure_ascii=False))
    (base / "inat_review_coco.json").write_text(json.dumps(tri.review, ensure_ascii=False))
    (base / "triage_stats.json").write_text(json.dumps(tri.stats, indent=2, ensure_ascii=False))

    # 中置信 → LS 导入任务
    tasks = to_ls_tasks(tri.review, image_url_prefix=args.ls_url_prefix)
    (base / "label_studio_tasks.json").write_text(json.dumps(tasks, ensure_ascii=False))
    print(
        f"[ls] ④ {len(tasks)} 张中置信 → label_studio_tasks.json（人审后 --import-ls 回收）",
        flush=True,
    )

    n_prev = render_previews(coco, img_dir, base / "previews", limit=args.preview_limit)
    print(f"[preview] 渲染 {n_prev} 张教师打框预览 → {base / 'previews'}", flush=True)


def _import_ls(export_path: Path, base: Path) -> None:
    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    verified = from_ls_export(tasks)
    out = base / "inat_verified_coco.json"
    out.write_text(json.dumps(verified, ensure_ascii=False))
    n_box = len(verified["annotations"])
    print(f"[ls-import] {len(verified['images'])} 图 / {n_box} 人审框 → {out}（md_human_verified）")


def main() -> None:
    ap = argparse.ArgumentParser(description="iNat→MD→分层→LS 伪标注流程")
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--max-obs", type=int, default=4000)
    ap.add_argument("--per-taxon-cap", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--api-sleep", type=float, default=1.0)
    ap.add_argument("--md-version", default="MDV6-apa-rtdetr")
    ap.add_argument("--conf-hi", type=float, default=0.7)
    ap.add_argument("--conf-lo", type=float, default=0.2)
    ap.add_argument("--preview-limit", type=int, default=40)
    ap.add_argument("--ls-url-prefix", default="/data/local-files/?d=")
    ap.add_argument("--import-ls", type=Path, default=None, help="LS 人审导出 json → 回收 verified")
    args = ap.parse_args()

    base = Path(args.raw_root) / SUBPATH
    base.mkdir(parents=True, exist_ok=True)
    if args.import_ls:
        _import_ls(args.import_ls, base)
    else:
        _pull_and_label(args, base)


if __name__ == "__main__":
    main()
