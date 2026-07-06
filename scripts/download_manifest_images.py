"""annotation-first 下半段：GPU 时读下载清单 → 下字节 → 产 index.csv（GbifBirdsAdapter 吃）。

配 `build_europe_download_manifest.py`（上半段，本地建 URL 清单，不下字节）。本脚本在**有带宽/GPU
的机器**上跑：读清单每行下图 → 存 `<out>/<ebird_code>/<hash>.jpg` → 写 `index.csv`
（path/ebird_code/scientific_name/license/group_key/lat/lon/observed_at）。

path 相对 out 根（manifest 可移植）；仅成功下到的图进 index（无缺图）。Flickr `_b`(1024)→`_z`(640)
省带宽。可续跑（文件在跳过）、线程、单张失败跳过。图级 license 若要严核，在此按响应头/EXIF 复查
（03 坑：GBIF occurrence license ≠ 图级）——首版信 occurrence license。

用法（GPU box / 有带宽处）：
  python scripts/download_manifest_images.py \
    --manifest data/region/europe/europe_download_manifest.csv --out /data/classify_raw/europe
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

INDEX_FIELDS = [
    "path",
    "ebird_code",
    "scientific_name",
    "license",
    "group_key",
    "lat",
    "lon",
    "observed_at",
]


def _resize_url(u: str, flickr_size: str) -> str:
    """Flickr `_b.jpg`→`_<size>.jpg`（省带宽）；iNat original→medium。其余原样。"""
    if "staticflickr.com" in u and u.endswith("_b.jpg"):
        return u[: -len("_b.jpg")] + f"_{flickr_size}.jpg"
    if "inaturalist" in u:
        return u.replace("/original.", "/medium.").replace("/large.", "/medium.")
    return u


def _download(row: dict, out: Path, size: str) -> dict | None:
    code = row["ebird_code"]
    name = hashlib.sha1(row["url"].encode()).hexdigest()[:16] + ".jpg"  # noqa: S324 — 仅去重命名
    dst = out / code / name
    rel = str(dst.relative_to(out))
    if dst.exists() and dst.stat().st_size > 0:  # 续跑
        return {**_index_row(row), "path": rel}
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(_resize_url(row["url"], size), dst)  # noqa: S310
    except Exception:  # noqa: BLE001 — 个别失败跳过（无缺图进 index）
        dst.unlink(missing_ok=True)
        return None
    return {**_index_row(row), "path": rel}


def _index_row(row: dict) -> dict:
    return {
        "ebird_code": row["ebird_code"],
        "scientific_name": row["scientific_name"],
        "license": row["license"],
        "group_key": row.get("group_key", ""),
        "lat": row.get("lat", ""),
        "lon": row.get("lon", ""),
        "observed_at": row.get("observed_at", ""),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="下载清单图 → index.csv")
    ap.add_argument("--manifest", required=True, help="build_europe_download_manifest.py 产的 csv")
    ap.add_argument("--out", required=True, help="图与 index.csv 落地根")
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--size", default="z", help="Flickr 尺寸 z=640 c=800 b=1024")
    ap.add_argument("--limit", type=int, default=0, help=">0 只下前 N 行（冒烟）")
    ap.add_argument(
        "--per-species-cap",
        type=int,
        default=0,
        help=">0 每种最多下 N 张（清单=可得性、下载=预算，解耦；train cap 在此调）",
    )
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Path(a.manifest).open(encoding="utf-8") as mf:
        rows = list(csv.DictReader(mf))
    if a.per_species_cap:  # 从清单容量里按训练预算取（多样性：清单已按 GBIF 顺序，够杂）
        seen: dict[str, int] = {}
        capped = []
        for r in rows:
            c = r["ebird_code"]
            if seen.get(c, 0) < a.per_species_cap:
                capped.append(r)
                seen[c] = seen.get(c, 0) + 1
        rows = capped
    if a.limit:
        rows = rows[: a.limit]
    print(f"下载 {len(rows)} 图行 → {out}（{a.jobs} 线程）…", flush=True)

    index_path = out / "index.csv"
    ok = 0
    with index_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        w.writeheader()
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            for res in ex.map(lambda r: _download(r, out, a.size), rows):
                if res:
                    w.writerow(res)
                    ok += 1
            fh.flush()
    print(f"成功 {ok}/{len(rows)} → {index_path}", flush=True)


if __name__ == "__main__":
    main()
