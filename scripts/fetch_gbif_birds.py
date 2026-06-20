"""GBIF 鸟图直下（分类训练源，[[ADR-0005]]）：按 datasetKey + Aves + CC0/CC-BY 查 GBIF →
每种限额 → 并行下图 → 产 `index.csv`（GbifBirdsAdapter 吃）。

为什么：iNat ToS 限商用（先用/商用前澄清）、非 iNat 公民科学（naturgucker 等，图多在 Flickr 逐条
CC-BY）是 commercial-clean 主力。GBIF API 给逐条 license + media URL + 学名 + 经纬/日期。
每种限额按 speciesKey facet → 逐种查（均衡，不被单一常见种淹没）。

下载：Flickr `_b`(1024)→`_z`(640) 省带宽；iNat URL→medium(500)。urllib 走 env http(s)_proxy
（box 上先 `source /etc/network_turbo`）。仅成功下到的图进 index → 无缺图。

用法（box，edge env）：
  python scripts/fetch_gbif_birds.py --dataset-key <key> --out <dir> --per-class-cap 500
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GBIF = "https://api.gbif.org/v1/occurrence/search"
AVES = 212  # GBIF class Aves taxonKey


def _get(params: dict, retries: int = 5) -> dict:
    """GBIF 查询，带退避重试（网络抖动/RemoteDisconnected 不应崩整个 fetch）。"""
    url = GBIF + "?" + urllib.parse.urlencode(params, doseq=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310
                return json.load(r)
        except Exception:  # noqa: BLE001 — 重试瞬时网络错误
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def _resize_url(u: str, flickr_size: str) -> str:
    """Flickr `_b.jpg`→`_<size>.jpg`（省带宽）；iNat original→medium(500)。其余原样。"""
    if "staticflickr.com" in u and u.endswith("_b.jpg"):
        return u[: -len("_b.jpg")] + f"_{flickr_size}.jpg"
    if "inaturalist" in u:
        return u.replace("/original.", "/medium.").replace("/large.", "/medium.")
    return u


def _facet_species(base: dict, max_species: int) -> list[int]:
    """speciesKey facet → 该源样本最多的 top-N speciesKey（均衡，不被单一常见种淹没）。"""
    facet = _get({**base, "facet": "speciesKey", "facetLimit": max_species, "limit": 0})
    return [c["name"] for c in facet["facets"][0]["counts"]] if facet.get("facets") else []


def _select_species(base: dict, sk: int, cap: int) -> list[dict]:
    """单种查（≤cap）→ 该种样本（学名/license/media URL/经纬/日期/观测者）。"""
    out: list[dict] = []
    got = 0
    for offset in range(0, cap, 300):
        q = {**base, "speciesKey": sk, "limit": min(300, cap - got), "offset": offset}
        page = _get(q)
        for rec in page["results"]:
            media = [m["identifier"] for m in rec.get("media", []) if m.get("identifier")]
            if not media or not rec.get("species"):
                continue
            out.append(
                {
                    "sk": sk,
                    "scientific_name": rec["species"],
                    "license": rec.get("license", ""),
                    "url": media[0],
                    "group_key": rec.get("recordedBy") or rec.get("occurrenceID") or "",
                    "lat": rec.get("decimalLatitude"),
                    "lon": rec.get("decimalLongitude"),
                    "observed_at": rec.get("eventDate"),
                }
            )
            got += 1
        if page["endOfRecords"] or got >= cap:
            break
    return out


def _download(args: tuple[dict, Path, str]) -> dict | None:
    rec, img_dir, size = args
    dst = img_dir / str(rec["sk"]) / (urllib.parse.quote(rec["url"], safe="")[-40:] + ".jpg")
    rel = str(dst.relative_to(img_dir.parent))  # 相对 out 根（manifest 可移植）
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return {**rec, "path": rel}
    try:
        urllib.request.urlretrieve(_resize_url(rec["url"], size), dst)  # noqa: S310
        return {**rec, "path": rel}
    except Exception:  # noqa: BLE001 — 个别失败跳过
        dst.unlink(missing_ok=True)
        return None


_FIELDS = ["path", "scientific_name", "license", "group_key", "lat", "lon", "observed_at"]


def fetch(
    dataset_key: str,
    out: Path,
    licenses: list[str],
    cap: int,
    max_species: int,
    jobs: int,
    size: str,
) -> Path:
    """逐种：查 → 下图 → **增量写 index**（下载即开始、进度可见、中断不丢已下）。"""
    base = {
        "datasetKey": dataset_key,
        "taxonKey": AVES,
        "mediaType": "StillImage",
        "license": licenses,
    }
    species = _facet_species(base, max_species)
    print(f"[gbif] {len(species)} species (top {max_species}); 逐种下载…", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    idx = out / "index.csv"
    total_ok = total_sel = skipped = 0
    with (
        idx.open("w", newline="", encoding="utf-8") as fh,
        ThreadPoolExecutor(max_workers=jobs) as ex,
    ):
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        fh.flush()
        for si, sk in enumerate(species, 1):
            try:
                recs = _select_species(base, sk, cap)
            except Exception as e:  # noqa: BLE001 — 单种持续失败 → 跳过该种，不崩整体
                skipped += 1
                print(f"[gbif] species {sk} skipped: {e!r}", flush=True)
                continue
            total_sel += len(recs)
            ok = [r for r in ex.map(_download, ((r, img_dir, size) for r in recs)) if r]
            for r in ok:
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in _FIELDS})
            fh.flush()  # 增量落盘 → 中断后 index.csv 仍是有效部分集
            total_ok += len(ok)
            if si % 10 == 0 or si == len(species):
                print(
                    f"[gbif] {si}/{len(species)} species | {total_ok} imgs "
                    f"({total_sel} sel, {skipped} skip)",
                    flush=True,
                )
    print(
        f"[gbif] wrote {idx}: {total_ok} imgs / {len(species)} species ({skipped} skip)",
        flush=True,
    )
    return idx


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="GBIF 鸟图直下 → index.csv")
    ap.add_argument("--dataset-key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--license", default="CC0_1_0,CC_BY_4_0")
    ap.add_argument("--per-class-cap", type=int, default=500)
    ap.add_argument("--max-species", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--size", default="z")  # Flickr: z=640 c=800 b=1024
    a = ap.parse_args(argv)
    if "http_proxy" not in __import__("os").environ:
        print("[gbif][WARN] 无 http_proxy；建议先 source /etc/network_turbo", file=sys.stderr)
    fetch(
        a.dataset_key,
        Path(a.out),
        a.license.split(","),
        a.per_class_cap,
        a.max_species,
        a.jobs,
        a.size,
    )


if __name__ == "__main__":
    main()
