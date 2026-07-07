"""欧洲类集「annotation-first 下载清单」——本地备好数据集，字节留到 GPU 再下（用户 2026-07-07）。

补分类管线缺口：现有 `fetch_gbif_birds.py` 是 datasetKey 驱动 + fetch/download 耦合；
本脚本**物种清单驱动**（读 europe_image_coverage.jsonl 的 gbif_key）+ **只建索引不下字节**。
产出的清单 = GPU 时下载器的输入（每行加 path 即成 GbifBirdsAdapter 吃的 index.csv）。

口径（③守红线，同 R1.2 审计）：GBIF `mediaType=StillImage` + `license=CC0/CC_BY`（去 NC），
**扣 iNat**（datasetKey 50c9509d，ToS 禁商用）。逐条留 license/datasetKey → 下载时再核图级 license。

字段（index.csv 超集，下载器补 path 即可）：ebird_code · scientific_name · url · license ·
  dataset_key · group_key(recordedBy,防泄漏 split) · lat · lon · observed_at

可续跑（已完成 ebird_code 跳过）、限速（5 线程 + 429 长退避）、增量落盘。

用法（keyless，联网，后台；~900 种×多页，长）：
  PYTHONPATH=src python scripts/build_europe_download_manifest.py --cap 400
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EUROPE = Path(__file__).resolve().parents[1] / "data" / "region" / "europe"
OCC = "https://api.gbif.org/v1/occurrence/search"
INAT_DATASET = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"  # iNaturalist RG（商用禁，扣除）
CC_CLEAN = ["CC0_1_0", "CC_BY_4_0"]  # 去 NC
# 只要活体观测：博物馆标本（PRESERVED_SPECIMEN 死鸟剥制）域不符须排除（08）；
# MACHINE_OBSERVATION=相机陷阱，反贴喂食器域。
BASIS = ["HUMAN_OBSERVATION", "MACHINE_OBSERVATION", "OBSERVATION"]
POLITE = 0.5  # 每请求节流（守规矩，避免触发 GBIF IP 限流；每 worker ~2 req/s）
FIELDS = [
    "ebird_code",
    "scientific_name",
    "url",
    "license",
    "dataset_key",
    "group_key",
    "lat",
    "lon",
    "observed_at",
]
_LICENSE = {
    "creativecommons.org/publicdomain/zero": "CC0_1_0",
    "creativecommons.org/licenses/by/": "CC_BY_4_0",
}


def _norm_license(u: str) -> str:
    for frag, code in _LICENSE.items():
        if frag in (u or ""):
            return code
    return u or ""


def _get(params: dict, retries: int = 8) -> dict:
    full = OCC + "?" + urllib.parse.urlencode(params, doseq=True)
    for attempt in range(retries):
        try:
            time.sleep(POLITE)  # 守规矩节流（避免 GBIF IP 限流）
            with urllib.request.urlopen(full, timeout=60) as r:  # noqa: S310
                return json.load(r)
        except urllib.error.HTTPError as e:  # noqa: PERF203
            if e.code == 429 and attempt < retries - 1:
                time.sleep(min(60, 4 * 2**attempt))
                continue
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def collect_species(sp: dict, cap: int) -> list[dict]:
    """单种分页取非 iNat CC0/CC-BY 图（≤cap）→ 清单行（不下字节）。"""
    key = sp["gbif_key"]
    rows: list[dict] = []
    seen_urls: set[str] = set()
    base = {
        "taxonKey": key,
        "mediaType": "StillImage",
        "license": CC_CLEAN,
        "basisOfRecord": BASIS,  # 排标本、只活体观测（08）
    }
    offset = 0
    while len(rows) < cap:
        page = _get({**base, "limit": 300, "offset": offset})
        for rec in page.get("results", []):
            if rec.get("datasetKey") == INAT_DATASET:
                continue  # 扣 iNat
            media = [m["identifier"] for m in rec.get("media", []) if m.get("identifier")]
            if not media:
                continue
            url = media[0]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            rows.append(
                {
                    "ebird_code": sp["ebird_code"],
                    "scientific_name": rec.get("species") or sp["sci_name"],
                    "url": url,
                    "license": _norm_license(rec.get("license", "")),
                    "dataset_key": rec.get("datasetKey", ""),
                    "group_key": rec.get("recordedBy") or rec.get("occurrenceID") or "",
                    "lat": rec.get("decimalLatitude"),
                    "lon": rec.get("decimalLongitude"),
                    "observed_at": rec.get("eventDate"),
                }
            )
            if len(rows) >= cap:
                break
        offset += 300
        if page.get("endOfRecords") or offset >= 10000:  # GBIF offset 上限保护
            break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=400, help="每种最多收多少图（annotation-first）")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    species = [
        json.loads(line)
        for line in (EUROPE / "europe_image_coverage.jsonl").read_text().splitlines()
    ]
    species = [s for s in species if s.get("gbif_key") and (s.get("img_ccby_clean") or 0) > 0]

    manifest_path = EUROPE / "europe_download_manifest.csv"
    done: set[str] = set()
    if manifest_path.exists():  # 续跑：跳过已收的种
        with manifest_path.open(encoding="utf-8") as fh:
            done = {row["ebird_code"] for row in csv.DictReader(fh)}
    todo = [s for s in species if s["ebird_code"] not in done]
    print(f"清单 {len(species)} 种（clean>0），已完成 {len(done)}，本次收 {len(todo)}…", flush=True)

    lock = threading.Lock()
    new = not manifest_path.exists()
    fh = manifest_path.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    total_rows = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(collect_species, s, args.cap): s for s in todo}
        for fut, s in list(futs.items()):
            try:
                rows = fut.result()
            except Exception as e:  # noqa: BLE001 — 单种失败不带崩，续跑可补
                print(f"  ! {s['ebird_code']} {s['sci_name']}: {str(e)[:60]}", flush=True)
                continue
            with lock:
                for r in rows:
                    w.writerow(r)
                fh.flush()
                total_rows += len(rows)
                completed += 1
                if completed % 50 == 0:
                    print(f"  {completed}/{len(todo)} 种, 累计 {total_rows} 图行", flush=True)
    fh.close()

    # 汇总（小文件，可提交）
    with manifest_path.open(encoding="utf-8") as f2:
        all_rows = list(csv.DictReader(f2))
    per_species: dict[str, int] = {}
    per_dataset: dict[str, int] = {}
    for r in all_rows:
        per_species[r["ebird_code"]] = per_species.get(r["ebird_code"], 0) + 1
        per_dataset[r["dataset_key"]] = per_dataset.get(r["dataset_key"], 0) + 1
    summary = {
        "role": "annotation-first 下载清单（欧洲类集；字节留 GPU 下）",
        "cap_per_species": args.cap,
        "n_species_with_images": len(per_species),
        "n_image_rows": len(all_rows),
        "top_datasets": dict(sorted(per_dataset.items(), key=lambda kv: -kv[1])[:12]),
        "note": "url 索引非字节；下载器补 path→index.csv→GbifBirdsAdapter；图级 license 下载再核",
    }
    (EUROPE / "europe_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\n清单落盘: {len(all_rows)} 图行 / {len(per_species)} 种 → {manifest_path}", flush=True)
    print(f"  top datasets: {list(summary['top_datasets'].items())[:5]}", flush=True)


if __name__ == "__main__":
    main()
