"""欧洲类集「逐种商用干净图覆盖」审计（数据打磨主线 · docs/classify/06 R1.2）。

**主信号 = 每种能拿到多少 commercial-clean 图** —— 这决定粒度：图够 → 种级叶子；
图少 → 折叠属级 / 待补；图 0 → cloud/回退。occurrence「常规 vs 迷鸟」的划分**放最后**
（用户 2026-07-07：迷鸟在原产地可能有大把图，别按 occurrence 先误折叠有图的种）。

信号口径（③守红线）：
  - GBIF `mediaType=StillImage` + `license=CC0_1_0,CC_BY_4_0`（去 NC）。
  - **全局**（不限欧洲）——迷鸟原产地图也能训（用户点）。
  - **扣掉 iNat**（datasetKey 50c9509d…）：iNat 是 GBIF 最大鸟图源但 ToS 禁商用 →
    真·clean 信号 = 总 CC-BY − iNat CC-BY。三个数都留（透明）。
  - GBIF occurrence 级 license 是近似（图级 license 下载时再逐条核，03 坑）——审计取近似够用。

用法（keyless，联网，~1211 种×3 调用，几分钟；建议后台）：
  PYTHONPATH=src python scripts/audit_europe_image_coverage.py

输出 data/region/europe/：
  - `europe_image_coverage.jsonl` —— 逐种 gbif_key/match/img_ccby_{total,inat,clean}
  - `europe_coverage_summary.json` —— 粒度分层分布（clean 图量分桶）
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EUROPE = Path(__file__).resolve().parents[1] / "data" / "region" / "europe"
MATCH = "https://api.gbif.org/v1/species/match"
OCC = "https://api.gbif.org/v1/occurrence/search"
INAT_DATASET = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"  # iNaturalist RG（商用禁，扣除）
CC_CLEAN = ["CC0_1_0", "CC_BY_4_0"]  # 去 NC（list → doseq 出两个 license= 参数，非逗号串）
WORKERS = 5  # GBIF 对高并发 429，克制并发 + 长退避


def _get(url: str, params: dict, retries: int = 8) -> dict:
    full = url + "?" + urllib.parse.urlencode(params, doseq=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(full, timeout=60) as r:  # noqa: S310
                return json.load(r)
        except urllib.error.HTTPError as e:  # noqa: PERF203
            if e.code == 429 and attempt < retries - 1:
                time.sleep(min(60, 4 * 2**attempt))  # 429 指数长退避
                continue
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
        except Exception:  # noqa: BLE001 — 重试瞬时网络错误
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def _count(params: dict) -> int:
    """occurrence 命中数（limit=0 只取 count）。"""
    return int(_get(OCC, {**params, "limit": 0}).get("count", 0))


def audit_one(rec: dict) -> dict:
    try:
        return _audit_one(rec)
    except Exception as e:  # noqa: BLE001 — 单种失败不带崩全局；标 ERROR 可重跑
        return {
            "ebird_code": rec["ebird_code"],
            "sci_name": rec["sci_name"],
            "genus": rec["genus"],
            "family_code": rec["family_code"],
            "gbif_key": None,
            "match_type": "ERROR",
            "error": str(e)[:80],
            "img_ccby_total": None,
            "img_ccby_inat": None,
            "img_ccby_clean": None,
        }


def _audit_one(rec: dict) -> dict:
    sci = rec["sci_name"]
    m = _get(MATCH, {"name": sci, "class": "Aves", "strict": "false"})
    key = m.get("usageKey")
    out = {
        "ebird_code": rec["ebird_code"],
        "sci_name": sci,
        "genus": rec["genus"],
        "family_code": rec["family_code"],
        "gbif_key": key,
        "match_type": m.get("matchType", "NONE"),
    }
    if not key or m.get("matchType") == "NONE":
        out.update(img_ccby_total=0, img_ccby_inat=0, img_ccby_clean=0)
        return out
    base = {"taxonKey": key, "mediaType": "StillImage", "license": CC_CLEAN}
    total = _count(base)
    inat = _count({**base, "datasetKey": INAT_DATASET})
    out.update(img_ccby_total=total, img_ccby_inat=inat, img_ccby_clean=max(0, total - inat))
    return out


def main() -> None:
    recs = [json.loads(line) for line in (EUROPE / "europe_species.jsonl").read_text().splitlines()]
    print(f"审计 {len(recs)} 种（非 iNat CC0/CC-BY 全局图量）…", flush=True)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(audit_one, recs), 1):
            results.append(r)
            if i % 100 == 0:
                print(f"  {i}/{len(recs)}", flush=True)
    results.sort(
        key=lambda d: d["img_ccby_clean"] if d["img_ccby_clean"] is not None else -1, reverse=True
    )

    with (EUROPE / "europe_image_coverage.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 粒度分层分桶（clean 图量）——种级叶子候选 vs 折叠/待补
    def bucket(n: int | None) -> str:
        if n is None:
            return "z_error"  # GBIF 调用失败，可重跑
        if n >= 500:
            return "a_rich_>=500"
        if n >= 100:
            return "b_ok_100-499"
        if n >= 30:
            return "c_thin_30-99"
        if n >= 1:
            return "d_sparse_1-29"
        return "e_none_0"

    dist: dict[str, int] = {}
    for r in results:
        dist[bucket(r["img_ccby_clean"])] = dist.get(bucket(r["img_ccby_clean"]), 0) + 1
    no_match = sum(1 for r in results if r["match_type"] == "NONE")
    n_error = sum(1 for r in results if r["match_type"] == "ERROR")
    summary = {
        "role": "逐种商用干净图覆盖（粒度主信号；常规/迷鸟划分放最后）",
        "signal": "GBIF StillImage · CC0/CC_BY · 全局 · 扣 iNat",
        "n_species": len(results),
        "n_no_gbif_match": no_match,
        "n_error": n_error,
        "clean_bucket_dist": dict(sorted(dist.items())),
        "note": "clean=总CC-BY−iNat；图级 license 下载时再逐条核（03 坑）",
    }
    (EUROPE / "europe_coverage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\n粒度分桶（clean 图量）: {summary['clean_bucket_dist']}", flush=True)
    print(f"未匹配 GBIF: {no_match}", flush=True)
    # 抽验：常见喂食器种应富，外来/迷鸟应贫
    idx = {r["sci_name"]: r for r in results}
    for name in ["Parus major", "Passer domesticus", "Turdus merula", "Struthio camelus"]:
        r = idx.get(name)
        if r:
            c, t, i = r["img_ccby_clean"], r["img_ccby_total"], r["img_ccby_inat"]
            print(f"  验 {name}: clean={c} (total={t} inat={i})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
