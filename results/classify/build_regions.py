#!/usr/bin/env python
"""用 GBIF occurrence(免 key)给我们的 360 种建「区域在场清单」(plan §5.4)。

对每个国家/区域，逐种查 GBIF：该学名在该国是否有 ≥THRESH 条 occurrence → 在场。
输出 regions/<cc>.json = 在场学名数组(小写，与 manifest label / taxon_key 同键)。
学名键直接用我们的类标(小写)，无需 eBird —— 地域 mask 是推理后处理，键一致即可。

用法（box，需联网）：
  python results/classify/build_regions.py \
    --manifest /root/autodl-tmp/classify_raw/processed/manifest.json \
    --countries DK,GB,US --out-dir results/classify/regions --thresh 10
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

GBIF_OCC = "https://api.gbif.org/v1/occurrence/search"
GBIF_MATCH = "https://api.gbif.org/v1/species/match"


def titlecase_binomial(lab: str) -> str:
    """'passer domesticus' → 'Passer domesticus'。"""
    parts = lab.split()
    return (parts[0].capitalize() + " " + " ".join(parts[1:])).strip() if parts else lab


def _get(url, params, retries=4):
    """带退避重试的 GET（不走 proxy；GBIF 直连快）。"""
    import time

    for k in range(retries):
        try:
            r = requests.get(url, params=params, timeout=40)
            if r.status_code == 200:
                return r.json()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5 * (k + 1))
    return None


def usage_key(sci: str) -> int | None:
    """学名 → GBIF backbone usageKey（解析同义名/纠错；NONE 则 None）。"""
    j = _get(GBIF_MATCH, {"name": sci, "kingdom": "Animalia", "rank": "SPECIES"})
    if j and j.get("matchType") != "NONE" and j.get("usageKey"):
        return int(j["usageKey"])
    return None


def country_species(cc: str, facet_limit: int = 3000) -> dict[int, int]:
    """一次 facet 查询拿某国全部鸟种 {speciesKey: occurrence数}（taxonKey=212=Aves）。"""
    j = _get(
        GBIF_OCC,
        {"country": cc, "taxonKey": 212, "facet": "speciesKey", "facetLimit": facet_limit, "limit": 0},
    )
    if not j or not j.get("facets"):
        return {}
    return {int(c["name"]): int(c["count"]) for c in j["facets"][0]["counts"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--countries", required=True, help="逗号分隔 ISO2，如 DK,GB,US")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--thresh", type=int, default=10, help="≥此 occurrence 数算在场（滤偶见）")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    labels = list(json.loads(Path(args.manifest).read_text(encoding="utf-8"))["class_to_idx"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(labels)} 种；区域 {args.countries}；阈值 ≥{args.thresh}")

    # 1) 学名 → GBIF usageKey（一次，缓存）
    with ThreadPoolExecutor(args.workers) as ex:
        keys = list(ex.map(lambda lab: usage_key(titlecase_binomial(lab)), labels))
    lab_key = {lab: uk for lab, uk in zip(labels, keys) if uk}
    print(f"usageKey 解析: {len(lab_key)}/{len(labels)} 命中")

    summary = {}
    for cc in args.countries.split(","):
        cc = cc.strip()
        facet = country_species(cc)  # 1 次查询拿该国全部鸟种
        if not facet:
            print(f"  {cc}: facet 查询失败,跳过")
            continue
        present = [lab for lab, uk in lab_key.items() if facet.get(uk, 0) >= args.thresh]
        (out_dir / f"{cc}.json").write_text(json.dumps(present, ensure_ascii=False), encoding="utf-8")
        summary[cc] = {"in_region": len(present), "of_total": len(labels), "country_species": len(facet)}
        print(f"  {cc}: 在场 {len(present)}/{len(labels)} (该国共 {len(facet)} 鸟种)")

    (out_dir / "regions_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
