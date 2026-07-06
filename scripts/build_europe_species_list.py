"""构建 v1 首发地域「欧洲物种类集」清单（docs/classify/06 · ADR-0008 路线 A）。

分工（呼应 registry「只管是谁、不管在哪」）：
  - **是谁 + 层级树** ← vendored eBird registry（data/taxonomy/ebird_clements_2025）
  - **在哪**          ← eBird 各国 checklist（本脚本，`spplist/{country}` 并集）

输出 `data/region/europe/`：
  - `europe_species.jsonl` —— 落 registry 的欧洲种（ebird_code + genus/family/order），
    按 taxon_order 排序；这是端侧模型的**类集 universe**（ever-recorded 上界，含迷鸟；
    「常规 vs 迷鸟」分层是后续 GBIF occurrence 频次步）。
  - `europe_summary.json` —— provenance：registry 版本、国家清单、逐国数、掉出码、日期。

用法（需免费 eBird API key）：
  EBIRD_API_KEY=... PYTHONPATH=src python scripts/build_europe_species_list.py

注：产物已提交，消费方无需 key；只有**重建**清单才需要 key + 联网。
"""

from __future__ import annotations

import datetime
import json
import os
import time
import urllib.request
from pathlib import Path

from edge_cam.data.ebird_registry import EbirdRegistry

# 核心欧洲国家（EU + EFTA + UK + 巴尔干 + 北欧；排除 RU/TR/高加索的跨洲亚洲部分，
# 避免把西伯利亚/中亚种拖进类集——聚焦首发市场地理）
EUROPE_COUNTRIES = [
    "GB",
    "IE",
    "FR",
    "DE",
    "NL",
    "BE",
    "LU",
    "ES",
    "PT",
    "IT",
    "AT",
    "CH",
    "LI",
    "DK",
    "SE",
    "NO",
    "FI",
    "IS",
    "PL",
    "CZ",
    "SK",
    "HU",
    "SI",
    "HR",
    "BA",
    "RS",
    "ME",
    "MK",
    "AL",
    "GR",
    "BG",
    "RO",
    "MD",
    "UA",
    "BY",
    "LT",
    "LV",
    "EE",
    "MT",
    "CY",
    "AD",
    "MC",
    "SM",
    "FO",
    "GI",
]

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "region" / "europe"


def fetch_country_spplist(country: str, key: str) -> list[str]:
    req = urllib.request.Request(
        f"https://api.ebird.org/v2/product/spplist/{country}",
        headers={"X-eBirdApiToken": key},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main() -> None:
    key = os.environ.get("EBIRD_API_KEY")
    if not key:
        raise SystemExit("需 EBIRD_API_KEY（免费：https://ebird.org/api/keygen）")

    union: set[str] = set()
    per_country: dict[str, int] = {}
    for cc in EUROPE_COUNTRIES:
        codes = fetch_country_spplist(cc, key)
        per_country[cc] = len(codes)
        union |= set(codes)
        time.sleep(0.15)  # eBird 礼貌限速

    reg = EbirdRegistry.load()
    present, missing = reg.coverage(sorted(union))  # present = 落 registry 的种码
    present.sort(key=lambda c: reg.species[c]["taxon_order"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "europe_species.jsonl").open("w", encoding="utf-8") as f:
        for c in present:
            d = reg.species[c]
            f.write(
                json.dumps(
                    {
                        "ebird_code": c,
                        "sci_name": d["sci_name"],
                        "genus": d["genus"],
                        "family_code": d["family_code"],
                        "family_sci": d["family_sci"],
                        "order": d["order"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    genera = {reg.species[c]["genus"] for c in present}
    families = {reg.species[c]["family_code"] for c in present}
    summary = {
        "role": "v1 首发欧洲类集 universe（ever-recorded 上界，含迷鸟）",
        "registry_version": reg.version,
        "source": "eBird checklist spplist/{country} 并集 × vendored registry",
        "fetched_at": datetime.date.today().isoformat(),
        "countries": EUROPE_COUNTRIES,
        "n_union_codes": len(union),
        "n_in_registry": len(present),
        "n_dropped": len(missing),
        "dropped_sample": missing[:20],
        "n_genera": len(genera),
        "n_families": len(families),
        "per_country": per_country,
    }
    (OUT_DIR / "europe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"欧洲类集: {len(present)} 种 / {len(genera)} 属 / {len(families)} 科 → {OUT_DIR}")
    print(f"  并集 {len(union)} 码，掉出 {len(missing)}（外来码/亚种 form/非种）")


if __name__ == "__main__":
    main()
