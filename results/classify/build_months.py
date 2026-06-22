#!/usr/bin/env python
"""月份(物候)在场清单：GBIF 逐种月度 occurrence(可限国家)→ 某月在场物种清单。

物候 mask 同地域：训全局头，推理期按"当月在场种"缩候选。某国冬季缺夏候鸟 → 候选变窄。
对每种查 GBIF facet=month（可加 country），缓存逐月计数，再按目标月份导出 allowed 清单。

用法（box，需联网）：
  python results/classify/build_months.py --manifest <manifest.json> \
    --country DK --months 1,7 --out-dir /root/autodl-tmp/months --thresh 10
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from build_regions import GBIF_OCC, _get, titlecase_binomial, usage_key


def species_month_counts(uk: int, cc: str | None) -> dict[int, int]:
    params = {"taxonKey": uk, "facet": "month", "facetLimit": 12, "limit": 0}
    if cc:
        params["country"] = cc
    j = _get(GBIF_OCC, params)
    if not j or not j.get("facets"):
        return {}
    return {int(c["name"]): int(c["count"]) for c in j["facets"][0]["counts"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--country", default=None, help="ISO2，限定国家物候（如 DK）；空=全球")
    ap.add_argument("--months", required=True, help="逗号分隔月份，如 1,7")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--thresh", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    labels = list(json.loads(Path(args.manifest).read_text(encoding="utf-8"))["class_to_idx"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.country or "GLOBAL"
    print(f"{len(labels)} 种；国家 {tag}；月份 {args.months}")

    with ThreadPoolExecutor(args.workers) as ex:
        keys = list(ex.map(lambda lab: usage_key(titlecase_binomial(lab)), labels))
    lab_key = {lab: uk for lab, uk in zip(labels, keys) if uk}
    print(f"usageKey: {len(lab_key)}/{len(labels)}")

    items = list(lab_key.items())
    with ThreadPoolExecutor(args.workers) as ex:
        mcs = list(ex.map(lambda kv: species_month_counts(kv[1], args.country), items))
    by_lab = {lab: mc for (lab, _), mc in zip(items, mcs)}

    summary = {}
    for m in [int(x) for x in args.months.split(",")]:
        present = [lab for lab in lab_key if by_lab.get(lab, {}).get(m, 0) >= args.thresh]
        name = f"{tag}_{m:02d}"
        (out_dir / f"{name}.json").write_text(json.dumps(present, ensure_ascii=False), encoding="utf-8")
        summary[name] = {"in_month": len(present), "of_total": len(labels)}
        print(f"  月{m:02d}({tag}): 在场 {len(present)}/{len(labels)}")

    (out_dir / "months_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
