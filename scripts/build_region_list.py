"""构建地域「在场物种」清单 region.json（B.5 地域过滤消融用）。

region.json = 一个 eBird species_code 字符串数组；run_envelope 的 regional_json 吃它，
经 manifest 的 taxon_key 落到 logit 列做 mask（不在清单的类 logit 置 -inf）。

三种来源：
  1. eBird API（需免费 key）：真实地区物种清单（推荐，B.5 有意义的对比靠它）
       curl -H "X-eBirdApiToken: <KEY>" \
         "https://api.ebird.org/v2/product/spplist/US-CA" -o region_codes.json
     再： python scripts/build_region_list.py --codes-file region_codes.json \
            --map data/processed/birds525/ebird_map.csv --out regions/us_ca.json
  2. 手动清单：任意 species_code json 数组文件，同上 --codes-file。
  3. demo：从映射表里确定性取一半 code（仅验证机制可跑，**数字无地域意义**）
       python scripts/build_region_list.py --demo --map data/processed/birds525/ebird_map.csv \
         --out regions/demo.json

只保留与本数据集 ebird_map 交集的 code（清单外的类本就不在头里）。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def map_codes(map_csv: str | Path) -> list[str]:
    """ebird_map.csv → 本数据集已映射的 eBird code 列表（保序去重）。"""
    seen: list[str] = []
    with Path(map_csv).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            c = row.get("ebird_code")
            if c and c not in seen:
                seen.append(c)
    return seen


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="构建地域在场物种 region.json")
    p.add_argument("--map", required=True, help="ebird_map.csv（本数据集 label→code）")
    p.add_argument("--out", required=True, help="输出 region.json（code 字符串数组）")
    p.add_argument("--codes-file", help="eBird species_code json 数组（API/手动）")
    p.add_argument("--demo", action="store_true", help="demo：取映射 code 前一半（无地域意义）")
    p.add_argument("--fraction", type=float, default=0.5, help="demo 取比例")
    args = p.parse_args(argv)

    dataset_codes = map_codes(args.map)
    if args.demo:
        region = dataset_codes[: max(1, int(len(dataset_codes) * args.fraction))]
        note = "DEMO（无地域意义，仅验证机制）"
    elif args.codes_file:
        raw = set(json.loads(Path(args.codes_file).read_text(encoding="utf-8")))
        region = [c for c in dataset_codes if c in raw]  # 只留与数据集交集
        note = f"来自 {args.codes_file}"
    else:
        p.error("需 --demo 或 --codes-file 之一")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(region, indent=2), encoding="utf-8")
    cov = len(region) / len(dataset_codes) if dataset_codes else 0
    print(f"[region] {note}：{len(region)}/{len(dataset_codes)} 种在场（{cov:.0%}）→ {out}")


if __name__ == "__main__":
    main()
