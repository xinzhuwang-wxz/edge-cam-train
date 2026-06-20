"""构建「数据集俗名 → eBird code」映射表（ADR-0002 / B.5 地域过滤前置）。

输入：eBird taxonomy CSV（COMMON_NAME, SPECIES_CODE 列；从公共端点拉：
  curl -sL "https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=csv&cat=species" -o ebird_taxonomy.csv
）+ 一份 manifest（取其 class 列）。
输出：label,ebird_code,scientific_name 的 csv → 喂 EbirdTaxonomy.from_csv / prep 的 taxonomy_csv。

匹配 = 俗名归一（大写、去标点、折空白）后等值匹配；报未匹配项（人工补别名）。

用法：
  python scripts/build_ebird_mapping.py --taxonomy ebird_taxonomy.csv \
    --manifest data/processed/birds525/manifest.json \
    --out data/processed/birds525/ebird_map.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def norm(name: str) -> str:
    """俗名归一：大写 + & →AND + **撇号删除**（Abbott's→Abbotts，对齐 BIRDS-525 无撇号）
    + 其余非字母数字→空格 + 折空白。"""
    s = name.upper().replace("&", " AND ")
    s = s.replace("'", "").replace("’", "")  # 撇号删除而非变空格（关键）
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_ebird(path: str | Path) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
    """eBird taxonomy CSV → (按归一俗名索引, 按去空格归一索引)。

    去空格档作回退：吸收 'Oyster Catcher'↔'Oystercatcher'、'Bush Tit'↔'Bushtit' 类空格差异。
    """
    by_norm: dict[str, tuple[str, str]] = {}
    by_despace: dict[str, tuple[str, str]] = {}
    with Path(path).open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            common = row.get("COMMON_NAME") or row.get("comName")
            code = row.get("SPECIES_CODE") or row.get("speciesCode")
            sci = row.get("SCIENTIFIC_NAME") or row.get("sciName") or ""
            if common and code:
                key = norm(common)
                by_norm[key] = (code, sci)
                by_despace.setdefault(key.replace(" ", ""), (code, sci))
    return by_norm, by_despace


def manifest_labels(path: str | Path) -> list[str]:
    return list(json.loads(Path(path).read_text(encoding="utf-8"))["class_to_idx"])


def load_aliases(path: str | Path | None) -> dict[str, str]:
    """别名表 csv（label,ebird_code）→ {label: ebird_code}。人工补 148 未匹配里可救的。"""
    if not path:
        return {}
    out: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("label") and row.get("ebird_code"):
                out[row["label"]] = row["ebird_code"]
    return out


def build(taxonomy: str, manifest: str, out: str, aliases: str | None = None) -> None:
    by_norm, by_despace = load_ebird(taxonomy)
    code_to_sci = {code: sci for code, sci in by_norm.values()}
    alias_map = load_aliases(aliases)
    labels = manifest_labels(manifest)
    rows: list[tuple[str, str, str]] = []
    unmatched: list[str] = []
    n_alias = 0
    for label in labels:
        key = norm(label)
        hit = by_norm.get(key) or by_despace.get(key.replace(" ", ""))  # 精确 → 去空格回退
        if hit:
            rows.append((label, hit[0], hit[1]))
        elif label in alias_map:  # 人工别名回退
            code = alias_map[label]
            rows.append((label, code, code_to_sci.get(code, "")))
            n_alias += 1
        else:
            unmatched.append(label)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "ebird_code", "scientific_name"])
        w.writerows(rows)

    n = len(labels)
    extra = f"（含别名回退 {n_alias}）" if n_alias else ""
    print(f"[ebird-map] 匹配 {len(rows)}/{n}（{len(rows) / n:.1%}）{extra}→ {out_path}")
    if unmatched:
        print(f"[ebird-map] 未匹配 {len(unmatched)} 个（需人工补别名/确认改名）：")
        for u in unmatched[:30]:
            print(f"  - {u}")
        if len(unmatched) > 30:
            print(f"  …还有 {len(unmatched) - 30} 个")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="构建数据集俗名→eBird code 映射表")
    p.add_argument("--taxonomy", required=True, help="eBird taxonomy CSV")
    p.add_argument("--manifest", required=True, help="DatasetManifest json（取 class 列）")
    p.add_argument("--out", required=True, help="输出 label,ebird_code csv")
    p.add_argument("--aliases", help="人工别名表 csv（label,ebird_code），补未匹配项")
    args = p.parse_args(argv)
    build(args.taxonomy, args.manifest, args.out, aliases=args.aliases)


if __name__ == "__main__":
    main()
