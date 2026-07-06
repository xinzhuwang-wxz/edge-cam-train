"""eBird/Clements 物种 registry 加载器（ADR-0002 · 命门层级树的数据地基）。

registry = 姊妹 repo `bird-tagger/taxonomy` 的产物，已 vendor 进本仓
`data/taxonomy/ebird_clements_2025/`（版本 pin 见同目录 `_meta.json`；不依赖姊妹 repo
活路径，保 provenance 干净）。它只管**「是谁」**——`ebird_code → genus/family/order`——
**不含区域/分布**（区域是独立一层，从 eBird checklist / GBIF occurrence 来，见 05 §模式1）。

本模块的职责 = 把 registry 读成 `ebird_code → 祖先键` 查询，并为命门尺子
（`eval/hierarchical.py` 的 `Hierarchy`）**按类集顺序**产出 genus/family 数组——
这正是 `hierarchical_usability` 做层级 roll-up 所缺的那座桥。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# vendored registry 默认根：<repo>/data/taxonomy/ebird_clements_2025/
# __file__ = <repo>/src/edge_cam/data/ebird_registry.py → parents[3] = <repo>
DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "taxonomy" / "ebird_clements_2025"


@dataclass(frozen=True)
class EbirdRegistry:
    """`ebird_code → {genus, family_code, family_sci, order, sci_name}` 的只读视图。

    version = 权威 + 版本 + raw_sha256 短哈希（随产物记录，保跨集/跨轮可复现）。"""

    species: dict[str, dict]
    version: str

    @classmethod
    def load(cls, root: str | Path = DEFAULT_ROOT) -> EbirdRegistry:
        root = Path(root)
        species_path = root / "species.jsonl"
        if not species_path.exists():
            raise FileNotFoundError(
                f"registry 未找到：{species_path}（应已 vendor；见 data/taxonomy/ 或 ADR-0002）"
            )
        species: dict[str, dict] = {}
        for line in species_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            species[rec["ebird_code"]] = rec
        version = cls._read_version(root / "_meta.json")
        return cls(species=species, version=version)

    @staticmethod
    def _read_version(meta_path: Path) -> str:
        if not meta_path.exists():
            return "ebird-unversioned"
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        ver = m.get("ebird_authority_ver", "?")
        sha = str(m.get("raw_sha256", ""))[:12]
        return f"{m.get('authority', 'eBird/Clements')}-{ver}-{sha}"

    def __contains__(self, code: str) -> bool:
        return code in self.species

    def genus(self, code: str) -> str | None:
        rec = self.species.get(code)
        return rec["genus"] if rec else None

    def family(self, code: str) -> str | None:
        """family_code（规范键，非学名）——层级 roll-up 用稳定 code 而非可变学名。"""
        rec = self.species.get(code)
        return rec["family_code"] if rec else None

    def coverage(self, class_codes: list[str]) -> tuple[list[str], list[str]]:
        """把类集划成 (命中 registry 的, 缺失的) 两组码——建 Hierarchy 前先审覆盖。"""
        present = [c for c in class_codes if c in self.species]
        missing = [c for c in class_codes if c not in self.species]
        return present, missing

    def hierarchy_arrays(self, class_codes: list[str]) -> tuple[list[str], list[str]]:
        """按类集顺序产出 (genus_keys, family_keys)，喂 `Hierarchy`。

        每个 class index i 对应 class_codes[i] 的属码/科码，与命门 metric 的类顺序对齐。
        缺失码**不静默编造**——直接报错（调用方须先 `coverage()` 决定丢弃/回退，
        对齐 EbirdTaxonomy 未映射返回 None 的纪律）。"""
        missing = [c for c in class_codes if c not in self.species]
        if missing:
            raise KeyError(
                f"{len(missing)} 个类码不在 registry（如 {missing[:5]}）；"
                "先用 coverage() 剔除或改用属级键，勿建残缺层级树"
            )
        genus = [self.species[c]["genus"] for c in class_codes]
        family = [self.species[c]["family_code"] for c in class_codes]
        return genus, family
