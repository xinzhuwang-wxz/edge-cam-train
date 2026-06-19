"""物种 taxonomy 归一（plan §5.2；ADR-0002）。

taxon_key 是跨集合并、地域 mask、层级回退的**规范键**。规范键 = eBird/Clements 物种 code
（带版本）。这是一个 seam：`Taxonomy` 协议 + 多个 adapter——

- `IdentityTaxonomy`：占位，把类名小写化当 key。**仅 feasibility 默认**（无 eBird 表时），
  产的不是规范 eBird 键，跨集合并/地域 mask 用它会失效（RegionalMask 已显式校验报错）。
- `EbirdTaxonomy`：真 adapter，由「源标签 → eBird code」映射表构造。**每个数据源用其映射表
  实例化一个**（BIRDS-525 俗名表、开源集学名表各一份）→ 都解析到同一套 eBird 规范键，可合并。

⚠️ 映射表是版本化数据产物（ADR-0002）：填充 BIRDS-525 全量 525 俗名→eBird 是独立数据步
（拉 eBird/Clements checklist + 匹配），此模块提供 seam 与加载器，表内容按源维护。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Protocol


class Taxonomy(Protocol):
    """label → taxon_key 的映射协议（规范键应为 eBird code）。"""

    def to_taxon_key(self, label: str) -> str | None: ...


class IdentityTaxonomy:
    """占位：把类名规范化（小写 + 折叠空白）当 taxon_key。

    ⚠️ 不是真 taxonomy——不做学名解析、不带版本、不支持层级、**对不上 eBird 地域清单**。
    仅作无 eBird 映射表时的 feasibility 默认；真键用 EbirdTaxonomy。"""

    def to_taxon_key(self, label: str) -> str | None:
        key = " ".join(label.split()).lower()
        return key or None


def _norm(label: str) -> str:
    """标签归一（大小写/空白无关匹配），与映射表键统一。"""
    return " ".join(label.split()).strip().lower()


class EbirdTaxonomy:
    """真 adapter：源标签 → eBird code（规范键，ADR-0002）。

    用一张「源标签 → eBird code」表构造（每个数据源一份表 → 一个 adapter 实例）。
    标签按 _norm 归一后匹配（大小写/空白无关）；未映射返回 None（调用方决定层级回退/丢弃，
    不静默编造键）。version 随产物记录，保证跨集稳定。"""

    def __init__(self, mapping: dict[str, str], version: str = "ebird-unversioned") -> None:
        if not mapping:
            raise ValueError("EbirdTaxonomy: 映射表为空（需 源标签→eBird code）")
        self.version = version
        self._map = {_norm(k): v for k, v in mapping.items()}

    def to_taxon_key(self, label: str) -> str | None:
        return self._map.get(_norm(label))

    @property
    def coverage_keys(self) -> set[str]:
        """表内所有 eBird code（用于校验区域清单/manifest 键集是否对齐）。"""
        return set(self._map.values())

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        label_col: str = "label",
        code_col: str = "ebird_code",
        version: str = "ebird-unversioned",
    ) -> EbirdTaxonomy:
        """从 csv（含 label,ebird_code 两列）加载映射表。

        BIRDS-525 → eBird 表即以此形态维护（label = 数据集俗名，ebird_code = eBird 物种码）。
        """
        rows = list(csv.DictReader(Path(path).read_text(encoding="utf-8").splitlines()))
        mapping = {r[label_col]: r[code_col] for r in rows if r.get(label_col) and r.get(code_col)}
        return cls(mapping, version=version)
