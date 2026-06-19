"""物种 taxonomy 归一（plan §5.2）。

taxon_key 是跨集合并、地域 mask、层级回退的规范键。真实项目应以 eBird/Clements
学名（带版本）为键；当前先用占位实现把类名规范化，**接口保持不变**，后续替换。"""

from __future__ import annotations

from typing import Protocol


class Taxonomy(Protocol):
    """label → taxon_key 的映射协议。"""

    def to_taxon_key(self, label: str) -> str | None: ...


class IdentityTaxonomy:
    """占位：把类名规范化（小写 + 折叠空白）当 taxon_key。

    ⚠️ 不是真 taxonomy——不做学名解析、不带版本、不支持层级。替换为 eBird/Clements
    映射时只需实现同一 `to_taxon_key` 接口，下游（地域 mask / 层级回退）无需改动。"""

    def to_taxon_key(self, label: str) -> str | None:
        key = " ".join(label.split()).lower()
        return key or None
