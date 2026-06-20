"""细分类数据集 adapter 抽象（[[ADR-0002]] taxonomy seam / [[ADR-0003]] / [[ADR-0005]] 许可）。

镜像 `adapters/detect/`：每源一个自注册 adapter，声明 `ClassifySpec`（license 过滤/taxonomy/role/
清洗/split），实现 `load_raw()`。基类把公共逻辑（**逐图 license 过滤**→只留 CC0/CC-BY、源标签→
taxon_key([[ADR-0002]])、丢未映射、每类限额、按 split_unit 确定性防泄漏划分、写 provenance）收口；
具体 adapter 只管"怎么读 raw"（深基类 + 薄 adapter）。

分类特有（vs 检测）：① **license 逐图过滤是一等步骤**（iNat 默认 NC，[[ADR-0005]]）
② 带经纬度/日期（防泄漏 split；区域先验另由 GBIF occurrence 建，非此图）。
`assemble()` → 训练/评估共用的 `DatasetManifest`（label = taxon_key）。文档：docs/classify/01 §3b。
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field

from edge_cam.contracts.schemas.dataset import DatasetManifest, SampleRecord
from edge_cam.data.taxonomy import IdentityTaxonomy, Taxonomy

Role = str  # "train" | "eval_only"


def normalize_license(raw: str) -> str:
    """各种 license 串 → 规范标签（CC0 / CC-BY / CC-BY-NC / -SA / -ND / 组合 / other）。

    兼容 iNat 自由文本 + GBIF 的 CC URL/枚举（如 `CC_BY_4_0`、creativecommons by/4.0 URL）。
    """
    s = (raw or "").lower()
    if "cc0" in s or "publicdomain/zero" in s or "public domain" in s:
        return "CC0"
    if "by" not in s:
        return "other"
    nc = "nc" in s or "noncommercial" in s or "non-commercial" in s
    nd = "nd" in s or "noderiv" in s
    sa = "sa" in s or "sharealike" in s or "share-alike" in s
    tag = "CC-BY" + ("-NC" if nc else "") + ("-SA" if sa else "") + ("-ND" if nd else "")
    return tag


@dataclass
class ClassifySpec:
    """一个分类数据源的声明（license 过滤 / taxonomy / 角色 / 清洗 / split）。"""

    name: str
    source: str  # provenance 溯源名
    raw_format: str  # "gbif_api" | "inat_opendata" | "imagefolder" | ...
    taxonomy: Taxonomy = field(default_factory=IdentityTaxonomy)  # 源标签 → eBird taxon_key
    license_allow: tuple[str, ...] = ("CC0", "CC-BY")  # 可商用（默认排除 NC/ND/SA）
    role: Role = "train"  # train | eval_only
    split_unit: str = "observation"  # observation|observer|location|image（防泄漏分组）
    max_per_class: int | None = None  # 每类样本上限（控长尾）

    def __post_init__(self) -> None:
        if self.role not in ("train", "eval_only"):
            raise ValueError(f"{self.name}: role 须 train|eval_only，得 {self.role!r}")
        if self.split_unit not in ("observation", "observer", "location", "image"):
            raise ValueError(f"{self.name}: split_unit 非法 {self.split_unit!r}")


@dataclass
class ClassifyRawSample:
    """adapter 解析出的原始样本（未过滤未映射）。raw_label = 源物种标签（学名/俗名/taxon）。"""

    path: str  # 相对 root（或绝对/URL）
    raw_label: str
    license: str
    group_key: str | None = None  # observation_id/observer/location（防泄漏 split）；None→按 path
    lat: float | None = None
    lon: float | None = None
    observed_at: str | None = None


def _split_of(key: str, seed: str) -> str:
    """按 key 确定性划分 train/val/test = 80/10/10（同 group_key 必同 split，防泄漏）。"""
    h = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else "val" if h < 90 else "test"


def _hash_key(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class ClassifyDatasetAdapter(ABC):
    """细分类数据源 adapter 基类。子类只实现 load_raw()；过滤/映射/限额/划分/溯源在基类。"""

    def __init__(self, spec: ClassifySpec) -> None:
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @abstractmethod
    def load_raw(self) -> Iterable[ClassifyRawSample]:
        """按本源 raw 格式解析出 ClassifyRawSample（源标签未映射、未过滤 license）。"""
        ...

    def build_records(self) -> list[SampleRecord]:
        """RawSample → SampleRecord（逐图 license 过滤 → taxon_key 映射 → 丢未映射 → 每类限额 →
        确定性防泄漏 split → provenance）。label = taxon_key（eBird code）。"""
        s = self.spec
        recs: list[SampleRecord] = []
        for raw in self.load_raw():
            if normalize_license(raw.license) not in s.license_allow:
                continue  # 逐图 license 过滤（只留可商用，[[ADR-0005]]）
            key = s.taxonomy.to_taxon_key(raw.raw_label)
            if key is None:
                continue  # 未映射到 taxon_key → 丢（不静默编造，[[ADR-0002]]）
            recs.append(
                SampleRecord(
                    path=raw.path,
                    label=key,
                    split=_split_of(raw.group_key or raw.path, s.name),  # type: ignore[arg-type]
                    source=s.source,
                    license=normalize_license(raw.license),
                    taxon_key=key,
                )
            )
        return _cap_per_class(recs, s.max_per_class)


def _cap_per_class(recs: list[SampleRecord], cap: int | None) -> list[SampleRecord]:
    """每类样本上限（控长尾）：按 path 哈希确定性留前 cap 个；None=不限。"""
    if cap is None:
        return recs
    counts: dict[str, int] = {}
    kept: list[SampleRecord] = []
    for r in sorted(recs, key=lambda r: _hash_key(r.path)):
        n = counts.get(r.label, 0)
        if n < cap:
            kept.append(r)
            counts[r.label] = n + 1
    return kept


def assemble(
    adapters: list[ClassifyDatasetAdapter], *, name: str = "feeder_classify", version: str = "v1"
) -> DatasetManifest:
    """各 adapter → 合并 DatasetManifest（按 taxon_key 合并物种类；role=train 的并集）。

    eval_only 源（如自建跨源 holdout）单独走（本轮无）。class_to_idx 由 taxon_key 全集排序生成。
    """
    records: list[SampleRecord] = []
    for a in adapters:
        if a.spec.role != "train":
            continue
        records.extend(a.build_records())
    labels = sorted({r.label for r in records})
    class_to_idx = {lab: i for i, lab in enumerate(labels)}
    return DatasetManifest(
        name=name, version=version, seed=0, class_to_idx=class_to_idx, records=records
    )


# ── adapter 注册表（加新源 = 注册一个，组装 caller 不改）──
_ADAPTERS: dict[str, type[ClassifyDatasetAdapter]] = {}


def register_adapter(name: str, cls: type[ClassifyDatasetAdapter]) -> None:
    _ADAPTERS[name] = cls


def get_adapter_cls(name: str) -> type[ClassifyDatasetAdapter]:
    try:
        return _ADAPTERS[name]
    except KeyError:
        raise ValueError(
            f"未知分类 adapter {name!r}；可选：{sorted(_ADAPTERS)}（新源 register_adapter 注册）"
        ) from None


def available_adapters() -> list[str]:
    return sorted(_ADAPTERS)


def build_adapter(name: str, raw_root: str, **overrides) -> ClassifyDatasetAdapter:
    """按名构造已注册 adapter（约定 __init__(raw_root, **overrides)）。"""
    return get_adapter_cls(name)(raw_root, **overrides)  # type: ignore[call-arg]
