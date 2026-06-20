"""检测数据集 adapter 抽象（[[ADR-0003]] #6 / [[ADR-0004]] 5类）。

每个开源/自建数据集 = 一个自包含 adapter：声明 `DatasetSpec`
（map/license/role/清洗/split/负样本配额），实现 `load_raw()`（按各自 raw 格式解析）。
基类把**公共逻辑**（标签映射→5类、丢未映射、负样本、按 split_unit 确定性划分、写 provenance）
收口；具体 adapter 只管"怎么读 raw"（深基类 + 薄 adapter）。

`assemble()` 据 spec.role/commercial_safe 路由：训练集=可商用&train，测试集=其 test 切，
评估集=feasibility 全量。加新源 = 写一个 adapter 注册，组装/训练/评估 caller 不改。

文档：docs/detect/01-数据集.md §4。
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field

from edge_cam.contracts.schemas.detection_manifest import DetBox, DetectionManifest, DetImageRecord

# 5 类 canonical（[[ADR-0004]]）。adapter 系统与 DetectionManifest 共用此索引。
FEEDER5_CATEGORIES: dict[str, int] = {
    "bird": 0,
    "squirrel": 1,
    "cat": 2,
    "person": 3,
    "other_animal": 4,
}

Role = str  # "train" | "eval_only"


@dataclass
class DatasetSpec:
    """一个数据集的声明（map/license/角色/清洗/split/负样本）。"""

    name: str
    raw_format: str  # "coco_json" | "fiftyone_oiv7" | "roboflow" | ...
    label_map: dict[str, str]  # 源标签 → 5类名（未列入 = 丢弃该标签）
    license: str
    commercial_safe: bool
    role: Role = "train"  # train | eval_only(feasibility 仅评估)
    exhaustive: bool = True  # 非穷尽(OIV7)→未标区域 ignore/不当负样本；穷尽→可作负样本
    split_unit: str = "image"  # image | location | sequence（相机陷阱按组防泄漏）
    max_per_class: int | None = None  # 每类抽样上限（控不平衡）
    negative_quota: int = 0  # 拉多少无框负样本（empty/no-target）
    attribution: bool = False  # 是否需逐图署名清册（OIV7/Roboflow 商用）

    def __post_init__(self) -> None:
        bad = set(self.label_map.values()) - set(FEEDER5_CATEGORIES)
        if bad:
            raise ValueError(f"{self.name}: label_map 目标含非 5 类 {sorted(bad)}")
        if self.role not in ("train", "eval_only"):
            raise ValueError(f"{self.name}: role 须 train|eval_only，得 {self.role!r}")
        if self.split_unit not in ("image", "location", "sequence"):
            raise ValueError(f"{self.name}: split_unit 非法 {self.split_unit!r}")


@dataclass
class RawSample:
    """adapter 解析出的原始样本（未映射）。box 标签为**源标签**，bbox=COCO [x,y,w,h]。"""

    path: str  # 相对 raw_root（或绝对）
    width: int
    height: int
    boxes: list[tuple[str, list[float]]] = field(default_factory=list)  # (源标签, [x,y,w,h])
    group_key: str | None = None  # split 分组键（location/sequence）；None→按 path
    is_negative: bool = False  # 显式负样本（empty/no-target）


def _split_of(key: str, seed: str) -> str:
    """按 key 确定性划分 train/val/test = 70/15/15（同 group_key 必同 split，防泄漏）。"""
    h = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


class DetectionDatasetAdapter(ABC):
    """检测数据集 adapter 基类。子类只实现 load_raw()；映射/负样本/划分/溯源在基类。"""

    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @abstractmethod
    def load_raw(self) -> Iterable[RawSample]:
        """按本数据集 raw 格式解析出 RawSample（源标签未映射）。"""
        ...

    def build_records(self) -> list[DetImageRecord]:
        """RawSample → DetImageRecord（映射 5 类、丢未映射、负样本、确定 split、provenance）。"""
        s = self.spec
        recs: list[DetImageRecord] = []
        for raw in self.load_raw():
            boxes: list[DetBox] = []
            for src_label, bbox in raw.boxes:
                tgt = s.label_map.get(src_label)
                if tgt is None:
                    continue  # 未映射 → 丢弃（非穷尽源的未标区域留 ignore，此处不当框）
                boxes.append(DetBox(bbox=list(bbox), category_id=FEEDER5_CATEGORIES[tgt]))
            # 无映射框：仅穷尽源 或 显式负样本 才当负样本（非穷尽源可能漏标真目标→不当负）
            if not boxes and not (raw.is_negative or s.exhaustive):
                continue
            split = _split_of(raw.group_key or raw.path, s.name)
            recs.append(
                DetImageRecord(
                    path=raw.path,
                    split=split,  # type: ignore[arg-type]
                    width=raw.width,
                    height=raw.height,
                    boxes=boxes,
                    source=s.name,
                    license=s.license,
                )
            )
        return recs


def assemble(
    adapters: list[DetectionDatasetAdapter],
    *,
    name: str = "feeder_detect",
    version: str = "v1",
) -> dict[str, DetectionManifest]:
    """据 spec.role/commercial_safe 路由 → train(含val)/test/eval_feasibility 三份 manifest。

    - 训练集 = 可商用 & role=train 的 train+val 切（权重天生可商用，无传染）。
    - test = 同批的 test 切（留出，按 split_unit 防泄漏）。
    - eval_feasibility = role=eval_only(feasibility) 全量（仅评估，绝不进训练）。
    """
    cats = dict(FEEDER5_CATEGORIES)
    train, test, feas = [], [], []
    for a in adapters:
        recs = a.build_records()
        if a.spec.role == "eval_only":
            feas.extend(recs)
            continue
        if not a.spec.commercial_safe:
            raise ValueError(
                f"{a.name}: role=train 但 commercial_safe=False —— 不可商用数据不得进训练"
                "（应设 role=eval_only，见 docs/detect/01 §3）"
            )
        for r in recs:
            (test if r.split == "test" else train).append(r)
    mk = lambda suffix, records: DetectionManifest(  # noqa: E731
        name=f"{name}_{suffix}", version=version, categories=cats, records=records
    )
    return {
        "train": mk("train", train),
        "test": mk("test", test),
        "eval_feasibility": mk("eval_feasibility", feas),
    }


# ── adapter 注册表（加新源 = 注册一个，组装 caller 不改）──
_ADAPTERS: dict[str, type[DetectionDatasetAdapter]] = {}


def register_adapter(name: str, cls: type[DetectionDatasetAdapter]) -> None:
    _ADAPTERS[name] = cls


def get_adapter_cls(name: str) -> type[DetectionDatasetAdapter]:
    try:
        return _ADAPTERS[name]
    except KeyError:
        raise ValueError(
            f"未知检测 adapter {name!r}；可选：{sorted(_ADAPTERS)}（新源 register_adapter 注册）"
        ) from None


def available_adapters() -> list[str]:
    return sorted(_ADAPTERS)
