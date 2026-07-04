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
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

# FEEDER5_CATEGORIES 规范源在 contracts 层（detection_manifest），此处导入 + 再导出（层次正确）。
from edge_cam.contracts.schemas.detection_manifest import (
    FEEDER5_CATEGORIES,
    DetBox,
    DetectionManifest,
    DetImageRecord,
)

Role = str  # "train" | "eval_only"

# acquire method 闭集（ADR-0006 D1）：数据从哪来、怎么来的规范方法。
ACQUIRE_METHODS = frozenset(
    {"lila_http", "s3_direct", "inat_open_data", "roboflow", "local_mirror", "manual"}
)


@dataclass
class AcquireSpec:
    """获取声明（ADR-0006 D1）：数据从哪来的单一事实源，落在 adapter 的 DatasetSpec 里。"""

    method: str  # ACQUIRE_METHODS 之一
    urls: list[str] = field(default_factory=list)  # 规范下载地址（archive/annotation/bucket 前缀）
    version: str = ""  # 数据集快照标识（日期/release tag）
    archive_sha256: dict[str, str] = field(default_factory=dict)  # {basename: sha256} 下载后校验

    def __post_init__(self) -> None:
        if self.method not in ACQUIRE_METHODS:
            raise ValueError(
                f"acquire method 非法 {self.method!r}（允许：{sorted(ACQUIRE_METHODS)}）"
            )


@dataclass
class AcquireReceipt:
    """获取收据（ADR-0006 D2）：落 `_acquire.json`，记"这批 raw 怎么来的"→ 可复现 + 可审计。"""

    source: str
    method: str
    urls: list[str]
    version: str
    archive_sha256: dict[str, str]
    downloaded_at: str  # caller 传入时间戳（纯函数化，便于测试）
    image_count: int = 0
    box_count: int = 0
    tool_versions: dict[str, str] = field(default_factory=dict)


@dataclass
class DatasetSpec:
    """一个数据集的声明（map/license/角色/清洗/split/负样本/获取）。"""

    name: str
    raw_format: str  # "coco_json" | "fiftyone_oiv7" | "roboflow" | ...
    label_map: dict[str, str]  # 源标签 → 5类名（未列入 = 丢弃该标签）
    license: str
    commercial_safe: bool
    role: Role = "train"  # train | eval_only(feasibility 仅评估)
    exhaustive: bool = True  # 非穷尽(OIV7)→未标区域 ignore/不当负样本；穷尽→可作负样本
    split_unit: str = "image"  # image | location | sequence（相机陷阱按组防泄漏）
    # 每类**含该类图**上限（控不平衡，None=不限）。int=所有类同一上限；
    # dict{类名:上限}=按类指定（只压指定类，未列入的类不限）——如 {"other_animal": 18000} 只压它。
    max_per_class: int | dict[str, int] | None = None
    negative_quota: int | None = 0  # 保留多少无框负样本（0=不留，None=全留，N=确定性留前 N）
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)  # train/val/test（可配）
    attribution: bool = False  # 是否需逐图署名清册（OIV7/Roboflow 商用）
    # 未在 label_map 命中的源标签的兜底映射（None=丢弃）。用于"整集同一粗类"的源
    # （如 Roboflow feeder 集 36 鸟种全 → bird），免逐个硬列、免漏种。
    catch_all_label: str | None = None
    # 框最小面积占图比（0=不滤）。滤掉远景小框，贴合观鸟器"鸟落镜头前"的中等尺度。
    # 安全：正样本图若因滤空 → 丢弃该图（非负样本），防假负污染。
    min_box_area_frac: float = 0.0
    # 数据集级署名（逐图 author 缺省时的回退，如 Roboflow 聚合集 → 引数据集本身，兑现 CC-BY §4）。
    default_author: str | None = None
    acquire: AcquireSpec | None = None  # 获取声明（ADR-0006 D1）；None=该 adapter 暂未声明获取

    def __post_init__(self) -> None:
        bad = set(self.label_map.values()) - set(FEEDER5_CATEGORIES)
        if bad:
            raise ValueError(f"{self.name}: label_map 目标含非 5 类 {sorted(bad)}")
        if self.catch_all_label is not None and self.catch_all_label not in FEEDER5_CATEGORIES:
            raise ValueError(f"{self.name}: catch_all_label 非 5 类 {self.catch_all_label!r}")
        if self.role not in ("train", "eval_only"):
            raise ValueError(f"{self.name}: role 须 train|eval_only，得 {self.role!r}")
        if self.split_unit not in ("image", "location", "sequence"):
            raise ValueError(f"{self.name}: split_unit 非法 {self.split_unit!r}")
        if isinstance(self.max_per_class, dict):
            bad_cap = set(self.max_per_class) - set(FEEDER5_CATEGORIES)
            if bad_cap:
                raise ValueError(f"{self.name}: max_per_class 含非 5 类键 {sorted(bad_cap)}")


@dataclass
class RawSample:
    """adapter 解析出的原始样本（未映射）。box 标签为**源标签**，bbox=COCO [x,y,w,h]。"""

    path: str  # 相对 raw_root（或绝对）
    width: int
    height: int
    boxes: list[tuple[str, list[float]]] = field(default_factory=list)  # (源标签, [x,y,w,h])
    group_key: str | None = None  # split 分组键（location/sequence）；None→按 path
    is_negative: bool = False  # 显式负样本（empty/no-target）
    # 逐样本溯源（ADR-0006 D4）：adapter 有则填，流到 DetImageRecord（兑现 CC-BY 逐图署名）。
    author: str | None = None
    original_url: str | None = None
    source_media_id: str | None = None
    asset_sha256: str | None = None
    # 该图所有框的来源（ADR-0006 D7）：gt=真标注 / md_pseudo=MD 伪标 / md_human_verified=人审通过。
    label_provenance: str = "gt"


def _split_of(key: str, seed: str, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> str:
    """按 key 确定性划分 train/val/test（同 group_key 必同 split，防泄漏）；ratios 可配。"""
    h = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), 16) % 100
    t = ratios[0] * 100
    return "train" if h < t else "val" if h < t + ratios[1] * 100 else "test"


def _clip_bbox(bbox: list[float], w: int, h: int) -> list[float] | None:
    """COCO [x,y,bw,bh] 裁到图内（抓 CCT 式越界/负坐标）；裁后退化(零负面积)→ None（丢）。
    图尺寸未知(w/h≤0)时原样返回（无从裁）。"""
    if w <= 0 or h <= 0:
        return bbox
    x, y, bw, bh = bbox
    x2, y2 = min(x + bw, float(w)), min(y + bh, float(h))
    x, y = max(0.0, x), max(0.0, y)
    if x2 - x <= 0 or y2 - y <= 0:
        return None
    return [x, y, x2 - x, y2 - y]


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
        """RawSample → DetImageRecord（映射 5 类、丢未映射、负样本、确定 split、provenance）。

        负样本(0 框)仅穷尽源或显式 is_negative 才保留，受 negative_quota 限额；正样本受
        max_per_class 每类限额。两处限额均按 path 哈希确定性抽样（可复现、与 split 解耦）。
        """
        s = self.spec
        pos: list[DetImageRecord] = []
        neg: list[DetImageRecord] = []
        for raw in self.load_raw():
            boxes: list[DetBox] = []
            had_mapped = False  # 该图是否**原本**有映射框（滤 tiny 前）→ 防滤空当假负
            img_area = raw.width * raw.height
            for src_label, bbox in raw.boxes:
                tgt = s.label_map.get(src_label, s.catch_all_label)
                if tgt is None:
                    continue  # 未映射且无兜底 → 丢弃（非穷尽源未标区域留 ignore，此处不当框）
                had_mapped = True
                filter_on = s.min_box_area_frac > 0 and img_area > 0
                if filter_on and (bbox[2] * bbox[3]) / img_area < s.min_box_area_frac:
                    continue  # 远景小框 → 滤（贴合喂食器中等尺度）
                bbox = _clip_bbox(
                    bbox, raw.width, raw.height
                )  # 裁到图内（抓 CCT 式越界，退化则丢）
                if bbox is None:
                    continue
                boxes.append(
                    DetBox(
                        bbox=list(bbox),
                        category_id=FEEDER5_CATEGORIES[tgt],
                        label_provenance=raw.label_provenance,  # 框来源（ADR-0006 D7）
                    )
                )
            rec = DetImageRecord(
                path=raw.path,
                split=_split_of(raw.group_key or raw.path, s.name, s.split_ratios),  # type: ignore[arg-type]
                width=raw.width,
                height=raw.height,
                boxes=boxes,
                source=s.name,
                license=s.license,
                # 逐样本署名（ADR-0006 D4）：raw 有则带上，缺省回退数据集级 default_author。
                author=raw.author or s.default_author,
                original_url=raw.original_url,
                source_media_id=raw.source_media_id,
                asset_sha256=raw.asset_sha256,
            )
            if boxes:
                pos.append(rec)
            elif had_mapped:
                continue  # 原有框但全被 tiny 滤空 → 丢弃（非负样本，防假负污染 §5.1）
            elif raw.is_negative or s.exhaustive:
                neg.append(rec)
            # else: 非穷尽源的纯未映射图 → 丢（可能漏标真目标，§5.1 防污染负样本）
        return _cap_per_class(pos, s.max_per_class) + _cap_negatives(neg, s.negative_quota)

    def raw_dir(self, raw_root: str | Path) -> Path:
        """本源 raw 目录 raw_root/<layer>/<name>（layer=commercial|feasibility，D1）。"""
        layer = "commercial" if self.spec.commercial_safe else "feasibility"
        return Path(raw_root) / layer / self.name

    def acquire(self, raw_root: str | Path, *, now: str) -> AcquireReceipt:
        """按 spec.acquire 下载/校验 raw + 落 `_acquire.json` 收据（ADR-0006 D2/D3）。

        method=manual：只校验 raw 就位 + archive_sha256，缺失抛可执行错误（不静默放行）。
        其他 method：子类覆写 `_fetch()` 实下载；基类统一校验 + 收据。**幂等**（已校验则跳下载）。
        `now` 由调用方传入（时间戳纯函数化，便于测试/复现）。
        """
        acq = self.spec.acquire
        if acq is None:
            raise ValueError(f"{self.name}: 未声明 acquire（DatasetSpec.acquire=None）")
        dest = self.raw_dir(raw_root)
        dest.mkdir(parents=True, exist_ok=True)
        # 幂等门：**有声明 checksum 且全通过**才跳下载；否则必 fetch。
        # 修 bug：动态源 archive_sha256 空 → `_checksums_ok` 空真 → 曾误跳 `_fetch` 致 0 图。
        # 无 checksum=无法证明已下完 → 必 fetch（`_fetch` 内部自幂等：跳已下）。
        if acq.method != "manual" and not (
            acq.archive_sha256 and self._checksums_ok(dest, acq.archive_sha256)
        ):
            self._fetch(dest)  # 子类网络下载
        self._verify_checksums(dest, acq.archive_sha256)
        receipt = AcquireReceipt(
            source=self.name,
            method=acq.method,
            urls=list(acq.urls),
            version=acq.version,
            archive_sha256=dict(acq.archive_sha256),
            downloaded_at=now,
        )
        (dest / "_acquire.json").write_text(
            json.dumps(asdict(receipt), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return receipt

    def _fetch(self, dest: Path) -> None:
        """网络下载（非 manual 源由子类按 method 覆写）；base 未覆写即报错。"""
        raise NotImplementedError(
            f"{self.name}: method={self.spec.acquire.method} 需在 adapter 覆写 _fetch()"  # type: ignore[union-attr]
        )

    @staticmethod
    def _checksums_ok(dest: Path, sha: dict[str, str]) -> bool:
        """全部声明的 archive 均存在且 sha256 匹配（空声明视为已就位，供幂等短路）。"""
        return all(
            (dest / n).exists() and _sha256_file(dest / n) == want for n, want in sha.items()
        )

    def _verify_checksums(self, dest: Path, sha: dict[str, str]) -> None:
        """校验声明的 archive：缺失抛可执行错误（含下载 URL），哈希不符抛错。"""
        for name, want in sha.items():
            f = dest / name
            if not f.exists():
                urls = self.spec.acquire.urls if self.spec.acquire else []  # type: ignore[union-attr]
                raise FileNotFoundError(f"{self.name}: 缺 {name}（期望在 {dest}）。请获取：{urls}")
            got = _sha256_file(f)
            if got != want:
                raise ValueError(
                    f"{self.name}: {name} sha256 不符（期望 {want[:12]}… 得 {got[:12]}…）"
                )


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """流式分块算文件 sha256（大 archive 友好）。"""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _hash_key(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()


def _cap_negatives(neg: list[DetImageRecord], quota: int | None) -> list[DetImageRecord]:
    """负样本限额：None=全留，<=0=不留，N=按 path 哈希确定性留前 N。"""
    if quota is None:
        return neg
    if quota <= 0:
        return []
    return sorted(neg, key=lambda r: _hash_key(r.path))[:quota]


def _cap_per_class(
    pos: list[DetImageRecord], cap: int | dict[str, int] | None
) -> list[DetImageRecord]:
    """每类限额：含该类的图 ≤ 上限（多类图任一类无上限或未满即留，**不误伤欠采样类**）；
    确定性按 path 哈希。cap=int → 所有类同一上限；cap=dict{类名:上限} → 仅压指定类，余者不限。"""
    if cap is None:
        return pos
    if isinstance(cap, dict):
        cap_by_id = {FEEDER5_CATEGORIES[n]: c for n, c in cap.items()}
    else:
        cap_by_id = dict.fromkeys(FEEDER5_CATEGORIES.values(), cap)
    counts: dict[int, int] = dict.fromkeys(cap_by_id, 0)
    kept: list[DetImageRecord] = []
    for r in sorted(pos, key=lambda r: _hash_key(r.path)):
        classes = {b.category_id for b in r.boxes}
        if any(c not in cap_by_id or counts[c] < cap_by_id[c] for c in classes):
            kept.append(r)
            for c in classes:
                if c in counts:
                    counts[c] += 1
    return kept


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


def build_adapter(name: str, raw_root: str, **overrides) -> DetectionDatasetAdapter:
    """按名构造已注册 adapter。约定：具体 adapter __init__(raw_root, **overrides)。

    overrides 透传给 adapter（如 negative_quota/max_per_class/splits）。配置驱动组装的统一入口
    （build CLI 只认名字 + raw_root + overrides，不 import 具体类）。
    """
    return get_adapter_cls(name)(raw_root, **overrides)  # type: ignore[call-arg]
