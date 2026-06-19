"""观鸟器（backyard bird-feeder cam）粗检测大类清单 + COCO/OIV7 映射（plan §5.1）。

盘点观鸟器实际会碰到的动物：目标 bird + 常见访客（松鼠/猫/狗）+ 后院野生动物
（浣熊/兔/鹿/狐/臭鼬/刺猬/熊）。**不含 livestock**（马/牛/羊/斑马/长颈鹿——观鸟器见不到）。

- bird 在检测层统一为单一 `bird` 大类（细分交分类器，plan §5.1）。
- 来源：COCO 10 动物里取 bird/cat/dog/bear；其余野生动物取自 OIV7。
- 跨集合并经此显式映射表（plan §C.9），避免 schema 漂移。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CoarseClass:
    """一个统一大类 + 它在各源数据集里的原始标签名。"""

    name: str
    coco: tuple[str, ...] = ()
    oiv7: tuple[str, ...] = ()


# 顺序即 class index（bird=0 优先）。可按部署场景裁剪。
FEEDER_CAM_CLASSES: list[CoarseClass] = [
    CoarseClass("bird", coco=("bird",), oiv7=("Bird",)),
    CoarseClass("squirrel", oiv7=("Squirrel",)),
    CoarseClass("cat", coco=("cat",), oiv7=("Cat",)),
    CoarseClass("dog", coco=("dog",), oiv7=("Dog",)),
    CoarseClass("raccoon", oiv7=("Raccoon",)),
    CoarseClass("rabbit", oiv7=("Rabbit",)),
    CoarseClass("deer", oiv7=("Deer",)),
    CoarseClass("fox", oiv7=("Fox",)),
    CoarseClass("skunk", oiv7=("Skunk",)),
    CoarseClass("hedgehog", oiv7=("Hedgehog",)),
    CoarseClass("bear", coco=("bear",), oiv7=("Bear",)),
]

# 核心 4 类（plan §5.1：随场景可裁剪到这几类）
CORE_CLASSES = ("bird", "squirrel", "cat", "dog")


def class_index(classes: list[CoarseClass] | None = None) -> dict[str, int]:
    """{大类名: idx}，顺序即清单顺序。"""
    classes = classes or FEEDER_CAM_CLASSES
    return {c.name: i for i, c in enumerate(classes)}


def _source_to_unified(classes: list[CoarseClass], attr: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in classes:
        for label in getattr(c, attr):
            out[label] = c.name
    return out


def coco_to_unified(classes: list[CoarseClass] | None = None) -> dict[str, str]:
    """COCO 原始标签 → 统一大类（如 'bird'→'bird'）。"""
    return _source_to_unified(classes or FEEDER_CAM_CLASSES, "coco")


def oiv7_to_unified(classes: list[CoarseClass] | None = None) -> dict[str, str]:
    """OIV7 原始标签 → 统一大类（如 'Squirrel'→'squirrel'）。"""
    return _source_to_unified(classes or FEEDER_CAM_CLASSES, "oiv7")


def source_labels(attr: str, classes: list[CoarseClass] | None = None) -> list[str]:
    """某源数据集里要拉取的原始标签列表（去重、保序）。"""
    classes = classes or FEEDER_CAM_CLASSES
    seen: list[str] = []
    for c in classes:
        for label in getattr(c, attr):
            if label not in seen:
                seen.append(label)
    return seen


@dataclass
class DetectionDataConfig:
    """检测数据下载/合并配置（feeder-cam 子集）。"""

    classes: list[str] = field(default_factory=lambda: [c.name for c in FEEDER_CAM_CLASSES])
    max_per_class: int = 1500  # 控盘：每类上限，避免拉全量数十 GB
    splits: tuple[str, ...] = ("train", "validation")
    out_dir: str = "data/processed/detection_feeder"

    def selected(self) -> list[CoarseClass]:
        names = set(self.classes)
        return [c for c in FEEDER_CAM_CLASSES if c.name in names]
