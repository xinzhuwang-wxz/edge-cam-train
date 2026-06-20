"""级联推理 + 评估（[[ADR-0003]] C2）。

infer：检测 → 取 bird 框 → 最小尺寸门控 → crop → 分类 → **置信门控 + 层级回退**
(种置信不足/框太小 → 回退报粗类 bird;非鸟直接出粗类)。
evaluate：在带 species GT 的数据集上算 检出率 / 级联 top-1 / 回退率。

检测 decode 在 `Detector` seam 之后(端侧 A7 OpenCV / 离线 ORT+NanoDet 各自实现 adapter);
分类在 `Classifier` seam 之后(ONNX)。两个 seam 让级联编排可单测(fake),不依赖检测 env。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image

from edge_cam.data.crop import Box, crop_with_padding, passes_min_size


@dataclass
class Detection:
    """一个检测结果:框(原图像素 x1y1x2y2)+ 粗类 id + 置信。"""

    box: Box
    class_id: int
    score: float


class Detector(Protocol):
    """检测 seam:整图 → 检测列表(已 decode/NMS 到原图坐标)。"""

    def detect(self, image: Image.Image) -> list[Detection]: ...


class Classifier(Protocol):
    """细分类 seam:crop → (top1 类索引, 置信, top5 索引)。"""

    def classify(self, crop: Image.Image) -> tuple[int, float, list[int]]: ...


@dataclass
class CascadeResult:
    """级联单帧结论。level: species(报种) / coarse(报粗类) / none(无目标)。"""

    level: str
    coarse_class: int | None = None  # 粗类 id(如 bird)
    species_idx: int | None = None  # 细分类种索引(仅 level==species)
    confidence: float = 0.0
    top5: list[int] = field(default_factory=list)


@dataclass
class CascadeReport:
    """级联评估:检出率 / 级联 top-1(种) / 回退率。"""

    n: int
    bird_hit_rate: float
    cascade_top1: float  # 仅在 bird-detected 子集上算的种准确率(报种或回退均计入分母)
    fallback_rate: float  # 检出 bird 但回退报粗类的比例


class CascadePipeline:
    """粗检测 → crop → 细分类,贯穿置信门控 + 层级回退。"""

    def __init__(
        self,
        detector: Detector,
        classifier: Classifier,
        *,
        bird_class: int = 0,
        det_conf: float = 0.3,
        species_conf: float = 0.5,
        padding: float = 0.15,
        clf_size: int = 224,
        min_side: int = 32,
        min_area_frac: float = 0.003,
    ) -> None:
        self.detector = detector
        self.classifier = classifier
        self.bird_class = bird_class
        self.det_conf = det_conf
        self.species_conf = species_conf
        self.padding = padding
        self.clf_size = clf_size
        self.min_side = min_side
        self.min_area_frac = min_area_frac

    def infer(self, image: Image.Image) -> CascadeResult:
        dets = [d for d in self.detector.detect(image) if d.score >= self.det_conf]
        if not dets:
            return CascadeResult(level="none")

        birds = [d for d in dets if d.class_id == self.bird_class]
        if not birds:
            # 非鸟:直接出粗类(置信最高的)
            top = max(dets, key=lambda d: d.score)
            return CascadeResult(level="coarse", coarse_class=top.class_id, confidence=top.score)

        bird = max(birds, key=lambda d: d.score)
        wh = (image.width, image.height)
        # 最小尺寸门控:框太小 → 不报种,回退粗类 bird
        if not passes_min_size(bird.box, wh, self.min_side, self.min_area_frac):
            return CascadeResult(
                level="coarse", coarse_class=self.bird_class, confidence=bird.score
            )

        crop = crop_with_padding(image, bird.box, padding=self.padding, size=self.clf_size)
        idx, conf, top5 = self.classifier.classify(crop)
        # 置信门控:种置信不足 → 层级回退报粗类 bird(宁粗不错)
        if conf < self.species_conf:
            return CascadeResult(level="coarse", coarse_class=self.bird_class, confidence=conf)
        return CascadeResult(
            level="species",
            coarse_class=self.bird_class,
            species_idx=idx,
            confidence=conf,
            top5=top5,
        )

    def evaluate(self, samples: list[tuple[Image.Image, int]]) -> CascadeReport:
        """samples: [(image, species_gt_idx)]。检出率/级联top-1(在检出 bird 的子集上)/回退率。"""
        n = len(samples)
        hit = correct = fallback = 0
        for image, gt in samples:
            res = self.infer(image)
            is_bird = res.coarse_class == self.bird_class and res.level in ("species", "coarse")
            if not is_bird:
                continue
            hit += 1
            if res.level == "coarse":  # 回退报粗类(没报种)
                fallback += 1
            elif res.species_idx == gt:
                correct += 1
        return CascadeReport(
            n=n,
            bird_hit_rate=hit / n if n else 0.0,
            cascade_top1=correct / hit if hit else 0.0,
            fallback_rate=fallback / hit if hit else 0.0,
        )
