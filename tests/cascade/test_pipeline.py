"""级联编排（[[ADR-0003]] C2）：置信门控 + 层级回退 + 非鸟出粗类 + evaluate（fake,快）。"""

from __future__ import annotations

from PIL import Image

from edge_cam.cascade import CascadePipeline, Detection


class FakeDetector:
    def __init__(self, dets: list[Detection]) -> None:
        self._dets = dets

    def detect(self, image):  # noqa: ANN001
        return self._dets


class FakeClassifier:
    def __init__(self, idx: int, conf: float) -> None:
        self._idx, self._conf = idx, conf

    def classify(self, crop):  # noqa: ANN001
        return self._idx, self._conf, [self._idx, 1, 2, 3, 4]


def _img():
    return Image.new("RGB", (640, 480))


BIG_BIRD = Detection(box=(100, 100, 400, 400), class_id=0, score=0.9)  # 够大


def test_bird_high_conf_reports_species() -> None:
    p = CascadePipeline(FakeDetector([BIG_BIRD]), FakeClassifier(42, 0.8), species_conf=0.5)
    r = p.infer(_img())
    assert r.level == "species" and r.species_idx == 42


def test_bird_low_conf_falls_back_to_coarse() -> None:
    p = CascadePipeline(FakeDetector([BIG_BIRD]), FakeClassifier(42, 0.3), species_conf=0.5)
    r = p.infer(_img())
    assert r.level == "coarse" and r.coarse_class == 0 and r.species_idx is None


def test_small_bird_box_gated_to_coarse() -> None:
    tiny = Detection(box=(10, 10, 20, 20), class_id=0, score=0.9)  # 短边10<32 门控
    p = CascadePipeline(FakeDetector([tiny]), FakeClassifier(42, 0.99))
    r = p.infer(_img())
    assert r.level == "coarse" and r.coarse_class == 0


def test_non_bird_reports_coarse_directly() -> None:
    cat = Detection(box=(50, 50, 300, 300), class_id=2, score=0.8)
    p = CascadePipeline(FakeDetector([cat]), FakeClassifier(42, 0.99))
    r = p.infer(_img())
    assert r.level == "coarse" and r.coarse_class == 2


def test_no_detection_returns_none() -> None:
    p = CascadePipeline(FakeDetector([]), FakeClassifier(42, 0.99))
    assert p.infer(_img()).level == "none"


def test_det_conf_threshold_filters() -> None:
    low = Detection(box=(100, 100, 400, 400), class_id=0, score=0.1)
    p = CascadePipeline(FakeDetector([low]), FakeClassifier(42, 0.99), det_conf=0.3)
    assert p.infer(_img()).level == "none"


def test_evaluate_metrics() -> None:
    # 3 张鸟图:1 对(species=7)、1 错(species=8 vs gt7)、1 回退(低置信)
    samples = [
        (_img(), 7),
        (_img(), 7),
        (_img(), 7),
    ]
    # 都检出 bird;分类器固定返回 idx=7 conf=0.8 → 全对 → top1=1.0, 回退0
    p = CascadePipeline(FakeDetector([BIG_BIRD]), FakeClassifier(7, 0.8), species_conf=0.5)
    rep = p.evaluate(samples)
    assert rep.bird_hit_rate == 1.0
    assert rep.cascade_top1 == 1.0
    assert rep.fallback_rate == 0.0
