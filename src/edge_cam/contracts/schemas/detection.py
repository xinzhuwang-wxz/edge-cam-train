"""检测标注 pydantic 契约（LLM 批量打标的合法标签靶子 + 硬闸）。

分类侧已有 DatasetManifest 闭集校验；检测侧此前走裸 COCO（无校验）→ LLM 打标会自创类名/
bbox 乱填。本契约：① category 锁进 **5 类粗检测闭集**（[[ADR-0004]]，`FEEDER5_CATEGORIES`）；
② bbox 合法性校验；③ 与标准 COCO labels.json 互转，接现有 NanoDet 流水。

LLM 产出 → DetImageLabels 校验（越界/幻觉当场 raise）→ to_coco → 入库训练。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from edge_cam.contracts.schemas.detection_manifest import FEEDER5_CATEGORIES

# 5 类闭集 = 唯一合法 category（与 FEEDER5_CATEGORIES 单一事实源对齐，防 LLM 自创类名）
CoarseLabel = Literal["bird", "squirrel", "cat", "person", "other_animal"]

_ALLOWED = set(FEEDER5_CATEGORIES)


def _assert_label_set_in_sync() -> None:
    """启动期自检：Literal 与 FEEDER5_CATEGORIES 不漂移（漏一个就炸，强制同步）。"""
    literal = set(CoarseLabel.__args__)  # type: ignore[attr-defined]
    if literal != _ALLOWED:
        raise RuntimeError(f"CoarseLabel 与 FEEDER5_CATEGORIES 漂移：{literal ^ _ALLOWED}")


_assert_label_set_in_sync()


class BBox(BaseModel):
    """像素框 [x, y, w, h]（COCO 口径，左上角 + 宽高）。校验非负、正尺寸、不超图。"""

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    w: float = Field(gt=0)
    h: float = Field(gt=0)

    def fits_in(self, width: int, height: int) -> bool:
        return self.x + self.w <= width and self.y + self.h <= height


class DetAnnotation(BaseModel):
    """单个检测框：合法 category（11 类闭集）+ bbox + 可选置信度。"""

    label: CoarseLabel
    bbox: BBox
    confidence: float | None = Field(default=None, ge=0, le=1)


class DetImageLabels(BaseModel):
    """一张图的检测打标产出（LLM/人工产这个，过闸后转 COCO）。"""

    file_name: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    annotations: list[DetAnnotation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _bbox_in_image(self) -> DetImageLabels:
        for a in self.annotations:
            if not a.bbox.fits_in(self.width, self.height):
                raise ValueError(
                    f"{self.file_name}: bbox {a.bbox} 超出图像 {self.width}x{self.height}"
                )
        return self


def validate_llm_labels(raw: dict | list[dict]) -> list[DetImageLabels]:
    """LLM 产出（dict/list）过闸 → DetImageLabels 列表；越界类/非法 bbox 当场 raise。"""
    items = raw if isinstance(raw, list) else [raw]
    return [DetImageLabels.model_validate(it) for it in items]


def to_coco(labels: list[DetImageLabels], *, classes: list[str] | None = None) -> dict:
    """DetImageLabels 列表 → 标准 COCO labels.json（接 NanoDet/FiftyOne 流水）。"""
    names = classes or list(FEEDER5_CATEGORIES)
    cat_id = {name: i + 1 for i, name in enumerate(names)}  # COCO category_id 从 1
    categories = [{"id": i + 1, "name": n, "supercategory": "animal"} for i, n in enumerate(names)]
    images, annotations = [], []
    ann_id = 1
    for img_id, img in enumerate(labels, start=1):
        images.append(
            {"id": img_id, "file_name": img.file_name, "width": img.width, "height": img.height}
        )
        for a in img.annotations:
            b = a.bbox
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id[a.label],
                    "bbox": [b.x, b.y, b.w, b.h],
                    "area": b.w * b.h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return {"categories": categories, "images": images, "annotations": annotations}


def from_coco(coco: dict) -> list[DetImageLabels]:
    """标准 COCO labels.json → DetImageLabels 列表（同时校验 category 在闭集内）。"""
    id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    by_image: dict[int, DetImageLabels] = {}
    for img in coco.get("images", []):
        by_image[img["id"]] = DetImageLabels(
            file_name=img["file_name"], width=img["width"], height=img["height"]
        )
    for a in coco.get("annotations", []):
        x, y, w, h = a["bbox"]
        name = id_to_name[a["category_id"]]
        by_image[a["image_id"]].annotations.append(
            DetAnnotation(label=name, bbox=BBox(x=x, y=y, w=w, h=h))  # type: ignore[arg-type]
        )
    return list(by_image.values())
