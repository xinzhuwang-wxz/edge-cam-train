"""ENA24-detection（LILA）→ 5 类（[[ADR-0004]]）。东北美 ~10k 图**全带框**，23 类，单帧。

COCO-Camera-Traps JSON（`ena24_public.json`：images/annotations/categories）。**全带框 →
exhaustive=True**（缺类区域当背景安全）。image 记录只有 id/file_name/width/height，**无
location/seq → 按 image 划分**（无防泄漏分组键，单帧无连拍）。许可 CDLA-Permissive → 可商用。

下载（AutoDL，annotation-first 见 docs/detect/01 §9；任选 GCP/Azure/AWS 源）：
  base=https://storage.googleapis.com/public-datasets-lila/ena24
  - 标注: {base}/ena24_public.json
  - 图片: {base}/ena24.zip（3.6GB，解压到 image_root）

label_map 据 agent 实测 `ena24_public.json` categories（23 类，id 8 缺）。上线前可跑
`audit_unmapped()` 复核（数据集类目字符串以实际 json 为准）。
"""

from __future__ import annotations

from pathlib import Path

from edge_cam.data.adapters.detect.base import AcquireSpec, DatasetSpec, register_adapter
from edge_cam.data.adapters.detect.coco_json import CocoJsonAdapter

_LILA = "https://storage.googleapis.com/public-datasets-lila/ena24"

# ENA24 原类目（精确字符串）→ 5 类。Vehicle/Horse 不映射（喂食器见不到 → 丢）。
# Dog/各野生哺乳 → other_animal（ADR-0004：非 bird/squirrel/cat 哺乳动物归并）。
ENA24_LABEL_MAP: dict[str, str] = {
    "Bird": "bird",
    "Wild Turkey": "bird",
    "American Crow": "bird",
    "Chicken": "bird",
    "Eastern Gray Squirrel": "squirrel",
    "Eastern Fox Squirrel": "squirrel",
    "Domestic Cat": "cat",
    "Dog": "other_animal",
    "Eastern Chipmunk": "other_animal",
    "Woodchuck": "other_animal",
    "White_Tailed_Deer": "other_animal",  # 字面下划线（实测 json 原样）
    "Virginia Opossum": "other_animal",
    "Eastern Cottontail": "other_animal",
    "Striped Skunk": "other_animal",
    "Red Fox": "other_animal",
    "Northern Raccoon": "other_animal",
    "Grey Fox": "other_animal",
    "Coyote": "other_animal",
    "Bobcat": "other_animal",
    "American Black Bear": "other_animal",
    # 不映射（丢）：Vehicle, Horse
}


class Ena24Adapter(CocoJsonAdapter):
    """ENA24 → 5 类（全带框、单帧、按 image 划分、CDLA 可商用）。"""

    SUBPATH = "commercial/ena24"
    JSON_NAME = "ena24_public.json"
    IMAGE_SUBDIR = ""  # file_name 已是相对图片路径；image_root=SUBPATH

    def __init__(
        self,
        raw_root: str,
        *,
        negative_quota: int | None = 0,  # 全带框、~无空图 → 默认不取负样本
        max_per_class: int | None = None,
        **spec_overrides,
    ) -> None:
        base = Path(raw_root) / self.SUBPATH
        image_root = f"{self.SUBPATH}/{self.IMAGE_SUBDIR}".rstrip("/")
        spec = DatasetSpec(
            name="ena24",
            raw_format="coco_camera_traps",
            label_map=ENA24_LABEL_MAP,
            license="CDLA-Permissive",
            commercial_safe=True,
            role="train",
            exhaustive=True,
            split_unit="image",
            acquire=AcquireSpec(  # LILA（暂无自动下载器 → manual：acquire 校验就位 + 给下载 URL）
                method="manual",
                urls=[f"{_LILA}/ena24_public.json", f"{_LILA}/ena24.zip"],
                version="lila",
            ),
            negative_quota=negative_quota,
            max_per_class=max_per_class,
            **spec_overrides,
        )
        super().__init__(spec, json_path=base / self.JSON_NAME, image_root=image_root)


register_adapter("ena24", Ena24Adapter)
