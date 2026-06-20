"""Open Images V7（官方直下版）→ 5 类（[[ADR-0004]]）。**绕开 fiftyone**（py3.8 装不上，见
docs/detect/03 §4）：`scripts/fetch_oiv7_direct.py` 过滤 bbox CSV + 加速并行 S3 下图 → 生成
`oiv7_coco.json`（categories=OIV7 display name），本 adapter 用 CocoJsonAdapter 吃它。

与 fiftyone 版（`fiftyone_oiv7.py`，注册名 `open_images_v7_fiftyone`，留待 py3.10+）等价语义：
非穷尽（按类拉，未标类不当负样本）、CC-BY 可商用 + 逐图署名。**直下版注册为默认 `open_images_v7`**。
"""

from __future__ import annotations

from pathlib import Path

from edge_cam.data.adapters.detect.base import DatasetSpec, register_adapter
from edge_cam.data.adapters.detect.coco_json import CocoJsonAdapter
from edge_cam.data.adapters.detect.fiftyone_oiv7 import OIV7_LABEL_MAP


class Oiv7DirectAdapter(CocoJsonAdapter):
    """Open Images V7 直下 COCO → 5 类（非穷尽、CC-BY 可商用 + 署名）。"""

    SUBPATH = "commercial/open_images_v7"
    JSON_NAME = "oiv7_coco.json"

    def __init__(
        self,
        raw_root: str,
        *,
        negative_quota: int | None = 0,
        max_per_class: int | None = None,
        **spec_overrides,
    ) -> None:
        base = Path(raw_root) / self.SUBPATH
        spec = DatasetSpec(
            name="open_images_v7",
            raw_format="oiv7_direct_coco",
            label_map=OIV7_LABEL_MAP,
            license="CC-BY-4.0",
            commercial_safe=True,
            role="train",
            exhaustive=False,  # 按类拉 → 未标类区域不当负样本
            split_unit="image",
            attribution=True,  # 逐图署名清册
            negative_quota=negative_quota,
            max_per_class=max_per_class,
            **spec_overrides,
        )
        # oiv7_coco.json 的 file_name 已含 images/<split>/ 前缀 → image_root=SUBPATH
        super().__init__(spec, json_path=base / self.JSON_NAME, image_root=self.SUBPATH)


register_adapter("open_images_v7", Oiv7DirectAdapter)
