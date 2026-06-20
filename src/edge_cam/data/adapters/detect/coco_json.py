"""COCO/COCO-Camera-Traps 格式通用 adapter（ENA24/Caltech-CT/NACTI/COCO2017/Roboflow 共用）。

这些数据集的标注都是一份 `instances` JSON（images/annotations/categories）。本 adapter 把
**解析**收口（建 image_id→annotations、源类目名作 box 标签、0 标注图当负样本、相机陷阱按
`group_key_field`(location/seq_id) 取分组键防泄漏），具体数据集只声明 `DatasetSpec` + json 路径
（深基类 + 薄 adapter，[[ADR-0003]]）。映射→5类/限额/split/provenance 全在基类 `build_records`。

`audit_unmapped()`：列出 json 里**未被 label_map 覆盖**的类目 —— 上 AutoDL 拿到真实数据后先跑它，
据此校正 label_map（各数据集的类目字符串以实际 json 为准，见 docs/detect/01-数据集.md §4/§6）。
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from edge_cam.data.adapters.detect.base import DatasetSpec, DetectionDatasetAdapter, RawSample


class CocoJsonAdapter(DetectionDatasetAdapter):
    """读一份 COCO instances JSON → RawSample（源类目名未映射）。

    Args:
        spec: 数据集声明（label_map 用**源类目名**作 key）。
        json_path: instances JSON 路径。
        image_root: 图片根（写入 RawSample.path 时相对它；缺省用 file_name 原样）。
        group_key_field: 相机陷阱分组键在 image 记录里的字段名（如 "location"/"seq_id"）；
            None → 按 path 分 split（网图源）。
        path_field: image 记录里取相对路径的字段（默认 "file_name"）。
    """

    def __init__(
        self,
        spec: DatasetSpec,
        json_path: str | Path,
        image_root: str | Path | None = None,
        *,
        group_key_field: str | None = None,
        path_field: str = "file_name",
    ) -> None:
        super().__init__(spec)
        self.json_path = Path(json_path)
        self.image_root = Path(image_root) if image_root is not None else None
        self.group_key_field = group_key_field
        self.path_field = path_field

    def _load_coco(self) -> dict:
        return json.loads(self.json_path.read_text(encoding="utf-8"))

    def load_raw(self) -> Iterable[RawSample]:
        coco = self._load_coco()
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        anns_by_img: dict[int, list[dict]] = defaultdict(list)
        for a in coco.get("annotations", []):
            anns_by_img[a["image_id"]].append(a)
        for img in coco["images"]:
            boxes: list[tuple[str, list[float]]] = []
            for a in anns_by_img.get(img["id"], []):
                name = cats.get(a["category_id"])
                bbox = a.get("bbox")
                if name is None or bbox is None:
                    continue  # 未映射类目 或 无框注解（如相机陷阱 empty 帧）→ 不计框
                boxes.append((name, [float(v) for v in bbox]))
            gk = None
            if self.group_key_field is not None:
                raw_gk = img.get(self.group_key_field)
                gk = str(raw_gk) if raw_gk is not None else None
            file_name = img[self.path_field]
            path = str(Path(self.image_root) / file_name) if self.image_root else file_name
            yield RawSample(
                path=path,
                width=int(img.get("width", 0)),
                height=int(img.get("height", 0)),
                boxes=boxes,
                group_key=gk,
                is_negative=len(boxes) == 0,  # 0 标注 = 真空图（穷尽源/显式负样本才在基类保留）
            )

    def audit_unmapped(self) -> dict[str, int]:
        """{未映射源类目名: 标注框数}（含该类目却不在 label_map → 上线前据此校正映射）。"""
        coco = self._load_coco()
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        counts: dict[str, int] = defaultdict(int)
        for a in coco.get("annotations", []):
            name = cats.get(a["category_id"])
            if name is not None and name not in self.spec.label_map:
                counts[name] += 1
        return dict(counts)
