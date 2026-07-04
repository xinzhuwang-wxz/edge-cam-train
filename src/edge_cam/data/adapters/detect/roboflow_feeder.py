"""Roboflow Universe 喂鸟器域检测集 → 5 类（[[ADR-0004]]）。**补 feeder 真实域**——当前最大短板
（docs/detect/01 §2 列了 Roboflow 却从未落 adapter；bird 框全来自网图/相机陷阱，无"鸟停喂食器"域）。

Roboflow 导出标准 COCO（`train/valid/test` 各 `_annotations.coco.json` + 图）→ 本 adapter 合并为
单 coco 交基类解析，再按 path 重 split（各 split 的 image id 独立 → 合并时全局重编号）。

获取（ADR-0006 D2）：`_fetch()` 用 `roboflow` SDK 按 workspace/project/version 下载（需
`ROBOFLOW_API_KEY`，box 上跑）。**许可逐个核**（§4 商用须 CC-BY/Public/MIT）；label_map 为占位
——上 box 拿到真实 export 后跑 `audit_unmapped()` 据实校正（类目字符串以实际 json 为准）。
"""

from __future__ import annotations

import json
from pathlib import Path

from edge_cam.data.adapters.detect.base import AcquireSpec, DatasetSpec, register_adapter
from edge_cam.data.adapters.detect.coco_json import CocoJsonAdapter


class RoboflowFeederAdapter(CocoJsonAdapter):
    """Roboflow 喂鸟器域检测集 → 5 类（穷尽标注、逐图署名、config 驱动 project/label_map）。"""

    SUBPATH = "commercial/roboflow_feeder"

    # 占位 label_map（上 box audit_unmapped 据实 export 校正）。喂食器常见类目 → 5 类。
    DEFAULT_LABEL_MAP: dict[str, str] = {
        # bird-recognition/bird-feeder v4 实核 5 种鸟（audit 2026-07-04）→ bird（细分交分类器）
        "house_sparrow": "bird",
        "coal_tit": "bird",
        "great_tit": "bird",
        "blackbird": "bird",
        "blue_tit": "bird",
        # 通用别名（其它 feeder 项目 / 手动 override 用）
        "bird": "bird",
        "Bird": "bird",
        "squirrel": "squirrel",
        "Squirrel": "squirrel",
        "cat": "cat",
        "Cat": "cat",
        "person": "person",
        "Person": "person",
    }

    def __init__(
        self,
        raw_root: str,
        *,
        name: str = "roboflow_feeder",
        subpath: str | None = None,  # 缺省 = commercial/<name>（多 Roboflow 集各自独立盘位）
        workspace: str = "bird-recognition",
        project: str = "bird-feeder",
        version: int = 1,
        label_map: dict[str, str] | None = None,
        catch_all_label: str | None = None,  # 整集同粗类（feeder 集 N 鸟种全→bird）时设 "bird"
        license: str = "CC-BY-4.0",  # 逐个核（§4）
        negative_quota: int | None = 0,
        max_per_class: int | dict[str, int] | None = None,
        **spec_overrides,
    ) -> None:
        self._subpath = subpath or f"commercial/{name}"
        self._base = Path(raw_root) / self._subpath
        self._rf = (workspace, project, version)
        spec = DatasetSpec(
            name=name,
            raw_format="roboflow_coco",
            label_map=label_map or self.DEFAULT_LABEL_MAP,
            catch_all_label=catch_all_label,
            license=license,
            commercial_safe=True,
            role="train",
            exhaustive=True,  # Roboflow 标注穷尽 → 缺类区域当背景/负样本安全
            split_unit="image",
            attribution=True,  # CC-BY 逐图署名
            acquire=AcquireSpec(
                method="roboflow",
                urls=[f"https://universe.roboflow.com/{workspace}/{project}"],
                version=str(version),
            ),
            negative_quota=negative_quota,
            max_per_class=max_per_class,
            **spec_overrides,
        )
        super().__init__(spec, json_path=self._base, image_root=self._subpath)

    def _load_coco(self) -> dict:
        """合并 Roboflow export 下所有 `*/_annotations.coco.json` → 单 coco。

        各 split 的 image/annotation id 独立 → 全局重编号防冲突；file_name 前缀 split 目录名
        （与磁盘布局一致）；categories 取首个文件（Roboflow 各 split 类目一致）。
        """
        images: list[dict] = []
        anns: list[dict] = []
        cats: list[dict] | None = None
        next_img_id = 1
        next_ann_id = 1
        for p in sorted(self._base.glob("**/_annotations.coco.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            if cats is None:
                cats = d.get("categories")
            split_dir = p.parent.name  # train/valid/test
            id_remap: dict[int, int] = {}
            for im in d.get("images", []):
                gid = next_img_id
                next_img_id += 1
                id_remap[im["id"]] = gid
                images.append({**im, "id": gid, "file_name": f"{split_dir}/{im['file_name']}"})
            for a in d.get("annotations", []):
                if a["image_id"] not in id_remap:
                    continue
                anns.append({**a, "id": next_ann_id, "image_id": id_remap[a["image_id"]]})
                next_ann_id += 1
        return {"images": images, "annotations": anns, "categories": cats or []}

    def _fetch(self, dest: Path) -> None:
        """roboflow SDK 下载 export 到 dest（需 ROBOFLOW_API_KEY，box 上跑）。"""
        import os

        key = os.environ.get("ROBOFLOW_API_KEY")
        if not key:
            raise RuntimeError(
                "roboflow_feeder: 缺 ROBOFLOW_API_KEY 环境变量（不入库，见 ADR-0006 D7）"
            )
        from roboflow import Roboflow  # lazy import（可选 extra）

        workspace, project, version = self._rf
        rf = Roboflow(api_key=key)
        rf.workspace(workspace).project(project).version(version).download(
            "coco", location=str(dest)
        )


class RoboflowBirdV2Adapter(RoboflowFeederAdapter):
    """leem-pf8fb/bird-v2（CC-BY-4.0）：清晰鸟类摄影 36 鸟种 → bird（catch_all）。域比 iNat 近
    （清晰、尺度合适），作 feeder 域清晰鸟补充。id0 空根类 bird-types 无框、无碍。"""

    def __init__(self, raw_root: str, **kw) -> None:
        kw.setdefault("name", "roboflow_birdv2")
        kw.setdefault("workspace", "leem-pf8fb")
        kw.setdefault("project", "bird-v2")
        kw.setdefault("version", 2)
        kw.setdefault("catch_all_label", "bird")  # 36 鸟种全 → bird
        super().__init__(raw_root, **kw)


class RoboflowMeprojectAdapter(RoboflowFeederAdapter):
    """meproject-pcsly/bird-feeder（CC-BY-4.0）：**真 feeder-cam 定拍**（时间戳水印）花园鸟种
    → bird。观鸟器域金标准。id0 空根类 birds 无框。"""

    def __init__(self, raw_root: str, **kw) -> None:
        kw.setdefault("name", "roboflow_meproject")
        kw.setdefault("workspace", "meproject-pcsly")
        kw.setdefault("project", "bird-feeder-hhjks")
        kw.setdefault("version", 6)
        kw.setdefault("catch_all_label", "bird")  # blue_tit/great_tit/robin/sparrow → bird
        super().__init__(raw_root, **kw)


register_adapter("roboflow_feeder", RoboflowFeederAdapter)
register_adapter("roboflow_birdv2", RoboflowBirdV2Adapter)
register_adapter("roboflow_meproject", RoboflowMeprojectAdapter)
