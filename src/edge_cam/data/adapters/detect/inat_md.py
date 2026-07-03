"""iNaturalist Open Data → MegaDetector 伪标注 → 5 类（[[ADR-0004]]）。**补 bird 覆盖/多样性**。

三步（ADR-0006 D7）：
  ① iNat Open Data S3 拉图（免鉴权流式 TSV；收紧 **CC0/CC-BY**、research-grade、有 geo、per-taxon
     配额）—— box 上跑（`_fetch`）；iNat 图**无 bbox**。
  ② MegaDetector 伪标注出框（**独立 GPU 阶段**，隔离 env/`pytorch-wildlife`；产 COCO，框
     `label_provenance=md_pseudo`）—— 不在 acquire 内。
  ③（可选）Label Studio 人审：md_pseudo → md_human_verified（信任分层，透明可审）。
本 adapter 读第 ②/③ 步产出的 MD-COCO 进 5 类管线。iNat 观测已筛 Aves → MD 的 animal 框即 bird。

许可：**收紧 CC0/CC-BY**（§4 商用，去一切 NC，[[bird-dataset-license-landscape]]）；逐图
author/original_url 经 attribution 流兑现署名（CC-BY）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from edge_cam.data.adapters.detect.base import AcquireSpec, DatasetSpec, register_adapter
from edge_cam.data.adapters.detect.coco_json import CocoJsonAdapter

# 商用收紧：只收 CC0 / CC-BY（去一切 NC/ND/SA-NC 变体，§4）。bird-tagger 收 NC（research）此处不可。
INAT_OPEN_LICENSES: frozenset[str] = frozenset({"CC0", "CC0-1.0", "CC-BY", "CC-BY-4.0"})


@dataclass
class InatObs:
    """一条 iNat 观测（S3 open-data join 后）：图 id + taxon + 许可 + geo + 作者。"""

    photo_id: str
    taxon_id: str
    license: str
    lat: float | None = None
    lon: float | None = None
    author: str | None = None
    quality_grade: str = "research"


def select_inat(
    obs: list[InatObs],
    *,
    per_taxon_cap: int,
    licenses: frozenset[str] = INAT_OPEN_LICENSES,
    require_geo: bool = True,
    require_research: bool = True,
) -> list[InatObs]:
    """iNat 观测 → 选中集（**纯函数可测**）：license 收紧 + research-grade + geo + per-taxon 配额。

    per-taxon 配额治长尾（常见种淹没稀有种）；确定性（保序，按到达顺序取前 N）。
    """
    kept: list[InatObs] = []
    counts: dict[str, int] = {}
    for o in obs:
        if o.license not in licenses:
            continue
        if require_research and o.quality_grade != "research":
            continue
        if require_geo and (o.lat is None or o.lon is None):
            continue
        if counts.get(o.taxon_id, 0) >= per_taxon_cap:
            continue
        counts[o.taxon_id] = counts.get(o.taxon_id, 0) + 1
        kept.append(o)
    return kept


class InatMdAdapter(CocoJsonAdapter):
    """iNat（MD 伪标注）→ 5 类。读 MD-COCO（框 md_pseudo/md_human_verified）；CC0/CC-BY 可商用。"""

    SUBPATH = "commercial/inat_md"
    MD_COCO = "inat_md_coco.json"  # MD 伪标注（独立阶段）产物

    def __init__(
        self,
        raw_root: str,
        *,
        label_map: dict[str, str] | None = None,
        license: str = "CC-BY-4.0",  # 源级；逐图真实许可经 attribution 流（CC0/CC-BY）
        label_provenance: str = "md_pseudo",  # 人审通过后传 md_human_verified
        negative_quota: int | None = 0,
        max_per_class: int | dict[str, int] | None = None,
        **spec_overrides,
    ) -> None:
        base = Path(raw_root) / self.SUBPATH
        spec = DatasetSpec(
            name="inat_md",
            raw_format="inat_md_coco",
            # iNat 观测已筛 Aves → MD 的 animal/bird 框即 bird（多物种/误框靠 Label Studio 人审兜）
            label_map=label_map or {"animal": "bird", "bird": "bird"},
            license=license,
            commercial_safe=True,
            role="train",
            exhaustive=False,  # iNat 图非穷尽标注（MD 只框动物）→ 未框区域不当负样本
            split_unit="image",
            attribution=True,  # CC-BY 逐图署名
            acquire=AcquireSpec(
                method="inat_open_data",
                urls=[
                    "https://inaturalist-open-data.s3.amazonaws.com",
                    "s3://inaturalist-open-data/metadata/",  # taxa/observations/photos TSV.gz
                ],
                version="open-data",
            ),
            negative_quota=negative_quota,
            max_per_class=max_per_class,
            **spec_overrides,
        )
        super().__init__(
            spec,
            json_path=base / self.MD_COCO,
            image_root=self.SUBPATH,
            label_provenance=label_provenance,
        )

    def _fetch(self, dest: Path) -> None:
        """iNat 获取是**多步 box 流程**（非单次下载），故此处显式指引而非静默：
        ① iNat Open Data S3 流式拉图（`select_inat` 过滤 CC0/CC-BY+research+geo+per-taxon）
        → ② MegaDetector 伪标注产 `inat_md_coco.json`（隔离 env/GPU）→ ③ 可选 Label Studio 人审。
        产物就位后 `build` 直接读。"""
        raise NotImplementedError(
            f"inat_md 获取为多步 box 流程：iNat S3 拉图 → MD 伪标注产 {self.MD_COCO} → 可选人审；"
            f"select_inat 是选图过滤（可测）。产物就位后 build 直接读（dest={dest}）。"
        )


register_adapter("inat_md", InatMdAdapter)
