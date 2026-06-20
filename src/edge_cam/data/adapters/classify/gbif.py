"""GBIF 鸟图 adapter（iNat / naturgucker / 挪 / 丹 等 GBIF 数据源通用，[[ADR-0005]]）。

`scripts/fetch_gbif_birds.py` 按 datasetKey + Aves + CC0/CC-BY 查 GBIF、并行下图、产 `index.csv`
（path/scientific_name/license/group_key/lat/lon/observed_at）。本 adapter **源无关**读这份 index →
ClassifyRawSample。各源下载差异（iNat S3 medium、naturgucker Flickr）在 fetch 处理，adapter 不管。

taxonomy：默认 `IdentityTaxonomy`（学名归一当 key）；给 `taxonomy_csv`（学名→eBird code）
则用 `EbirdTaxonomy`（真规范键，[[ADR-0002]]）。文档：docs/classify/01 §3b/§5。
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from edge_cam.data.adapters.classify.base import (
    ClassifyDatasetAdapter,
    ClassifyRawSample,
    ClassifySpec,
    register_adapter,
)
from edge_cam.data.taxonomy import EbirdTaxonomy, IdentityTaxonomy


class GbifBirdsAdapter(ClassifyDatasetAdapter):
    """读 fetch_gbif_birds 产的 index.csv → 分类样本（源无关）。"""

    def __init__(
        self,
        raw_root: str,
        *,
        index: str = "index.csv",
        source: str = "gbif",
        taxonomy_csv: str | None = None,
        max_per_class: int | None = None,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        **spec_overrides,
    ) -> None:
        tax = EbirdTaxonomy.from_csv(taxonomy_csv) if taxonomy_csv else IdentityTaxonomy()
        spec = ClassifySpec(
            name=source,
            source=source,
            raw_format="gbif_index",
            taxonomy=tax,
            split_unit="observer",  # group_key=拍摄者/观测，防泄漏
            max_per_class=max_per_class,
            split_ratios=split_ratios,
            **spec_overrides,
        )
        super().__init__(spec)
        self.index_path = Path(raw_root) / index

    def load_raw(self) -> Iterable[ClassifyRawSample]:
        with self.index_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                yield ClassifyRawSample(
                    path=row["path"],
                    raw_label=row["scientific_name"],
                    license=row["license"],
                    group_key=row.get("group_key") or None,
                    lat=float(row["lat"]) if row.get("lat") else None,
                    lon=float(row["lon"]) if row.get("lon") else None,
                    observed_at=row.get("observed_at") or None,
                )


register_adapter("gbif_birds", GbifBirdsAdapter)
