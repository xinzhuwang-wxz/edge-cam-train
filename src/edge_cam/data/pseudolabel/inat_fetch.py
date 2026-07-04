"""① iNat Open Data 拉图（API 枚举路线）。

**为何走 API 而非 S3 全量 dump**：只需数千张 bird（有界拉取），iNat 全量元数据 tar.gz（观测/照片
CSV 共数十 GB）解压再 join 代价过高；`/v1/observations` API 直接按 `taxon_id=3`(Aves) +
`quality_grade=research` + `photo_license=cc0,cc-by` + `geo=true` 分页枚举，逐照片带 license/作者，
正好喂 `select_inat`（复用其许可/research/geo/per-taxon 过滤），再并行下 open-data S3 的 medium 图。

纯函数 `parse_inat_api_page` / `medium_url` 可测；`fetch_inat_aves_obs` / `download_inat_photos`
是薄网络步骤（box 上跑）。所有照片经 open-data S3（CC0/CC-BY 可商用，逐图署名兑现 §4）。
"""

from __future__ import annotations

import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from edge_cam.data.adapters.detect.inat_md import InatObs

_API = "https://api.inaturalist.org/v1/observations"
AVES_TAXON_ID = 3  # iNat taxonomy: Aves（鸟纲）


def _parse_location(loc: str | None) -> tuple[float | None, float | None]:
    """iNat 观测 location "lat,lon" 串 → (lat, lon)；缺/坏 → (None, None)。"""
    if not loc or "," not in loc:
        return None, None
    try:
        a, b = loc.split(",", 1)
        return float(a), float(b)
    except ValueError:
        return None, None


def medium_url(square_or_any_url: str) -> str:
    """iNat 照片直链 → medium 档链接（API 默认给 square.jpg，换 medium 拿够训练的分辨率）。"""
    for size in ("square", "small", "thumb", "large", "original"):
        if f"/{size}." in square_or_any_url:
            return square_or_any_url.replace(f"/{size}.", "/medium.", 1)
    return square_or_any_url


def parse_inat_api_page(page: dict) -> list[InatObs]:
    """iNat `/v1/observations` 一页 JSON → InatObs 列表（**纯函数可测**）。

    每观测取第一张照片；`license` 用**照片**的 license_code（照片许可，非观测许可，规避 NC 传染）。
    license_code 小写（cc0/cc-by/cc-by-nc…）→ 大写归一，正好对上 `INAT_OPEN_LICENSES` 白名单
    （cc0→CC0、cc-by→CC-BY；NC 变体大写后仍被 `select_inat` 拒）。
    """
    out: list[InatObs] = []
    for res in page.get("results", []):
        photos = res.get("photos") or []
        if not photos:
            continue
        photo = photos[0]
        pid = photo.get("id")
        if pid is None:
            continue
        lic = (photo.get("license_code") or "").upper()  # None=保留权利 → "" → 被白名单拒
        lat, lon = _parse_location(res.get("location"))
        taxon = res.get("taxon") or {}
        user = res.get("user") or {}
        url = photo.get("url")
        out.append(
            InatObs(
                photo_id=str(pid),
                taxon_id=str(taxon.get("id", "")),
                license=lic,
                lat=lat,
                lon=lon,
                author=user.get("login"),
                quality_grade=res.get("quality_grade", ""),
                photo_url=medium_url(url) if url else None,
            )
        )
    return out


def fetch_inat_aves_obs(
    *,
    taxon_id: int = AVES_TAXON_ID,
    per_page: int = 200,
    max_obs: int = 6000,
    sleep: float = 1.0,
    opener: object | None = None,
) -> list[InatObs]:
    """iNat API 分页枚举 Aves research-grade CC0/CC-BY 有 geo 观测（薄网络步骤，box 上跑）。

    用 `id_above` 游标翻页（绕开 `page*per_page ≤ 10000` 窗口上限）；服务端已按 photo_license
    预过滤，返回后仍交 `select_inat` 复核 + per-taxon 配额。`sleep` 遵守 iNat 礼貌限速（≤1 req/s）。
    """
    import json as _json

    fetch = opener or urllib.request.urlopen
    collected: list[InatObs] = []
    id_above = 0
    while len(collected) < max_obs:
        q = (
            f"{_API}?taxon_id={taxon_id}&quality_grade=research"
            f"&photo_license=cc0,cc-by&geo=true&photos=true"
            f"&order_by=id&order=asc&per_page={per_page}&id_above={id_above}"
        )
        with fetch(q) as resp:  # type: ignore[operator]
            page = _json.loads(resp.read())
        results = page.get("results", [])
        if not results:
            break
        collected.extend(parse_inat_api_page(page))
        id_above = max(int(r["id"]) for r in results)
        if sleep:
            time.sleep(sleep)
    return collected[:max_obs]


def _download_one(args: tuple[str, str, Path]) -> str | None:
    photo_id, url, out_dir = args
    dst = out_dir / f"{photo_id}.jpg"
    if dst.exists() and dst.stat().st_size > 0:
        return photo_id
    try:
        urllib.request.urlretrieve(url, dst)  # 走 env http_proxy（box 上 source network_turbo）
        return photo_id
    except Exception:  # noqa: BLE001 — 个别图失败不致命，跳过
        dst.unlink(missing_ok=True)
        return None


def download_inat_photos(obs: list[InatObs], out_dir: Path, *, jobs: int = 16) -> list[str]:
    """并行下 medium 图到 out_dir/{photo_id}.jpg；返回成功 photo_id 列表（薄网络步骤）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(o.photo_id, o.photo_url, out_dir) for o in obs if o.photo_url]
    ok: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for res in ex.map(_download_one, tasks):
            if res:
                ok.append(res)
    return ok
