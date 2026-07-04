"""① iNat Open Data 拉图（API 枚举路线）。

**为何走 API 而非 S3 全量 dump**：只需数千张 bird（有界拉取），iNat 全量元数据 tar.gz（观测/照片
CSV 共数十 GB）解压再 join 代价过高；`/v1/observations` API 直接按 `taxon_id=3`(Aves) +
`quality_grade=research` + `photo_license=cc0,cc-by` + `geo=true` 分页枚举，逐照片带 license/作者，
正好喂 `select_inat`（复用其许可/research/geo/per-taxon 过滤），再并行下 open-data S3 的 medium 图。

纯函数 `parse_inat_api_page` / `medium_url` 可测；`fetch_inat_aves_obs` / `download_inat_photos`
是薄网络步骤（box 上跑）。所有照片经 open-data S3（CC0/CC-BY 可商用，逐图署名兑现 §4）。

**HTTP 走 curl（非 urllib）**：GFW 下 Python urllib 常在 IPv6/DNS 上挂起且 `timeout` 不生效；
curl `--ipv4 --max-time` 稳且可控。`_http_get_json`/`_http_download` 是可注入的薄封装，
纯逻辑（分页游标/防空转）经注入 fake fetcher 可测。
"""

from __future__ import annotations

import json as _json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from edge_cam.data.adapters.detect.inat_md import InatObs

_API = "https://api.inaturalist.org/v1/observations"
AVES_TAXON_ID = 3  # iNat taxonomy: Aves（鸟纲）


def _http_get_json(url: str, *, timeout: int = 30, retries: int = 3) -> dict:
    """curl 取 JSON（--ipv4 --max-time；重试兜网络抖动）。GFW 下比 urllib 稳、超时可控。"""
    last = ""
    for attempt in range(retries):
        try:
            out = subprocess.run(
                ["curl", "-sS", "--ipv4", "--max-time", str(timeout), url],
                capture_output=True,
                timeout=timeout + 10,
            )
            if out.returncode == 0 and out.stdout:
                return _json.loads(out.stdout)
            last = out.stderr.decode(errors="replace")[:200]
        except (subprocess.TimeoutExpired, ValueError) as e:  # ValueError=json 解析失败
            last = f"{type(e).__name__}"
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"curl GET 失败（{retries} 次）：{url[:80]} … {last}")


def _http_download(url: str, dst: Path, *, timeout: int = 30) -> bool:
    """curl 下单文件（--ipv4 --max-time -o）；成功且非空 → True。"""
    try:
        out = subprocess.run(
            ["curl", "-sS", "--ipv4", "--max-time", str(timeout), "-o", str(dst), url],
            capture_output=True,
            timeout=timeout + 10,
        )
    except subprocess.TimeoutExpired:
        dst.unlink(missing_ok=True)
        return False
    if out.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
        return True
    dst.unlink(missing_ok=True)
    return False


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


def _obs_query(taxon_id: int, per_page: int, id_above: int) -> str:
    return (
        f"{_API}?taxon_id={taxon_id}&quality_grade=research"
        f"&photo_license=cc0,cc-by&geo=true&photos=true"
        f"&order_by=id&order=asc&per_page={per_page}&id_above={id_above}"
    )


def fetch_inat_aves_obs(
    *,
    taxon_id: int = AVES_TAXON_ID,
    per_page: int = 200,
    max_obs: int = 6000,
    sleep: float = 1.0,
    fetch_json=None,
) -> list[InatObs]:
    """iNat API 分页枚举 Aves research-grade CC0/CC-BY 有 geo 观测（薄网络步骤，box 上跑）。

    用 `id_above` 游标翻页（绕开 `page*per_page ≤ 10000` 窗口上限）；服务端已按 photo_license
    预过滤，返回后仍交 `select_inat` 复核 + per-taxon 配额。`sleep` 遵守 iNat 礼貌限速（≤1 req/s）。
    `fetch_json`（url→dict）默认 curl；可注入 fake 测分页游标/防空转逻辑。

    **防空转**：若 results 非空但游标 `id_above` 不再前进（异常分页）→ break，避免翻遍 4M 观测。
    """
    fetch = fetch_json or _http_get_json
    collected: list[InatObs] = []
    id_above = 0
    while len(collected) < max_obs:
        page = fetch(_obs_query(taxon_id, per_page, id_above))
        results = page.get("results", [])
        if not results:
            break
        collected.extend(parse_inat_api_page(page))
        nxt = max(int(r["id"]) for r in results)
        if nxt <= id_above:  # 游标未前进 → 防死循环
            break
        id_above = nxt
        if sleep:
            time.sleep(sleep)
    return collected[:max_obs]


def _download_one(args: tuple[str, str, Path]) -> str | None:
    photo_id, url, out_dir = args
    dst = out_dir / f"{photo_id}.jpg"
    if dst.exists() and dst.stat().st_size > 0:
        return photo_id
    return photo_id if _http_download(url, dst) else None


def download_inat_photos(obs: list[InatObs], out_dir: Path, *, jobs: int = 16) -> list[str]:
    """并行下 medium 图到 out_dir/{photo_id}.jpg（curl --ipv4）；返回成功 photo_id 列表。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(o.photo_id, o.photo_url, out_dir) for o in obs if o.photo_url]
    ok: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for res in ex.map(_download_one, tasks):
            if res:
                ok.append(res)
    return ok
