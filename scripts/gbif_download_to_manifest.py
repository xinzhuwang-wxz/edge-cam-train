"""GBIF Download API → 欧洲下载清单（取代分页采集，一个 job 拿全，抗限流/flaky 网络）。

配 `build_europe_download_manifest.py`（分页版，撞 GBIF 限流+网络）。本脚本走 **GBIF 异步
Download API**：提交的谓词（Aves + StillImage + CC0/CC-BY + 活体观测 + 欧洲各国 + 扣 iNat）
在 GBIF 后台打包成 DWCA 存档 → 本脚本轮询、下载、解析成同格式 `europe_download_manifest.csv`。

DWCA 解析：occurrence.txt（gbifID→speciesKey/license/经纬/日期/recordedBy/datasetKey）
+ multimedia.txt（gbifID→identifier=URL）join；speciesKey→ebird_code 用 europe_image_coverage
的 gbif_key 映射（只留我们 1211 欧洲类集的种；GBIF 额外种丢）。

凭据 ~/.config/gbif/creds.json（user/pwd）。用法（worktree）：
  python scripts/gbif_download_to_manifest.py            # 轮询；SUCCEEDED 则下载+解析
  python scripts/gbif_download_to_manifest.py --key <k>  # 指定 download key
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

EUROPE = Path(__file__).resolve().parents[1] / "data" / "region" / "europe"
KEY_FILE = Path.home() / ".config" / "gbif" / "last_download_key.txt"
CREDS = Path.home() / ".config" / "gbif" / "creds.json"
DL_DIR = EUROPE / "gbif_dl"
LOCK = DL_DIR / ".lock"  # pidfile 锁：防监控 cron 反复触发起多个下载抢同一文件
OUT_FIELDS = [
    "ebird_code",
    "scientific_name",
    "url",
    "license",
    "dataset_key",
    "group_key",
    "lat",
    "lon",
    "observed_at",
]
_LICENSE = {"publicdomain/zero": "CC0_1_0", "licenses/by/": "CC_BY_4_0"}


def _norm_license(u: str) -> str:
    for frag, code in _LICENSE.items():
        if frag in (u or ""):
            return code
    return u or ""


def _media_license(s: str) -> str:
    """媒体级 license 严格归一（③红线：occurrence license≠图片 license，须按图片级筛）。
    返回 CC0_1_0 / CC_BY_4_0 / other（©版权保留/NC/SA/ND/空/未知 都归 other 剔除）。"""
    s = (s or "").lower()
    if "by-nc" in s or "by-sa" in s or "-nd" in s or "noncommercial" in s or "sharealike" in s:
        return "other"  # NC/SA/ND 红线，先判（含 'by' 不能误放行）
    if "publicdomain/zero" in s or "cc0" in s:
        return "CC0_1_0"
    if "/licenses/by/" in s or "licenses/by " in s:
        return "CC_BY_4_0"
    return "other"  # ©版权保留/见URL/空/未知 → 剔（不确定即不用）


def _already_parsed() -> bool:
    """清单已是 GBIF 全量 → 幂等跳过（防 cron 反复重下 825MB）。"""
    s = EUROPE / "europe_manifest_summary.json"
    if not s.exists():
        return False
    try:
        return "Download" in json.loads(s.read_text()).get("source", "")
    except (ValueError, OSError):
        return False


def _running() -> bool:
    """已有下载/解析实例在跑（pidfile 锁，检查 PID 存活）→ 跳过，防抢同一文件。"""
    if not LOCK.exists():
        return False
    try:
        os.kill(int(LOCK.read_text()), 0)
        return True
    except (ProcessLookupError, ValueError, OSError):
        return False  # 死锁/无效 → 视为未锁


def status(key: str) -> dict:
    return json.load(
        urllib.request.urlopen(f"https://api.gbif.org/v1/occurrence/download/{key}", timeout=30)
    )


def download_zip(key: str, expected: int = 0) -> Path:
    """断点续传下载 DWCA 存档（HTTP Range），抗 flaky 网络的 IncompleteRead/超时。"""
    DL_DIR.mkdir(parents=True, exist_ok=True)
    dst = DL_DIR / f"{key}.zip"
    c = json.loads(CREDS.read_text())
    url = f"https://api.gbif.org/v1/occurrence/download/request/{key}.zip"
    auth = "Basic " + base64.b64encode(f"{c['user']}:{c['pwd']}".encode()).decode()
    print(f"下载存档 → {dst}（期望 {expected / 1e6:.0f}MB，断点续传）…", flush=True)
    for _ in range(60):
        pos = dst.stat().st_size if dst.exists() else 0
        if expected and pos >= expected:
            break  # 完整
        req = urllib.request.Request(url)
        req.add_header("Authorization", auth)
        if pos:
            req.add_header("Range", f"bytes={pos}-")  # 从断点续
        try:
            with (
                urllib.request.urlopen(req, timeout=180) as r,
                dst.open("ab" if pos else "wb") as f,
            ):
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if not expected:
                break  # 无期望大小则一次成功即完
        except Exception as e:  # noqa: BLE001 — IncompleteRead/超时 → 续传重试
            print(
                f"  中断 @{dst.stat().st_size / 1e6:.1f}MB，续传…（{type(e).__name__}）", flush=True
            )
            time.sleep(3)
    print(f"存档完成 {dst.stat().st_size / 1e6:.0f} MB", flush=True)
    return dst


def _cols(header: str) -> dict[str, int]:
    """DwC 列名（可能是完整 URI，取末段）→ 列号。"""
    out = {}
    for i, h in enumerate(header.rstrip("\n").split("\t")):
        out[h.rsplit("/", 1)[-1]] = i
    return out


def _field(p: list[str], col: dict[str, int], name: str) -> str:
    i = col.get(name)
    return p[i] if i is not None and i < len(p) else ""


def parse_to_manifest(zip_path: Path) -> None:
    # gbif_key → ebird_code（只留 1211 欧洲类集）
    key2code: dict[str, str] = {}
    code2sci: dict[str, str] = {}
    for line in (EUROPE / "europe_image_coverage.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("gbif_key"):
            key2code[str(d["gbif_key"])] = d["ebird_code"]
            code2sci[d["ebird_code"]] = d["sci_name"]

    with zipfile.ZipFile(zip_path) as z:
        # occurrence: gbifID → 元数据
        occ: dict[str, dict] = {}
        with z.open("occurrence.txt") as fh:
            tf = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
            col = _cols(tf.readline())
            for line in tf:
                p = line.rstrip("\n").split("\t")
                sk = _field(p, col, "speciesKey") or _field(p, col, "taxonKey")
                code = key2code.get(sk)
                if not code:
                    continue  # 不在 1211 欧洲类集 → 丢
                occ[_field(p, col, "gbifID")] = {
                    "ebird_code": code,
                    "scientific_name": code2sci[code],
                    "license": _norm_license(_field(p, col, "license")),
                    "dataset_key": _field(p, col, "datasetKey"),
                    "group_key": _field(p, col, "recordedBy") or _field(p, col, "occurrenceID"),
                    "lat": _field(p, col, "decimalLatitude"),
                    "lon": _field(p, col, "decimalLongitude"),
                    "observed_at": _field(p, col, "eventDate"),
                }
        print(f"occurrence 命中 1211 类集: {len(occ)} 条", flush=True)

        # multimedia → URL，按**媒体级** license 严筛。③红线：occurrence license ≠ 图片 license
        # （实测 occurrence 全 CC-BY 的源，图片级过半是 ©版权保留 / NC-SA）
        n = 0
        drop = 0
        per: dict[str, int] = {}
        seen: set[str] = set()
        with (
            z.open("multimedia.txt") as fh,
            (EUROPE / "europe_download_manifest.csv").open(
                "w", newline="", encoding="utf-8"
            ) as out,
        ):
            w = csv.DictWriter(out, fieldnames=OUT_FIELDS)
            w.writeheader()
            tf = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
            col = _cols(tf.readline())
            gi, ui, mli = col.get("gbifID"), col.get("identifier"), col.get("license")
            for line in tf:
                p = line.rstrip("\n").split("\t")
                if gi is None or ui is None or gi >= len(p) or ui >= len(p):
                    continue
                gid, url = p[gi], p[ui]
                if not url or url in seen or gid not in occ:
                    continue
                mlic = _media_license(p[mli] if mli is not None and mli < len(p) else "")
                if mlic == "other":  # 媒体级非 CC0/CC-BY（©/NC/SA/未知）→ 剔（红线）
                    drop += 1
                    continue
                seen.add(url)
                w.writerow({**occ[gid], "url": url, "license": mlic})  # 用媒体级 license
                per[occ[gid]["ebird_code"]] = per.get(occ[gid]["ebird_code"], 0) + 1
                n += 1
    print(f"媒体级 license 严筛：留 {n}（CC0/CC-BY）/ 剔 {drop}（©版权/NC/SA/未知）", flush=True)
    summary = {
        "role": "annotation-first 下载清单（GBIF Download API，媒体级 CC0/CC-BY）",
        "source": "GBIF Occurrence Download DWCA",
        "n_image_rows": n,
        "n_species": len(per),
        "n_dropped_media_license": drop,
        "note": "Aves+StillImage+活体观测+欧洲各国+扣iNat；**媒体级** license 严筛 CC0/CC-BY"
        "（≠occurrence 级，防©/NC 混入）；speciesKey→ebird_code",
    }
    (EUROPE / "europe_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\n清单落盘: {n} 图行 / {len(per)} 种 → europe_download_manifest.csv", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=KEY_FILE.read_text().strip() if KEY_FILE.exists() else None)
    a = ap.parse_args()
    if not a.key:
        raise SystemExit("无 download key（先提交 Download 请求）")
    if _already_parsed():
        print("清单已是 GBIF 全量，跳过（幂等，不重下 825MB）。", flush=True)
        return
    if _running():
        print("已有下载/解析实例在跑，跳过本次。", flush=True)
        return
    st = status(a.key)
    print(
        f"GBIF download {a.key}: {st.get('status')} | records={st.get('totalRecords')}", flush=True
    )
    if st.get("status") != "SUCCEEDED":
        print("未就绪，稍后再试（PREPARING/RUNNING）。", flush=True)
        return
    DL_DIR.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(os.getpid()))
    try:
        parse_to_manifest(download_zip(a.key, int(st.get("size") or 0)))
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
