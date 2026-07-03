"""检测数据获取 CLI（ADR-0006 D3）：`acquire` → raw + `_acquire.json` 收据；`--list` 导出
「数据从哪来」的人读清单（D1，从各已注册 adapter 的 DatasetSpec.acquire 汇出，无需另维护 yaml）。

用法：
  python -m edge_cam.data.adapters.detect.acquire --list
  python -m edge_cam.data.adapters.detect.acquire --config <build.yaml> [--source ena24] [--dry-run]
"""

from __future__ import annotations

from edge_cam.data.adapters.detect.base import available_adapters, build_adapter


def list_sources(raw_root: str = ".") -> list[dict]:
    """导出所有已注册 adapter 的 acquire 声明（数据来源单一事实源，D1）。raw_root 仅用于构造。"""
    out: list[dict] = []
    for name in available_adapters():
        try:
            spec = build_adapter(name, raw_root).spec
        except Exception:  # noqa: BLE001 — 构造失败（缺可选依赖等）不阻塞清单
            continue
        acq = spec.acquire
        out.append(
            {
                "name": name,
                "license": spec.license,
                "commercial_safe": spec.commercial_safe,
                "role": spec.role,
                "acquire": None
                if acq is None
                else {"method": acq.method, "version": acq.version, "urls": list(acq.urls)},
            }
        )
    return out


def _print_list() -> None:
    for s in list_sources():
        acq = s["acquire"]
        head = f"{s['name']:<20} {s['license']:<24} role={s['role']}"
        if acq is None:
            print(f"{head}  acquire=<未声明>")
            continue
        print(f"{head}  method={acq['method']} version={acq['version']}")
        for u in acq["urls"]:
            print(f"    ↳ {u}")


def main(argv: list[str] | None = None) -> None:
    import argparse
    import datetime

    p = argparse.ArgumentParser(description="检测数据获取（ADR-0006 D3）")
    p.add_argument("--list", action="store_true", help="导出全源来源清单（不下载）")
    p.add_argument("--config", default=None, help="build config（选哪些源）")
    p.add_argument("--source", default=None, help="只 acquire 指定源（缺省=config 里全部）")
    p.add_argument("--dry-run", action="store_true", help="只打印将 acquire 什么，不真下")
    args = p.parse_args(argv)

    if args.list or not args.config:
        _print_list()
        return

    from edge_cam.data.adapters.detect.build import DetectBuildConfig

    cfg = DetectBuildConfig.from_yaml(args.config)
    now = datetime.datetime.now().isoformat(timespec="seconds")  # noqa: DTZ005 — 收据本地时间戳即可
    names = [args.source] if args.source else list(cfg.datasets)
    for name in names:
        ad = build_adapter(name, cfg.raw_root, **(cfg.datasets.get(name) or {}))
        if args.dry_run:
            print(f"[dry-run] {name}: acquire={ad.spec.acquire}")
            continue
        receipt = ad.acquire(cfg.raw_root, now=now)
        print(f"[acquire] {name} → {receipt.method} ({receipt.image_count} imgs)")


if __name__ == "__main__":
    main()
