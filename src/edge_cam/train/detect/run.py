"""检测训练入口（#14）：经 `TrainerBackend`(NanodetBackend)统一,不再 shell 直调 run_nanodet。

与分类 train 入口对等(都经 get_backend)→ TrainerBackend 两个 adapter 都真正在用。

cfg 形态(OmegaConf/dict):
    backend: nanodet
    detect: {config_path, checkpoint, nanodet_repo, nanodet_python}
    export: {enabled: true, out_path: ..., input_size: 416}

CLI:
    python -m edge_cam.train.detect.run --config <detect_train.yaml>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from edge_cam.train.backends import get_backend


def run_detect(cfg: Any) -> tuple[Any, Path | None]:
    """跑检测训练 + 可选导出 FP32 ONNX。返回 (训练句柄, onnx 路径|None)。

    经 get_backend(cfg.backend) → 与分类同一条训练 seam([[ADR-0003]] C1)。"""
    backend = get_backend(cfg.get("backend", "nanodet"))
    ref = backend.train(cfg)
    onnx = None
    export = cfg.get("export") or {}
    if export.get("enabled"):
        onnx = backend.export_fp32_onnx(ref, export["out_path"], export.get("input_size", 416))
    return ref, onnx


def main(argv: list[str] | None = None) -> None:
    import argparse

    import yaml
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="检测训练（经 TrainerBackend，#14）")
    parser.add_argument("--config", required=True, help="检测训练 cfg(yaml)")
    args = parser.parse_args(argv)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ref, onnx = run_detect(OmegaConf.create(raw))
    print(f"[detect] 训练完成 ref={ref}; onnx={onnx}")


if __name__ == "__main__":
    main()
