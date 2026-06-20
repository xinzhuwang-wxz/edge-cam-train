"""PTQ 校准集构建（plan §C.7：静默掉点首因）。

从 manifest 抽代表性图 → 一部分施加退化（夜视/噪声/压缩，贴近现场）→ 导 calib/ + dataset.txt
（pegasus 量化吃，plan §B.8）。检测/分类各建一份；优先真机回采图（早期不足用真实退化）。

本地 CPU 可跑。ORT-QDQ 模拟用的 calib_loader 可直接复用同一批图。"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image

from edge_cam.contracts.schemas.dataset import DatasetManifest, Split
from edge_cam.train.classify.augment import build_field_transform


def _tensor_to_pil(tensor) -> Image.Image:
    """field_transform 输出的归一化张量 → 反归一化回 uint8 PIL（仅为存盘可视）。"""
    import torch

    from edge_cam.train.classify.augment import IMAGENET_MEAN, IMAGENET_STD

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    arr = (tensor * std + mean).clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def build_calib_set(
    manifest: DatasetManifest,
    out_dir: str | Path,
    *,
    n: int = 500,
    split: Split = "train",
    degradation_ratio: float = 0.5,
    input_size: int = 224,
    seed: int = 0,
    data_root: str | None = None,
) -> Path:
    """抽 n 张图建校准集，degradation_ratio 比例施加退化；导 calib/ + dataset.txt。

    Returns:
        dataset.txt 路径（每行 "calib/xxx.jpg 0"，pegasus 量化用）。
    """
    out_dir = Path(out_dir)
    calib_dir = out_dir / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)

    records = [r for r in manifest.records if r.split == split]
    rng = random.Random(seed)
    rng.shuffle(records)
    records = records[:n]

    field = build_field_transform(input_size)
    lines: list[str] = []
    for i, record in enumerate(records):
        image = Image.open(manifest.resolve_path(record, data_root)).convert("RGB")
        if rng.random() < degradation_ratio:
            image = _tensor_to_pil(field(image))  # 夜视/噪声/压缩退化样本
        else:
            image = image.resize((input_size, input_size))
        rel = f"calib/{i:05d}.jpg"
        image.save(out_dir / rel, quality=90)
        lines.append(f"{rel} 0")

    dataset_txt = out_dir / "dataset.txt"
    dataset_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[calib] {len(lines)} 张 → {out_dir}（退化比例 {degradation_ratio:.0%}）")
    return dataset_txt
