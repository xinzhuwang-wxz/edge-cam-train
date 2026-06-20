"""从 DatasetManifest 构建 torch Dataset / Lightning DataModule。

manifest 是数据准备(slice 1/2)的产物 → 训练/评估共用同一划分与类索引，保证可复现。"""

from __future__ import annotations

from collections.abc import Callable

import lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from edge_cam.contracts.schemas.dataset import DatasetManifest, Split
from edge_cam.train.classify.augment import (
    build_eval_transform,
    build_field_transform,
    build_train_transform,
)


def inverse_freq_class_weights(manifest: DatasetManifest, split: Split = "train") -> list[float]:
    """类不均衡 → 反频类权重（按 class_to_idx 顺序，长度 num_classes）。

    w_i = total / (num_classes * count_i)，均值≈1（缺类权重 0）；喂 CrossEntropyLoss(weight=)。
    """
    counts = manifest.class_counts(split)
    total = sum(counts.values()) or 1
    num = manifest.num_classes
    weights = [0.0] * num
    for name, idx in manifest.class_to_idx.items():
        cnt = counts.get(name, 0)
        weights[idx] = total / (num * cnt) if cnt else 0.0
    return weights


class ManifestDataset(Dataset):
    """读取 manifest 中某个 split 的样本，返回 (image_tensor, label_idx)。"""

    def __init__(
        self,
        manifest: DatasetManifest,
        split: Split,
        transform: Callable | None = None,
        data_root: str | None = None,
    ) -> None:
        self.manifest = manifest
        self.records = [r for r in manifest.records if r.split == split]
        self.class_to_idx = manifest.class_to_idx
        self.transform = transform or build_eval_transform()
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        record = self.records[idx]
        image = Image.open(self.manifest.resolve_path(record, self.data_root)).convert("RGB")
        return self.transform(image), self.class_to_idx[record.label]


class ClassifyDataModule(L.LightningDataModule):
    """train/val/test DataLoader；train 用退化增强，val/test 用干净变换。"""

    def __init__(
        self,
        manifest: DatasetManifest,
        input_size: int = 224,
        batch_size: int = 64,
        num_workers: int = 4,
        degradation_strength: float = 1.0,
        data_root: str | None = None,
        crop_scale_min: float = 0.7,
        crop_ratio: tuple[float, float] = (0.75, 1.333),
        balanced_sampler: bool = False,
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.degradation_strength = degradation_strength
        self.data_root = data_root
        self.crop_scale_min = crop_scale_min
        self.crop_ratio = crop_ratio
        # 训练用类平衡过采样（WeightedRandomSampler 按类反频，治长尾）；与 class-weighted CE 二选一
        self.balanced_sampler = balanced_sampler

    def _loader(self, split: Split, transform: Callable, shuffle: bool) -> DataLoader:
        return DataLoader(
            ManifestDataset(self.manifest, split, transform, self.data_root),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        transform = build_train_transform(
            self.input_size, self.degradation_strength, self.crop_scale_min, self.crop_ratio
        )
        if not self.balanced_sampler:
            return self._loader("train", transform, shuffle=True)
        ds = ManifestDataset(self.manifest, "train", transform, self.data_root)
        counts = self.manifest.class_counts("train")
        weights = [1.0 / counts[r.label] for r in ds.records]  # 反频 → 各类期望等量
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            sampler=sampler,  # 与 shuffle 互斥
            num_workers=self.num_workers,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", build_eval_transform(self.input_size), shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", build_eval_transform(self.input_size), shuffle=False)

    def field_dataloader(self) -> DataLoader:
        """「类现场」test loader（domain-gap 代理评估，slice 4 用）。"""
        transform = build_field_transform(self.input_size, self.degradation_strength)
        return self._loader("test", transform, shuffle=False)
