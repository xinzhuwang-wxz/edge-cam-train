"""从 DatasetManifest 构建 torch Dataset / Lightning DataModule。

manifest 是数据准备(slice 1/2)的产物 → 训练/评估共用同一划分与类索引，保证可复现。"""

from __future__ import annotations

from collections.abc import Callable

import lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from edge_cam.contracts.schemas.dataset import DatasetManifest, Split
from edge_cam.train.classify.augment import (
    build_eval_transform,
    build_field_transform,
    build_train_transform,
)


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
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.degradation_strength = degradation_strength
        self.data_root = data_root

    def _loader(self, split: Split, transform: Callable, shuffle: bool) -> DataLoader:
        return DataLoader(
            ManifestDataset(self.manifest, split, transform, self.data_root),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        transform = build_train_transform(self.input_size, self.degradation_strength)
        return self._loader("train", transform, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", build_eval_transform(self.input_size), shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", build_eval_transform(self.input_size), shuffle=False)

    def field_dataloader(self) -> DataLoader:
        """「类现场」test loader（domain-gap 代理评估，slice 4 用）。"""
        transform = build_field_transform(self.input_size, self.degradation_strength)
        return self._loader("test", transform, shuffle=False)
