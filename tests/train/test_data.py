"""ManifestDataset / DataModule：从 manifest 取样、形状正确。"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler

from edge_cam.contracts.schemas.dataset import DatasetManifest, SampleRecord
from edge_cam.data.prep import DataPrepConfig, build_manifest
from edge_cam.train.classify.data import (
    ClassifyDataModule,
    ManifestDataset,
    inverse_freq_class_weights,
)


def _manifest(folder: Path):
    return build_manifest(DataPrepConfig(name="t", root=str(folder), seed=0))


def _imbalanced_manifest() -> DatasetManifest:
    # A:6 train / B:2 train / C:仅 val（train 缺）—— 制造长尾
    recs = [SampleRecord(path=f"a{i}.jpg", label="A", split="train") for i in range(6)]
    recs += [SampleRecord(path=f"b{i}.jpg", label="B", split="train") for i in range(2)]
    recs += [SampleRecord(path="c0.jpg", label="C", split="val")]
    return DatasetManifest(
        name="t", version="v0", seed=0, class_to_idx={"A": 0, "B": 1, "C": 2}, records=recs
    )


def test_inverse_freq_class_weights() -> None:
    w = inverse_freq_class_weights(_imbalanced_manifest(), "train")
    # total=8, num=3 → A:8/(3*6)、B:8/(3*2)；C train 缺 → 0
    assert len(w) == 3
    assert abs(w[0] - 8 / 18) < 1e-6 and abs(w[1] - 8 / 6) < 1e-6
    assert w[1] > w[0] and w[2] == 0.0  # 稀有类 B 权重更高；缺类 0


def test_balanced_sampler_uses_inverse_freq_weights() -> None:
    dm = ClassifyDataModule(
        _imbalanced_manifest(), input_size=64, batch_size=2, num_workers=0, balanced_sampler=True
    )
    loader = dm.train_dataloader()
    assert isinstance(loader.sampler, WeightedRandomSampler)
    ws = list(loader.sampler.weights)
    assert len(ws) == 8  # 6 A + 2 B（train）
    assert abs(ws[0] - 1 / 6) < 1e-6 and abs(ws[-1] - 1 / 2) < 1e-6  # A=1/6, B=1/2


def test_default_loader_has_no_sampler(flat_imagefolder: Path) -> None:
    dm = ClassifyDataModule(_manifest(flat_imagefolder), input_size=64, batch_size=2, num_workers=0)
    assert dm.train_dataloader().sampler is not None  # 默认 RandomSampler（shuffle），非 Weighted
    assert not isinstance(dm.train_dataloader().sampler, WeightedRandomSampler)


def test_dataset_getitem(flat_imagefolder: Path) -> None:
    manifest = _manifest(flat_imagefolder)
    ds = ManifestDataset(manifest, "train")
    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert isinstance(label, int)
    assert 0 <= label < manifest.num_classes


def test_dataset_only_split(flat_imagefolder: Path) -> None:
    manifest = _manifest(flat_imagefolder)
    train_n = manifest.counts_by_split()["train"]
    assert len(ManifestDataset(manifest, "train")) == train_n


def test_datamodule_train_loader_batches(flat_imagefolder: Path) -> None:
    manifest = _manifest(flat_imagefolder)
    dm = ClassifyDataModule(manifest, input_size=64, batch_size=2, num_workers=0)
    images, labels = next(iter(dm.train_dataloader()))
    assert images.shape[1:] == (3, 64, 64)
    assert images.shape[0] == labels.shape[0]
