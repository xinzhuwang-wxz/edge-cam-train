"""ManifestDataset / DataModule：从 manifest 取样、形状正确。"""

from __future__ import annotations

from pathlib import Path

import torch

from edge_cam.data.prep import DataPrepConfig, build_manifest
from edge_cam.train.classify.data import ClassifyDataModule, ManifestDataset


def _manifest(folder: Path):
    return build_manifest(DataPrepConfig(name="t", root=str(folder), seed=0))


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
