"""细分类 LightningModule：timm backbone + CE(label smoothing) + AdamW/cosine。

backbone 默认 EfficientNet-Lite0（plan §4.2 首选，去 SE/swish，最 NPU 安全）；
MobileNetV3-Large / RepVGG-A0 通过 config 切换做消融（plan §B.3）。"""

from __future__ import annotations

import lightning as L
import timm
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn


def topk_correct(logits: torch.Tensor, target: torch.Tensor, ks: tuple[int, ...]) -> dict[int, int]:
    """返回 {k: 命中数}，手算 top-k 避免额外依赖 torchmetrics。"""
    maxk = min(max(ks), logits.size(1))
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(target.view(-1, 1))
    return {k: int(correct[:, : min(k, maxk)].any(dim=1).sum().item()) for k in ks}


class Classifier(L.LightningModule):
    """timm 分类器封装；记录 top-1/top-5。"""

    def __init__(
        self,
        model_name: str = "efficientnet_lite0",
        num_classes: int = 1000,
        pretrained: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
        max_epochs: int = 80,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        hits = topk_correct(logits, targets, (1, 5))
        bs = targets.size(0)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=bs)
        self.log(f"{stage}_top1", hits[1] / bs, prog_bar=True, batch_size=bs)
        self.log(f"{stage}_top5", hits[5] / bs, batch_size=bs)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
