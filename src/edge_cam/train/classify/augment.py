"""退化增强（plan §6）：模拟边侧相机的真实退化，缩小 domain gap。

夜视灰度 / 低照噪声 / H.264 压缩伪影 / 运动模糊 / 远距离降采样。
同一组退化既作**训练增强**，也作 slice 4 的**「类现场」测试变换**（domain-gap 代理；
注意：是代理不是真现场，见 CONTEXT.md）。

用 torchvision.transforms.v2，全部 CPU 可跑。"""

from __future__ import annotations

import torch
from torchvision.transforms import v2

# timm/ImageNet 默认归一化；真实训练应取 timm data_config 的 mean/std（export 前可 fold 进首层）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AddGaussianNoise(v2.Transform):
    """低照噪声：在归一化后的张量上叠加高斯噪声。"""

    def __init__(self, std: float = 0.05) -> None:
        super().__init__()
        self.std = std

    def transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        if not isinstance(inpt, torch.Tensor):
            return inpt
        return inpt + torch.randn_like(inpt) * self.std


class RandomDownscale(v2.Transform):
    """远距离降采样：缩小再放大回原尺寸，丢细节。"""

    def __init__(self, min_scale: float = 0.4, max_scale: float = 0.8) -> None:
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        if not isinstance(inpt, torch.Tensor):
            return inpt
        h, w = inpt.shape[-2:]
        scale = float(torch.empty(1).uniform_(self.min_scale, self.max_scale).item())
        small = v2.functional.resize(inpt, [max(1, int(h * scale)), max(1, int(w * scale))])
        return v2.functional.resize(small, [h, w])


def _degradation_block(strength: float) -> list[v2.Transform]:
    """uint8 域的退化算子序列（概率随 strength 缩放）。"""
    p = strength
    return [
        v2.RandomGrayscale(p=0.3 * p),  # 夜视灰度
        v2.RandomApply([v2.JPEG(quality=(30, 75))], p=0.5 * p),  # H.264/JPEG 压缩伪影
        v2.RandomApply([RandomDownscale(0.4, 0.8)], p=0.5 * p),  # 远距离降采样
        # 运动模糊近似
        v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0))], p=0.4 * p),
    ]


def build_train_transform(size: int = 224, degradation_strength: float = 1.0) -> v2.Compose:
    """训练变换：几何增强 + 退化增强 + 归一化（+ 低照噪声）。"""
    return v2.Compose(
        [
            v2.PILToTensor(),
            v2.RandomResizedCrop(size, scale=(0.7, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            *_degradation_block(degradation_strength),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            v2.RandomApply([AddGaussianNoise(0.04)], p=0.3 * degradation_strength),
        ]
    )


def build_eval_transform(size: int = 224) -> v2.Compose:
    """干净评估变换：resize + 归一化（验证集口径）。"""
    return v2.Compose(
        [
            v2.PILToTensor(),
            v2.Resize([size, size], antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_field_transform(size: int = 224, strength: float = 1.0) -> v2.Compose:
    """「类现场」变换（slice 4 domain-gap 代理）：对评估图施加确定性较强的退化。

    ⚠️ 这是 domain gap 的**代理估计**，非真现场（plan §8 的 99.5%→88% 教训）。"""
    return v2.Compose(
        [
            v2.PILToTensor(),
            v2.Resize([size, size], antialias=True),
            v2.Grayscale(num_output_channels=3),
            v2.JPEG(quality=(25, 45)),
            RandomDownscale(0.35, 0.55),
            v2.GaussianBlur(kernel_size=5, sigma=(1.0, 2.5)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            AddGaussianNoise(0.05 * strength),
        ]
    )
