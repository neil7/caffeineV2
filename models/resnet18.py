"""
models.resnet18
==============
ResNet-18 adapted for CIFAR-10: 3×3 stride-1 stem, no max-pool, 10-way head.
(torchvision's default stem is built for 224×224 ImageNet and destroys CIFAR's
32×32 detail — the standard CIFAR fix is a 3×3 conv1 and Identity maxpool.)
"""

from typing import List

import torch.nn as nn
import torchvision

from caffeine.layer_registry import instrumented_layers


def resnet18_cifar(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """ResNet-18 with a CIFAR-style stem and a `num_classes` linear head."""
    weights = "IMAGENET1K_V1" if pretrained else None
    m = torchvision.models.resnet18(weights=weights)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def kfac_layers(model: nn.Module) -> List[str]:
    """
    Names of layers to instrument for K-FAC: every Conv2d (stem, residual
    blocks, downsample projections) plus the final `fc`. BatchNorm/ReLU/pool
    are skipped by `instrumented_layers`.
    """
    return [name for name, _ in instrumented_layers(model)]
