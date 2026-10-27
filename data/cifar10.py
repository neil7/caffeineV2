"""
data.cifar10
===========
CIFAR-10 loaders (torchvision) with the standard normalization and train-time
augmentation. Downloads to `root` on first use (kept out of git via .gitignore).
"""

from typing import Tuple

import torch
import torchvision
import torchvision.transforms as T

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _transforms(augment: bool) -> T.Compose:
    ops = []
    if augment:
        ops += [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    ops += [T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)]
    return T.Compose(ops)


def cifar10_loaders(
    root: str = "data/raw",
    batch_size: int = 128,
    augment: bool = True,
    num_workers: int = 8,
    download: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torchvision.datasets.CIFAR10]:
    """Return (train_loader, test_loader, train_dataset)."""
    train_ds = torchvision.datasets.CIFAR10(
        root, train=True, download=download, transform=_transforms(augment)
    )
    test_ds = torchvision.datasets.CIFAR10(
        root, train=False, download=download, transform=_transforms(False)
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader, train_ds
