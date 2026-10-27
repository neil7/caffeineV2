"""Model wrappers + K-FAC layer enumeration for each modality."""

from models.resnet18 import resnet18_cifar, kfac_layers as resnet_kfac_layers

__all__ = ["resnet18_cifar", "resnet_kfac_layers"]
