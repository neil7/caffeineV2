"""
caffeine.layer_registry
========================
Shared policy for which modules carry a K-FAC factorizable weight, plus the
2D-reshape rules that make Conv2d weights look like the (d_out, d_in) matrices
`kfac_newton_step` expects.

Instrument Linear and Conv2d only. Skip BatchNorm/LayerNorm/ReLU/MaxPool/
Embedding/Dropout — they have no W = A ⊗ G structure.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Module types whose weights factor as H_W ≈ A ⊗ G.
INSTRUMENTED_TYPES = (nn.Linear, nn.Conv2d)


def instrumented_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Return [(name, module), ...] for every Linear / Conv2d in the model,
    using `named_modules()` so the names are HDF5-safe (dots, no slashes) and
    match PyTorch's named_parameters() convention.
    """
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, INSTRUMENTED_TYPES)
    ]


def weight_to_2d(module: nn.Module, weight: torch.Tensor) -> torch.Tensor:
    """
    Reshape a layer's weight tensor to the 2D (d_out, d_in) form used by K-FAC.

      Linear : weight is already (d_out, d_in)
      Conv2d : (C_out, C_in, kH, kW) -> (C_out, C_in*kH*kW)   (-1 absorbs the rest)
    """
    if isinstance(module, nn.Conv2d):
        return weight.reshape(weight.shape[0], -1)
    return weight


def weight_grad_2d(module: nn.Module, name: str = "") -> np.ndarray:
    """
    Take module.weight.grad, reshape to 2D per `weight_to_2d`, return float32
    numpy. Raises if the gradient has not been populated (no backward pass).
    """
    if module.weight.grad is None:
        raise RuntimeError(
            f"Layer '{name or module}' has no weight.grad — run loss.backward() first."
        )
    g2d = weight_to_2d(module, module.weight.grad)
    return g2d.detach().cpu().numpy().astype(np.float32)
