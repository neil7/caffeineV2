"""
caffeine.grad_extract
=====================
Per-layer weight and gradient extraction for a forget point z, plus write-back
of the unlearned weights into a live model.

Bridges a PyTorch model and the numpy dicts ``KFACUnlearner.unlearn`` expects:

    grad_W_per_layer : {layer_name: ∇W (d_out, d_in) float32}
    model_weights    : {layer_name:  W (d_out, d_in) float32}

All matrices are 2D, reshaped by ``layer_registry.weight_to_2d`` so Conv2d
weights match their (C_out, C_in·kH·kW) factored form. Write-back reshapes
back to each module's native weight shape before copying.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from caffeine.layer_registry import (
    instrumented_layers,
    weight_to_2d,
    weight_grad_2d,
)


def model_weights_per_layer(model: nn.Module) -> Dict[str, np.ndarray]:
    """
    Return {layer_name: W} as 2D float32 numpy arrays for every instrumented
    layer. These are copies — mutating them does not touch the live model
    (use ``write_back_weights`` to push updates back).
    """
    out: Dict[str, np.ndarray] = {}
    for name, module in instrumented_layers(model):
        w2d = weight_to_2d(module, module.weight)
        out[name] = w2d.detach().cpu().numpy().astype(np.float32)
    return out


def extract_grad_W_per_layer(
    model: nn.Module,
    z: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
) -> Dict[str, np.ndarray]:
    """
    Run a single forward/backward pass for the forget point z and collect each
    instrumented layer's weight gradient as a 2D float32 numpy array.

    Parameters
    ----------
    model     : the trained model (left in eval mode, grads zeroed first)
    z         : input batch for the forget point(s)
    target    : matching label(s)
    criterion : loss function, e.g. nn.CrossEntropyLoss()

    Returns
    -------
    {layer_name: ∇W (d_out, d_in)}  — ready for KFACUnlearner.unlearn
    """
    model.zero_grad(set_to_none=True)
    loss = criterion(model(z), target)
    loss.backward()

    out: Dict[str, np.ndarray] = {}
    for name, module in instrumented_layers(model):
        out[name] = weight_grad_2d(module, name)
    return out


def write_back_weights(
    model: nn.Module,
    model_weights: Dict[str, np.ndarray],
) -> int:
    """
    Copy updated 2D weight matrices back into the live model, reshaping to each
    module's native weight shape (Conv2d expands C_in·kH·kW back to 4D).

    Returns the number of layers written. Done under ``torch.no_grad()`` so the
    in-place copy is not tracked by autograd.
    """
    written = 0
    with torch.no_grad():
        for name, module in instrumented_layers(model):
            if name not in model_weights:
                continue
            w = torch.as_tensor(
                model_weights[name],
                dtype=module.weight.dtype,
                device=module.weight.device,
            )
            module.weight.copy_(w.reshape(module.weight.shape))
            written += 1
    return written
