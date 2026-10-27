"""
experiments.unlearn
===================
Glue around KFACUnlearner: extract the forget point's per-layer gradients, run
the Newton step, write the corrected weights back into the live model.
"""

from typing import Callable, Dict, List

import numpy as np
import torch

from caffeine.grad_extract import (
    extract_grad_W_per_layer,
    model_weights_per_layer,
    write_back_weights,
)


def _encode_z(z: torch.Tensor) -> np.ndarray:
    """Flatten a forget-point input to a 1D float32 array for the cache hash."""
    return z.detach().reshape(-1).cpu().numpy().astype(np.float32)


def unlearn_point(
    model,
    unlearner,
    z: torch.Tensor,
    target: torch.Tensor,
    z_id: str,
    criterion,
    checkpoint_step: int,
    scale: float = 1.0,
):
    """
    Remove one forget point from `model` in place.

    `scale` lets the caller apply the leave-one-out 1/n factor to the applied
    update (pass scale=1/n to match the parameter change θ₋ᵢ − θ ≈ (1/n)H⁻¹∇Lᵢ;
    pass 1.0 for the raw Newton correction). The numpy engine is untouched.
    """
    grads = extract_grad_W_per_layer(model, z, target, criterion)
    if scale != 1.0:
        grads = {k: (v * scale).astype(np.float32) for k, v in grads.items()}
    weights = model_weights_per_layer(model)
    result = unlearner.unlearn(
        model_weights=weights,
        grad_W_per_layer=grads,
        z=_encode_z(z),
        z_id=z_id,
        checkpoint_step=checkpoint_step,
    )
    write_back_weights(model, weights)
    return result


def unlearn_set(
    model,
    unlearner,
    forget_batches,             # iterable of (z, target, z_id)
    criterion,
    checkpoint_step: int,
    scale: float = 1.0,
) -> Dict:
    """Unlearn a set of points; accumulate total time and influence norm."""
    total_time_ms = 0.0
    sq_norm = 0.0
    results: List = []
    for z, target, z_id in forget_batches:
        r = unlearn_point(
            model, unlearner, z, target, z_id, criterion, checkpoint_step, scale
        )
        total_time_ms += r.total_time_ms
        sq_norm += sum(v ** 2 for v in r.delta_norms.values())
        results.append(r)
    return {
        "n_points": len(results),
        "total_time_ms": round(total_time_ms, 2),
        "influence_norm": float(np.sqrt(sq_norm)),
        "results": results,
    }
