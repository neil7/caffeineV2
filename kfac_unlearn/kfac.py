"""
kfac_unlearn.kfac
=================
The core unlearning computation — the K-FAC Newton step.

Influence-function update (Koh & Liang 2017):
    Δθ ≈ −H⁻¹ ∇θ L(z, θ̂)

K-FAC approximation per layer:
    H_W ≈ A ⊗ G   →   H⁻¹ vec(∇W) = (A⁻¹ ⊗ G⁻¹) vec(∇W) = vec(G⁻¹ · ∇W · A⁻ᵀ)

This avoids ever forming the full Kronecker product. Pure numpy, float32.
"""

import numpy as np


def kfac_newton_step(
    grad_W: np.ndarray,
    A_inv: np.ndarray,
    G_inv: np.ndarray,
) -> np.ndarray:
    """
    Compute Δθ for one layer using the K-FAC identity:

        (A ⊗ G)⁻¹ · vec(∇W)  =  vec( G⁻¹ · ∇W · A⁻ᵀ )

    This avoids forming the full Kronecker product at any point.
    Cost: two matrix multiplications O(d³) instead of O(d⁶).

    Parameters
    ----------
    grad_W : (d_out, d_in) gradient of loss w.r.t. weight matrix W
    A_inv  : (d_in, d_in)  inverse input covariance
    G_inv  : (d_out, d_out) inverse output gradient covariance

    Returns
    -------
    delta_W : (d_out, d_in) parameter update for this layer

    Note
    ----
    This returns the raw Newton step −H⁻¹∇W (no 1/n training-set
    normalization). For influence-on-loss scores that match Koh & Liang's
    `predicted_loss_diff = (H⁻¹∇L_test)·∇L_train / n`, apply the 1/n factor
    at the call site (see benchmarks/evaluate.py).
    """
    # Δθ = −H⁻¹ ∇θ L  →  for one layer: −G⁻¹ · ∇W · A⁻ᵀ
    return -G_inv @ grad_W @ A_inv.T
