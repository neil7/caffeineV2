"""
benchmarks.evaluate
==================
The four CLAUDE.md unlearning metrics plus a correlation sanity check that
mirrors influence-release `experiments.test_retraining`.

  1. forget_acc   — accuracy on the forget set (should approach the retrained model's)
  2. retain_acc   — accuracy on the retain set (must not degrade)
  3. unlearn_time — wall-clock (ms) from UnlearningResult
  4. influence_norm — ‖Δθ‖ across layers/points (bounded sanity check)

The correlation check ports Koh & Liang: predicted influence carries the explicit
1/n training-set normalization, dotted against per-train-point gradients.
"""

import math
from typing import Callable, Dict, List

import numpy as np
import torch

from caffeine.grad_extract import extract_grad_W_per_layer
from caffeine.layer_registry import instrumented_layers
from kfac_unlearn import kfac_newton_step


@torch.no_grad()
def _accuracy(model, loader, device, forward):
    model.eval()
    correct = total = 0
    for batch in loader:
        logits, labels = forward(model, batch, device)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def evaluate(model, retain_loader, forget_loader, test_loader, device, forward) -> Dict:
    """forget / retain / test accuracy."""
    return {
        "forget_acc": _accuracy(model, forget_loader, device, forward),
        "retain_acc": _accuracy(model, retain_loader, device, forward),
        "test_acc": _accuracy(model, test_loader, device, forward),
    }


def forget_gap(unlearned_model, retrained_model, forget_loader, device, forward) -> float:
    """|acc_unlearned − acc_retrained| on the forget set (lower is better)."""
    a = _accuracy(unlearned_model, forget_loader, device, forward)
    b = _accuracy(retrained_model, forget_loader, device, forward)
    return abs(a - b)


def influence_norm(unlearn_results: List) -> float:
    """L2 norm of all per-layer Δθ across the unlearned points."""
    sq = 0.0
    for r in unlearn_results:
        sq += sum(v ** 2 for v in r.delta_norms.values())
    return math.sqrt(sq)


def unlearn_time_ms(unlearn_results: List) -> float:
    return float(sum(r.total_time_ms for r in unlearn_results))


def _kfac_hinv_test(model, store, checkpoint_step, test_grad_W, damping):
    """H⁻¹ ∇L_test per layer using cached/loaded K-FAC factors (no minus sign)."""
    hinv = {}
    for name, _ in instrumented_layers(model):
        A_inv, G_inv = store.load_kfac_inverse(
            name, store.nearest_checkpoint(name, checkpoint_step), damping=damping
        )
        # -kfac_newton_step(g) = +(A⊗G)⁻¹ vec(g)
        hinv[name] = -kfac_newton_step(test_grad_W[name], A_inv, G_inv)
    return hinv


def predicted_influence(
    model, store, checkpoint_step, damping,
    test_z, test_y, train_points, criterion, n_train,
) -> np.ndarray:
    """
    Koh & Liang predicted influence per train point:
        infl = (H⁻¹ ∇L_test) · ∇L_train / n
    `train_points` is an iterable of (z, y). Returns one score per point.
    """
    test_grad_W = extract_grad_W_per_layer(model, test_z, test_y, criterion)
    hinv = _kfac_hinv_test(model, store, checkpoint_step, test_grad_W, damping)

    scores = []
    for z, y in train_points:
        g = extract_grad_W_per_layer(model, z, y, criterion)
        s = sum(float(np.sum(hinv[name] * g[name])) for name in hinv)
        scores.append(s / n_train)
    return np.array(scores)


def correlation_check(
    predicted: np.ndarray, actual: np.ndarray
) -> float:
    """
    Pearson correlation of predicted influence vs actual retrain-without-point
    loss change — the influence-function faithfulness metric. `actual` is
    supplied by benchmarks.retrain_baseline.
    """
    from scipy.stats import pearsonr
    return float(pearsonr(predicted, actual)[0])
