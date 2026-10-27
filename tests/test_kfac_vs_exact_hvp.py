"""
Correctness gate: K-FAC influence vs exact-Hessian influence.

This is the load-bearing test. It validates the whole numpy K-FAC machinery
(KFACCollector A/G + batch-N scaling -> kfac_newton_step + 1/n) against the
exact influence definition from Koh & Liang 2017:

    influence(z_train) = (H⁻¹ ∇L_test) · ∇L_train(z) / n

computed with the FULL Hessian (torch double-backprop). Block-diagonal K-FAC is
a deliberate approximation, so the gate asserts a strong *correlation*, not
numeric equality.

Tier 1 — single Linear logistic regression: K-FAC's only error is the Kronecker
factorization E[a⊗g] ≈ E[a]⊗E[g] -> very high correlation.
Tier 2 — 2-layer MLP: block-diagonal + factorization error -> looser bound.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

torch.manual_seed(17)

scipy_stats = pytest.importorskip("scipy.stats")
pearsonr = scipy_stats.pearsonr

from caffeine.kfac_collector import KFACCollector
from caffeine.layer_registry import instrumented_layers
from kfac_unlearn import kfac_newton_step

EPS = 1e-2          # damping / Hessian ridge, used identically in both paths


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flat_grad(model, loss, params):
    g = torch.autograd.grad(loss, params, create_graph=False, retain_graph=True)
    return torch.cat([gi.reshape(-1) for gi in g])


def _full_hessian(model, X, y, params):
    """Exact Hessian of the mean CE loss w.r.t. the flattened params."""
    crit = nn.CrossEntropyLoss()
    flat0 = torch.cat([p.reshape(-1) for p in params]).detach()
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]

    def loss_of_flat(flat):
        idx = 0
        tensors = []
        for sh, ne in zip(shapes, numels):
            tensors.append(flat[idx:idx + ne].reshape(sh))
            idx += ne
        # functional forward using the supplied weights
        out = _functional_forward(model, X, tensors, params)
        return crit(out, y)

    H = torch.autograd.functional.hessian(loss_of_flat, flat0)
    return H.detach().numpy()


def _functional_forward(model, X, tensors, params):
    """Run model with `tensors` substituted for `params` (single/2-layer linear)."""
    # map params (by identity) to provided tensors
    repl = {id(p): t for p, t in zip(params, tensors)}
    h = X
    linears = [m for _, m in instrumented_layers(model)]
    for k, lin in enumerate(linears):
        W = repl[id(lin.weight)]
        b = lin.bias
        h = h @ W.t() + (b if b is not None else 0)
        if k < len(linears) - 1:
            h = torch.relu(h)
    return h


def _exact_influence(model, X, y, x_test, y_test, params, n):
    crit = nn.CrossEntropyLoss()
    H = _full_hessian(model, X, y, params)
    H += EPS * np.eye(H.shape[0])
    H_inv = np.linalg.inv(H)

    test_loss = crit(model(x_test), y_test)
    g_test = _flat_grad(model, test_loss, params).detach().numpy()
    hinv_test = H_inv @ g_test

    infl = np.zeros(len(X))
    for i in range(len(X)):
        li = crit(model(X[i:i + 1]), y[i:i + 1])
        gi = _flat_grad(model, li, params).detach().numpy()
        infl[i] = hinv_test @ gi / n
    return infl


def _kfac_influence(model, X, y, x_test, y_test, n):
    """K-FAC influence using collected A,G and kfac_newton_step (no 1/n inside)."""
    crit = nn.CrossEntropyLoss()
    layers = instrumented_layers(model)

    # collect A, G over the full training set
    collector = KFACCollector(model)
    with collector:
        model.zero_grad(set_to_none=True)
        loss = crit(model(X), y)
        loss.backward()

    A_inv, G_inv = {}, {}
    for name, _ in layers:
        A, G = collector.factors(name)
        A_inv[name] = np.linalg.inv(A + EPS * np.eye(A.shape[0], dtype=np.float32))
        G_inv[name] = np.linalg.inv(G + EPS * np.eye(G.shape[0], dtype=np.float32))

    # H⁻¹ ∇L_test, per layer (note: -kfac_newton_step gives +H⁻¹v)
    model.zero_grad(set_to_none=True)
    crit(model(x_test), y_test).backward()
    hinv_test = {}
    for name, mod in layers:
        gt = mod.weight.grad.detach().cpu().numpy().astype(np.float32)
        hinv_test[name] = -kfac_newton_step(gt, A_inv[name], G_inv[name])

    infl = np.zeros(len(X))
    for i in range(len(X)):
        model.zero_grad(set_to_none=True)
        crit(model(X[i:i + 1]), y[i:i + 1]).backward()
        s = 0.0
        for name, mod in layers:
            gi = mod.weight.grad.detach().cpu().numpy().astype(np.float32)
            s += float(np.sum(hinv_test[name] * gi))
        infl[i] = s / n
    return infl


def _make_data(n=400, d=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(d, generator=g)
    X = torch.randn(n, d, generator=g)
    logits = X @ w
    y = (logits + 0.3 * torch.randn(n, generator=g) > 0).long()
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# tests
# ─────────────────────────────────────────────────────────────────────────────

def _train(model, X, y, steps=300, lr=0.05):
    # Bounded Adam (not trained to saturation): keeps per-example CE gradients
    # non-vanishing so influence scores are well-defined. weight_decay keeps the
    # logits away from the ±∞ saturation that zeroes out the gradients.
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        opt.step()


def test_logreg_kfac_correlates_with_exact():
    torch.manual_seed(17)
    X, y = _make_data(n=400, d=10, seed=1)
    n = len(X)
    model = nn.Sequential(nn.Linear(10, 2, bias=False))
    _train(model, X, y)

    params = [model[0].weight]
    Xt, yt = _make_data(n=20, d=10, seed=99)
    x_test, y_test = Xt[:16], yt[:16]      # test batch (mirrors Koh's test set)

    exact = _exact_influence(model, X, y, x_test, y_test, params, n)
    kfac = _kfac_influence(model, X, y, x_test, y_test, n)

    r = pearsonr(exact, kfac)[0]
    assert r > 0.85, f"single-layer K-FAC vs exact correlation too low: r={r:.4f}"


def test_mlp_kfac_correlates_with_exact():
    torch.manual_seed(17)
    X, y = _make_data(n=400, d=10, seed=2)
    n = len(X)
    model = nn.Sequential(
        nn.Linear(10, 8, bias=False),
        nn.ReLU(),
        nn.Linear(8, 2, bias=False),
    )
    _train(model, X, y, steps=300)

    params = [model[0].weight, model[2].weight]
    Xt, yt = _make_data(n=20, d=10, seed=77)
    x_test, y_test = Xt[:16], yt[:16]      # test batch avoids dead-ReLU zero grad

    exact = _exact_influence(model, X, y, x_test, y_test, params, n)
    kfac = _kfac_influence(model, X, y, x_test, y_test, n)

    r = pearsonr(exact, kfac)[0]
    assert r > 0.5, f"MLP block-diagonal K-FAC vs exact correlation too low: r={r:.4f}"
