"""
experiments.layer_scores.ekfac_math
====================================
Pure-numpy EK-FAC score math (model-agnostic core, Ground Rule #6). No torch
dependency here so the correctness tests (T1-T4) can run against plain numpy
fixtures, independent of any model/hook plumbing.

Vec / Kronecker convention (pinned down empirically by test T1)
-----------------------------------------------------------------
For a layer y = Wx with W of shape (d_out, d_in):

    A = E[x xT]   (d_in, d_in),   Q_A, a   = eigh(A)   (descending)
    G = E[gT g]   (d_out, d_out), Q_G, gam = eigh(G)   (descending)
    C = Q_G.T @ V @ Q_A                      (rotate a (d_out,d_in) matrix
                                               V into the eigenbasis)
    Delta_theta* = Q_G @ (C / (Lambda + lam)) @ Q_A.T

Using column-major vec (`vec(X) = X.flatten(order="F")`, stacking columns),
the identity `vec(U @ Y @ W) = (W.T kron U) @ vec(Y)` gives:

    vec(Delta_theta*) = (Q_A kron Q_G) @ diag(1 / vec(Lambda + lam)) @ (Q_A kron Q_G).T @ vec(V)

i.e. the damped EK-FAC Hessian block is

    H = (Q_A kron Q_G) @ diag(vec(Lambda + lam)) @ (Q_A kron Q_G).T

(NOT `Q_G kron Q_A` as a naive reading of the Kronecker notation might
suggest — the appended CLAUDE.md spec's literal `(Q_G kron Q_A)` phrasing is
loose; this module's convention is the one that reproduces the spec's own
unambiguous formulas for `Delta_theta*` and `s_l`, verified by test T1/T2.)
"""

from typing import Tuple

import numpy as np


def vec(x: np.ndarray) -> np.ndarray:
    """Column-major vectorization: stacks columns of x top to bottom."""
    return x.flatten(order="F")


def unvec(v: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    return v.reshape(shape, order="F")


def eigh_factors(
    A: np.ndarray, G: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    A = Q_A diag(a) Q_A.T, G = Q_G diag(gamma) Q_G.T.

    Computed in float64 (torch.linalg.eigh / np.linalg.eigh return ascending
    eigenvalues; reordered descending here), returned as float32.
    """
    a, Q_A = np.linalg.eigh(A.astype(np.float64))
    gamma, Q_G = np.linalg.eigh(G.astype(np.float64))
    order_a = np.argsort(a)[::-1]
    order_g = np.argsort(gamma)[::-1]
    Q_A = Q_A[:, order_a].astype(np.float32)
    a = a[order_a].astype(np.float32)
    Q_G = Q_G[:, order_g].astype(np.float32)
    gamma = gamma[order_g].astype(np.float32)
    return Q_A, a, Q_G, gamma


def per_layer_damping(Lambda: np.ndarray, alpha: float) -> float:
    """lambda_l = alpha * mean(Lambda_l)."""
    return float(alpha * np.mean(Lambda))


def global_damping(mean_lambda_all_layers: float, alpha: float) -> float:
    """lambda = alpha * mean over all layers' Lambda."""
    return float(alpha * mean_lambda_all_layers)


def rotate_to_eigenbasis(V: np.ndarray, Q_A: np.ndarray, Q_G: np.ndarray) -> np.ndarray:
    """C = Q_G.T @ V @ Q_A, shape (d_out, d_in)."""
    return Q_G.T @ V @ Q_A


def layer_score(C: np.ndarray, Lambda: np.ndarray, lam: float) -> float:
    """s_l = sum_ij C_ij^2 / (Lambda_ij + lam)."""
    return float(np.sum(C.astype(np.float64) ** 2 / (Lambda.astype(np.float64) + lam)))


def full_step(
    C: np.ndarray, Q_A: np.ndarray, Q_G: np.ndarray, Lambda: np.ndarray, lam: float
) -> np.ndarray:
    """Delta_theta*_l = Q_G @ (C / (Lambda + lam)) @ Q_A.T."""
    return Q_G @ (C / (Lambda + lam)) @ Q_A.T


def ihvp_factored(
    V: np.ndarray, Q_A: np.ndarray, Q_G: np.ndarray, Lambda: np.ndarray, lam: float
) -> np.ndarray:
    """Factored inverse-Hessian-vector-product: H^{-1} vec(V), reshaped to V's shape."""
    C = rotate_to_eigenbasis(V, Q_A, Q_G)
    return full_step(C, Q_A, Q_G, Lambda, lam)


def dense_H(Q_A: np.ndarray, Q_G: np.ndarray, Lambda: np.ndarray, lam: float) -> np.ndarray:
    """
    Explicit damped EK-FAC Hessian block, float64, for correctness test T1.
    Shape (d_out*d_in, d_out*d_in), matching `vec` (column-major) ordering.
    """
    Q = np.kron(Q_A.astype(np.float64), Q_G.astype(np.float64))  # (d_in*d_out, d_in*d_out)
    diag = vec(Lambda.astype(np.float64)) + lam
    return (Q * diag) @ Q.T


def full_bytes(d_in: int, d_out: int) -> int:
    """fp32 bytes for Q_A, Q_G, Lambda: 4 * (d_in^2 + d_out^2 + d_in*d_out)."""
    return 4 * (d_in * d_in + d_out * d_out + d_in * d_out)


def sketch_bytes(r_A: int, r_G: int, d_in: int, d_out: int) -> int:
    return 4 * (r_A * d_in + r_G * d_out + r_A * r_G + 1)


def rank_r_sketch(
    V: np.ndarray,
    Q_A: np.ndarray,
    Q_G: np.ndarray,
    Lambda: np.ndarray,
    lam: float,
    r: int,
) -> dict:
    """
    Rank-r sketch score and bounds for one layer, per the spec's definitions.

    Returns dict with: r_A, r_G, s_in, L, U, m_in, m_out, sketch_bytes.
    """
    d_in = Q_A.shape[0]
    d_out = Q_G.shape[0]
    r_A = min(r, d_in)
    r_G = min(r, d_out)

    U_A = Q_A[:, :r_A]
    U_G = Q_G[:, :r_G]
    Lambda_blk = Lambda[:r_G, :r_A]

    mask = np.ones_like(Lambda, dtype=bool)
    mask[:r_G, :r_A] = False
    if r == 0 or not mask.any():
        tau = float(Lambda.max()) if r == 0 else 0.0
    else:
        tau = float(Lambda[mask].max())

    C_r = U_G.T @ V @ U_A
    m_in = float(np.sum(C_r.astype(np.float64) ** 2))
    v_fro2 = float(np.sum(V.astype(np.float64) ** 2))
    m_out = max(v_fro2 - m_in, 0.0)

    s_in = float(np.sum(C_r.astype(np.float64) ** 2 / (Lambda_blk.astype(np.float64) + lam)))
    L = s_in + m_out / (tau + lam)
    U = s_in + m_out / lam

    return {
        "r_A": r_A,
        "r_G": r_G,
        "s_in": s_in,
        "L": L,
        "U": U,
        "m_in": m_in,
        "m_out": m_out,
        "sketch_bytes": sketch_bytes(r_A, r_G, d_in, d_out),
    }
