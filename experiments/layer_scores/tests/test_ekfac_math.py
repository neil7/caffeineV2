"""
Correctness tests T1, T2, T3, T4, T6 (pure numpy, no torch) from the EK-FAC
layer-scores spec appended to CLAUDE.md. Run with:

    pytest experiments/layer_scores/tests/ -q

(not auto-discovered by the root `pytest` invocation, whose `testpaths` is
`["tests"]` — run this path explicitly.)
"""

import numpy as np
import pytest

from experiments.layer_scores.ekfac_math import (
    dense_H,
    eigh_factors,
    full_step,
    ihvp_factored,
    layer_score,
    rank_r_sketch,
    rotate_to_eigenbasis,
    vec,
)

RNG = np.random.default_rng(42)


def _random_spd(n, rng):
    M = rng.standard_normal((n, n)).astype(np.float64)
    return M @ M.T + n * np.eye(n)


def test_T1_ihvp_vs_explicit_fc_size():
    """fc layer: d_out=10, d_in=513 (512 features + 1 bias column)."""
    d_out, d_in = 10, 513
    rng = np.random.default_rng(1)
    A = _random_spd(d_in, rng)
    G = _random_spd(d_out, rng)
    Q_A, a, Q_G, gamma = eigh_factors(A, G)

    Lambda = rng.uniform(0.1, 2.0, size=(d_out, d_in)).astype(np.float64)
    lam = 0.1 * Lambda.mean()
    V = rng.standard_normal((d_out, d_in))

    factored = ihvp_factored(V, Q_A, Q_G, Lambda, lam)

    H = dense_H(Q_A, Q_G, Lambda, lam)
    x = np.linalg.solve(H, vec(V.astype(np.float64)))
    explicit = x.reshape((d_out, d_in), order="F")

    rel_err = np.linalg.norm(factored - explicit) / np.linalg.norm(explicit)
    assert rel_err < 1e-6, f"T1 failed: rel_err={rel_err}"


def test_T2_score_identity():
    rng = np.random.default_rng(2)
    d_out, d_in = 6, 9
    A = _random_spd(d_in, rng)
    G = _random_spd(d_out, rng)
    Q_A, a, Q_G, gamma = eigh_factors(A, G)
    Lambda = rng.uniform(0.1, 2.0, size=(d_out, d_in))
    lam = 0.1 * Lambda.mean()
    V = rng.standard_normal((d_out, d_in))

    C = rotate_to_eigenbasis(V, Q_A, Q_G)
    s = layer_score(C, Lambda, lam)

    ihvp = ihvp_factored(V, Q_A, Q_G, Lambda, lam)
    lhs = float(np.sum(V * ihvp))  # <vec V, vec IHVP(V)>

    assert np.isclose(lhs, s, rtol=1e-6), f"T2 failed: lhs={lhs} s={s}"


def test_T3_additivity_lemma():
    rng = np.random.default_rng(3)
    layer_shapes = [(4, 5), (3, 7), (6, 4)]
    layers = []
    for d_out, d_in in layer_shapes:
        A = _random_spd(d_in, rng)
        G = _random_spd(d_out, rng)
        Q_A, a, Q_G, gamma = eigh_factors(A, G)
        Lambda = rng.uniform(0.1, 2.0, size=(d_out, d_in))
        lam = 0.1 * Lambda.mean()
        V = rng.standard_normal((d_out, d_in))
        C = rotate_to_eigenbasis(V, Q_A, Q_G)
        s = layer_score(C, Lambda, lam)
        delta_star = full_step(C, Q_A, Q_G, Lambda, lam)
        H = dense_H(Q_A, Q_G, Lambda, lam)
        layers.append(dict(Q_A=Q_A, Q_G=Q_G, Lambda=Lambda, lam=lam, s=s,
                            delta_star=delta_star, H=H, shape=(d_out, d_in)))

    for trial in range(10):
        S = set(i for i in range(len(layers)) if rng.random() < 0.5)
        total_sq_norm = 0.0
        expected = sum(layers[l]["s"] for l in range(len(layers)) if l not in S)
        for l, layer in enumerate(layers):
            delta_S_l = layer["delta_star"] if l in S else np.zeros(layer["shape"])
            diff = vec((delta_S_l - layer["delta_star"]).astype(np.float64))
            total_sq_norm += float(diff @ layer["H"] @ diff)
        rel_err = abs(total_sq_norm - expected) / max(expected, 1e-12)
        assert rel_err < 1e-5, f"T3 failed trial {trial}: rel_err={rel_err}"


def test_T4_bounds():
    rng = np.random.default_rng(4)
    d_out, d_in = 8, 11
    A = _random_spd(d_in, rng)
    G = _random_spd(d_out, rng)
    Q_A, a, Q_G, gamma = eigh_factors(A, G)
    Lambda = rng.uniform(0.1, 2.0, size=(d_out, d_in))
    lam = 0.1 * Lambda.mean()

    for _ in range(5):
        V = rng.standard_normal((d_out, d_in))
        C = rotate_to_eigenbasis(V, Q_A, Q_G)
        s = layer_score(C, Lambda, lam)
        for r in [0, 1, 3, 5, 8, 11, 50]:
            res = rank_r_sketch(V, Q_A, Q_G, Lambda, lam, r)
            assert res["L"] <= s + 1e-8, f"T4 failed: L={res['L']} > s={s} (r={r})"
            assert s <= res["U"] + 1e-8, f"T4 failed: s={s} > U={res['U']} (r={r})"
        full_r = max(d_in, d_out)
        res_full = rank_r_sketch(V, Q_A, Q_G, Lambda, lam, full_r)
        assert np.isclose(res_full["L"], s, rtol=1e-5)
        assert np.isclose(res_full["U"], s, rtol=1e-5)


def test_T6_eigenbasis_descending_orthonormal():
    rng = np.random.default_rng(6)
    d_out, d_in = 7, 13
    A = _random_spd(d_in, rng)
    G = _random_spd(d_out, rng)
    Q_A, a, Q_G, gamma = eigh_factors(A, G)

    assert np.all(np.diff(a) <= 1e-9), "a not descending"
    assert np.all(np.diff(gamma) <= 1e-9), "gamma not descending"
    assert np.linalg.norm(Q_A.T @ Q_A - np.eye(d_in)) < 1e-5
    assert np.linalg.norm(Q_G.T @ Q_G - np.eye(d_out)) < 1e-5
