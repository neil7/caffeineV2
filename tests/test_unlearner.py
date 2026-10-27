"""Full KFACUnlearner pipeline on synthetic data + cache-hit accounting."""

import numpy as np

from kfac_unlearn import HDF5KFACStore, InMemoryCache, KFACUnlearner, kfac_newton_step


def _setup(store, spd_factors, step=5000):
    A, G = spd_factors
    store.save_kfac("fc", step=step, A=A, G=G)
    return A, G


def test_unlearn_applies_correct_delta(store, inmem_cache, spd_factors, rng):
    A, G = _setup(store, spd_factors)
    d_out, d_in = G.shape[0], A.shape[0]
    grad_W = rng.standard_normal((d_out, d_in)).astype(np.float32)
    W0 = rng.standard_normal((d_out, d_in)).astype(np.float32)
    z = rng.standard_normal(d_in).astype(np.float32)

    unlearner = KFACUnlearner(store, inmem_cache, damping=1e-4)
    weights = {"fc": W0.copy()}
    res = unlearner.unlearn(weights, {"fc": grad_W}, z, "z0", checkpoint_step=5000)

    A_inv, G_inv = store.load_kfac_inverse("fc", 5000, damping=1e-4)
    expected = W0 + kfac_newton_step(grad_W, A_inv, G_inv)
    assert np.allclose(weights["fc"], expected, atol=1e-4)
    assert res.layers_updated == ["fc"]
    assert res.cache_hits["fc"] == "hvp_miss"          # cold path first time


def test_cache_hit_second_request(store, inmem_cache, spd_factors, rng):
    A, G = _setup(store, spd_factors)
    d_out, d_in = G.shape[0], A.shape[0]
    grad_W = rng.standard_normal((d_out, d_in)).astype(np.float32)
    z = rng.standard_normal(d_in).astype(np.float32)
    unlearner = KFACUnlearner(store, inmem_cache, damping=1e-4)

    w1 = {"fc": np.zeros((d_out, d_in), np.float32)}
    unlearner.unlearn(w1, {"fc": grad_W}, z, "z0", 5000)
    # second identical request → hvp cache hit
    w2 = {"fc": np.zeros((d_out, d_in), np.float32)}
    res2 = unlearner.unlearn(w2, {"fc": grad_W}, z, "z0", 5000)
    assert res2.cache_hits["fc"] == "hvp_hit"
    assert np.allclose(w1["fc"], w2["fc"])


def test_hinv_hit_different_point(store, inmem_cache, spd_factors, rng):
    """Different data point at same checkpoint reuses cached inverse (hinv_hit)."""
    A, G = _setup(store, spd_factors)
    d_out, d_in = G.shape[0], A.shape[0]
    unlearner = KFACUnlearner(store, inmem_cache, damping=1e-4)

    z1 = rng.standard_normal(d_in).astype(np.float32)
    z2 = rng.standard_normal(d_in).astype(np.float32)
    g1 = rng.standard_normal((d_out, d_in)).astype(np.float32)
    g2 = rng.standard_normal((d_out, d_in)).astype(np.float32)

    unlearner.unlearn({"fc": np.zeros((d_out, d_in), np.float32)}, {"fc": g1}, z1, "a", 5000)
    res = unlearner.unlearn({"fc": np.zeros((d_out, d_in), np.float32)}, {"fc": g2}, z2, "b", 5000)
    assert res.cache_hits["fc"] == "hinv_hit"
