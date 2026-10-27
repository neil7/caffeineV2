"""Serialisation round-trip + InMemoryCache behaviour (Redis-parity semantics)."""

import numpy as np
import pytest

from kfac_unlearn import InMemoryCache, _pack_array, _unpack_array
from kfac_unlearn.cache import REDIS_TTL


def test_pack_unpack_roundtrip(rng):
    arr = rng.standard_normal((8, 5)).astype(np.float32)
    rec = _unpack_array(_pack_array(arr))
    assert np.array_equal(arr, rec)
    assert rec.dtype == arr.dtype and rec.shape == arr.shape
    # overhead = 4-byte length prefix + JSON meta header (varies with shape/dtype)
    overhead = len(_pack_array(arr)) - arr.nbytes
    assert 30 < overhead < 60


def test_hvp_set_get(inmem_cache, rng):
    z = rng.standard_normal(16).astype(np.float32)
    dW = rng.standard_normal((8, 16)).astype(np.float32)
    assert inmem_cache.get_hvp("conv1", z) is None          # miss
    inmem_cache.set_hvp("conv1", z, dW)
    assert np.allclose(inmem_cache.get_hvp("conv1", z), dW)  # hit
    # different z -> miss
    assert inmem_cache.get_hvp("conv1", z + 1) is None


def test_hinv_set_get(inmem_cache, spd_factors):
    A, G = spd_factors
    assert inmem_cache.get_hinv("fc", 5000) is None
    inmem_cache.set_hinv("fc", 5000, A, G)
    Ai, Gi = inmem_cache.get_hinv("fc", 5000)
    assert np.allclose(Ai, A) and np.allclose(Gi, G)


def test_invalidate_and_stats(inmem_cache, rng, spd_factors):
    z = rng.standard_normal(4).astype(np.float32)
    A, G = spd_factors
    inmem_cache.set_hvp("conv1", z, np.ones((2, 2), np.float32))
    inmem_cache.set_hinv("conv1", 1, A, G)
    inmem_cache.set_grad("p7", np.ones(3, np.float32))
    stats = inmem_cache.cache_stats()
    assert stats == {"hvp_keys": 1, "grad_keys": 1, "hinv_keys": 1}
    n = inmem_cache.invalidate_layer("conv1")
    assert n == 2                                            # hvp + hinv, not grad
    assert inmem_cache.cache_stats()["grad_keys"] == 1


def test_ttl_expiry(monkeypatch):
    """Entry past its TTL reads back as a miss (monotonic-clock advance)."""
    cache = InMemoryCache(ttl_overrides={"grad": 10})
    t = {"v": 1000.0}
    monkeypatch.setattr(cache, "_now", lambda: t["v"])
    cache.set_grad("p", np.ones(2, np.float32))
    assert cache.get_grad("p") is not None
    t["v"] += 11                                            # past 10s TTL
    assert cache.get_grad("p") is None


def test_default_ttls():
    assert REDIS_TTL == {"hvp": 3600, "grad": 1800, "hinv": 7200}
