"""Shared pytest fixtures: tmp HDF5 path, synthetic SPD K-FAC factors, caches."""

import numpy as np
import pytest

from kfac_unlearn import HDF5KFACStore, InMemoryCache


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def spd_factors(rng):
    """Symmetric positive-definite A (d_in) and G (d_out), float32."""
    d_in, d_out = 16, 8
    rA = rng.standard_normal((d_in, d_in)).astype(np.float32)
    rG = rng.standard_normal((d_out, d_out)).astype(np.float32)
    A = (rA.T @ rA) / d_in + 1e-3 * np.eye(d_in, dtype=np.float32)
    G = (rG.T @ rG) / d_out + 1e-3 * np.eye(d_out, dtype=np.float32)
    return A, G


@pytest.fixture
def store(tmp_path):
    return HDF5KFACStore(str(tmp_path / "kfac.h5"))


@pytest.fixture
def inmem_cache():
    return InMemoryCache()
