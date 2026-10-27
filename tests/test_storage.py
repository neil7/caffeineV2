"""HDF5KFACStore round-trip, nearest_checkpoint, overwrite guard, metadata."""

import numpy as np
import pytest


def test_save_load_roundtrip(store, spd_factors):
    A, G = spd_factors
    store.save_kfac("conv1", step=5000, A=A, G=G)
    A2, G2 = store.load_kfac("conv1", 5000)
    assert np.allclose(A, A2, atol=1e-5)
    assert np.allclose(G, G2, atol=1e-5)
    assert A2.dtype == np.float32 and G2.dtype == np.float32


def test_overwrite_guard(store, spd_factors):
    A, G = spd_factors
    store.save_kfac("fc", step=100, A=A, G=G)
    with pytest.raises(ValueError):
        store.save_kfac("fc", step=100, A=A, G=G)        # no overwrite
    store.save_kfac("fc", step=100, A=A * 2, G=G, overwrite=True)
    A2, _ = store.load_kfac("fc", 100)
    assert np.allclose(A2, A * 2, atol=1e-5)


def test_nearest_checkpoint(store, spd_factors):
    A, G = spd_factors
    for s in (1000, 5000, 9000):
        store.save_kfac("layer1.0", step=s, A=A, G=G)
    assert store.nearest_checkpoint("layer1.0", 4800) == 5000
    assert store.nearest_checkpoint("layer1.0", 8000) == 9000
    assert sorted(store.list_steps("layer1.0")) == [1000, 5000, 9000]


def test_missing_layer_raises(store):
    with pytest.raises(KeyError):
        store.load_kfac("nope", 1)
    with pytest.raises(KeyError):
        store.nearest_checkpoint("nope", 1)


def test_inverse_and_summary(store, spd_factors):
    A, G = spd_factors
    store.save_kfac("fc", step=1, A=A, G=G)
    A_inv, G_inv = store.load_kfac_inverse("fc", 1, damping=1e-4)
    # A_inv @ (A + λI) ≈ I
    lam = 1e-4
    assert np.allclose(A_inv @ (A + lam * np.eye(A.shape[0], dtype=np.float32)),
                       np.eye(A.shape[0]), atol=1e-3)
    summary = store.storage_summary()
    assert "fc" in summary and summary["fc"]["steps"] == [1]
    assert store.list_layers() == ["fc"]
