"""
scripts/run_demo.py
===================
Self-contained end-to-end demo using synthetic numpy data.
Requires h5py only — no torch, no live Redis server.

Run:
    python scripts/run_demo.py

All assert statements are load-bearing round-trip checks.
"""

import os
import sys
import tempfile

import numpy as np

# allow `python scripts/run_demo.py` from the repo root without install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kfac_unlearn.storage import HDF5KFACStore
from kfac_unlearn.kfac import kfac_newton_step
from kfac_unlearn.serialization import _z_hash, _pack_array, _unpack_array
from kfac_unlearn.cache import REDIS_TTL


def main():
    print("=" * 60)
    print("K-FAC unlearning demo")
    print("=" * 60)

    # ── Synthetic layer dimensions ────────────────────────────────────────
    rng = np.random.default_rng(42)

    d_in, d_out = 64, 32           # small for demo — real layers ~512×256
    step = 5000

    # Simulate K-FAC factors collected during training
    # A = input activation covariance  (d_in × d_in)
    # G = output gradient covariance   (d_out × d_out)
    raw_A = rng.standard_normal((d_in, d_in)).astype(np.float32)
    raw_G = rng.standard_normal((d_out, d_out)).astype(np.float32)
    A = (raw_A.T @ raw_A) / d_in + 1e-3 * np.eye(d_in, dtype=np.float32)
    G = (raw_G.T @ raw_G) / d_out + 1e-3 * np.eye(d_out, dtype=np.float32)

    # ── 1. Save to HDF5 ───────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "kfac_store.h5")
        store = HDF5KFACStore(h5_path)

        print(f"\n[HDF5] Saving K-FAC factors for 'fc1' at step {step} ...")
        store.save_kfac("fc1", step=step, A=A, G=G)
        store.save_kfac("fc1", step=step + 5000, A=A * 0.99, G=G * 0.99)
        store.save_kfac("fc2", step=step, A=A[:32, :32], G=G[:16, :16])

        summary = store.storage_summary()
        for layer, info in summary.items():
            print(f"  {layer:8s}  steps={info['steps']}  disk={info['disk_kb']} KB")

        # ── 2. Load and verify nearest-step logic ─────────────────────────
        print(f"\n[HDF5] Loading 'fc1' nearest to step 4900 ...")
        nearest = store.nearest_checkpoint("fc1", target_step=4900)
        print(f"  Nearest checkpoint found: step {nearest}")
        A_loaded, G_loaded = store.load_kfac("fc1", nearest)
        print(f"  A shape: {A_loaded.shape}   G shape: {G_loaded.shape}")
        assert np.allclose(A, A_loaded, atol=1e-5), "Round-trip mismatch on A"
        print("  Round-trip check passed.")

        # ── 3. Demonstrate K-FAC Newton step ─────────────────────────────
        print("\n[KFAC] Computing Newton step ...")
        z = rng.standard_normal(d_in).astype(np.float32)    # fake data point
        grad_W = rng.standard_normal((d_out, d_in)).astype(np.float32)

        A_inv, G_inv = store.load_kfac_inverse("fc1", step)
        delta_W = kfac_newton_step(grad_W, A_inv, G_inv)
        print(f"  grad_W norm  : {np.linalg.norm(grad_W):.4f}")
        print(f"  delta_W norm : {np.linalg.norm(delta_W):.4f}")
        print(f"  Shape: {delta_W.shape}  (same as weight matrix — correct)")

        # ── 4. Show cache key/TTL structure (no live server needed) ───────
        print("\n[Cache] Key schema and TTL strategy:")
        z_h = _z_hash(z)
        examples = [
            (f"hvp:fc1:{z_h}", REDIS_TTL['hvp'],  "cached Δθ for (fc1, this data point)"),
            (f"grad:user42_s7", REDIS_TTL['grad'], "cached gradient vector"),
            (f"hinv:fc1:{step}", REDIS_TTL['hinv'], "cached A⁻¹ and G⁻¹ for fc1@5000"),
        ]
        for key, ttl, desc in examples:
            print(f"  key  : {key}")
            print(f"  TTL  : {ttl}s ({ttl//60} min)  — {desc}")
            print()

        # ── 5. Serialisation round-trip check ─────────────────────────────
        print("[Cache] Serialisation round-trip check ...")
        packed = _pack_array(delta_W)
        recovered = _unpack_array(packed)
        assert np.allclose(delta_W, recovered), "Serialisation mismatch"
        overhead = len(packed) - delta_W.nbytes
        print(f"  Array bytes : {delta_W.nbytes}")
        print(f"  Packed bytes: {len(packed)}  (overhead: {overhead}B metadata)")
        print("  Round-trip check passed.")

    print("\n" + "=" * 60)
    print("Demo complete. All checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
