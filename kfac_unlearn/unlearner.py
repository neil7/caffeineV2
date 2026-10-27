"""
kfac_unlearn.unlearner
======================
Ties HDF5KFACStore and an HVPCache backend into a complete machine-unlearning
pipeline.

For each deletion request:
  1. Redis/InMemory: get hvp:{layer}:{z_hash}   → hit: apply Δθ immediately
  2. Redis/InMemory: get hinv:{layer}:{step}     → hit: skip inversion
  3. HDF5:  load_kfac(layer, nearest_step) → invert with damping λI, cache hinv
  4. Compute: delta_W = −G⁻¹ @ grad_W @ A⁻ᵀ, cache hvp
  5. Apply:  model_weights[layer] += delta_W
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from kfac_unlearn.cache import HVPCache
from kfac_unlearn.kfac import kfac_newton_step
from kfac_unlearn.storage import HDF5KFACStore


@dataclass
class UnlearningResult:
    """Returned by KFACUnlearner.unlearn — carries timing and cache metadata."""
    z_id: str
    layers_updated: list
    total_time_ms: float
    cache_hits: Dict[str, bool] = field(default_factory=dict)
    delta_norms: Dict[str, float] = field(default_factory=dict)


class KFACUnlearner:
    """
    Complete machine-unlearning pipeline over an HDF5 factor store and an
    HVPCache backend.

    Usage
    -----
    store     = HDF5KFACStore("kfac_factors.h5")
    cache     = InMemoryCache()           # or RedisHVPCache(host="localhost")
    unlearner = KFACUnlearner(store, cache)

    result = unlearner.unlearn(
        model_weights=model_weights,     # dict of {layer_name: np.ndarray}
        grad_W_per_layer=grads,          # dict of {layer_name: np.ndarray}
        z=data_point_to_forget,
        z_id="user_42_sample_7",
        checkpoint_step=5000,
    )
    """

    def __init__(
        self,
        store: HDF5KFACStore,
        cache: HVPCache,
        damping: float = 1e-4,
    ):
        self.store = store
        self.cache = cache
        self.damping = damping

    def unlearn(
        self,
        model_weights: Dict[str, np.ndarray],
        grad_W_per_layer: Dict[str, np.ndarray],
        z: np.ndarray,
        z_id: str,
        checkpoint_step: int,
    ) -> UnlearningResult:
        """
        Remove the influence of data point z from model_weights.

        Parameters
        ----------
        model_weights      : {layer_name: weight_matrix}  — modified in place
        grad_W_per_layer   : {layer_name: grad_matrix}    — ∇W for point z
        z                  : the raw data point (used as cache key)
        z_id               : human-readable ID for logging
        checkpoint_step    : which stored checkpoint to use for H factors

        Returns
        -------
        UnlearningResult with timing and cache hit information
        """
        t0 = time.perf_counter()
        layers_updated = []
        cache_hits = {}
        delta_norms = {}

        for layer_name, grad_W in grad_W_per_layer.items():

            # ── Step 1: try full HVP cache ────────────────────────────────
            delta_W = self.cache.get_hvp(layer_name, z)
            if delta_W is not None:
                cache_hits[layer_name] = "hvp_hit"
            else:
                cache_hits[layer_name] = "hvp_miss"

                # ── Step 2: try cached inverse factors ────────────────────
                hinv = self.cache.get_hinv(layer_name, checkpoint_step)
                if hinv is not None:
                    A_inv, G_inv = hinv
                    cache_hits[layer_name] = "hinv_hit"
                else:
                    # ── Step 3: load from HDF5 and invert ─────────────────
                    nearest = self.store.nearest_checkpoint(
                        layer_name, checkpoint_step
                    )
                    if nearest != checkpoint_step:
                        warnings.warn(
                            f"Step {checkpoint_step} not found for layer "
                            f"'{layer_name}', using nearest: {nearest}"
                        )
                    A_inv, G_inv = self.store.load_kfac_inverse(
                        layer_name, nearest, damping=self.damping
                    )
                    # cache the inverses — most expensive thing to recompute
                    self.cache.set_hinv(layer_name, nearest, A_inv, G_inv)

                # ── Step 4: compute Δθ for this layer ─────────────────────
                delta_W = kfac_newton_step(grad_W, A_inv, G_inv)

                # cache the full result so nearby deletions skip all of this
                self.cache.set_hvp(layer_name, z, delta_W)

            # ── Step 5: apply update ──────────────────────────────────────
            if layer_name in model_weights:
                model_weights[layer_name] += delta_W
                layers_updated.append(layer_name)
                delta_norms[layer_name] = float(np.linalg.norm(delta_W))

        total_ms = (time.perf_counter() - t0) * 1000

        return UnlearningResult(
            z_id=z_id,
            layers_updated=layers_updated,
            total_time_ms=round(total_ms, 2),
            cache_hits=cache_hits,
            delta_norms={k: round(v, 6) for k, v in delta_norms.items()},
        )
