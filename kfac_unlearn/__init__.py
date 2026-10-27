"""
kfac_unlearn
============
Efficient machine unlearning via K-FAC Hessian approximation and two-tier
(HDF5 + cache) paged storage.

Public API
----------
    HDF5KFACStore                      — persistent K-FAC factor store
    HVPCache, InMemoryCache,
    RedisHVPCache, make_cache          — cache backends
    kfac_newton_step                   — core Newton-step math (pure numpy)
    KFACUnlearner, UnlearningResult    — full unlearning pipeline
"""

__version__ = "0.1.0"

from kfac_unlearn.storage import HDF5KFACStore
from kfac_unlearn.cache import (
    HVPCache,
    InMemoryCache,
    RedisHVPCache,
    REDIS_TTL,
    make_cache,
)
from kfac_unlearn.kfac import kfac_newton_step
from kfac_unlearn.unlearner import KFACUnlearner, UnlearningResult
from kfac_unlearn.serialization import _z_hash, _pack_array, _unpack_array

__all__ = [
    "HDF5KFACStore",
    "HVPCache",
    "InMemoryCache",
    "RedisHVPCache",
    "REDIS_TTL",
    "make_cache",
    "kfac_newton_step",
    "KFACUnlearner",
    "UnlearningResult",
    "_z_hash",
    "_pack_array",
    "_unpack_array",
]
