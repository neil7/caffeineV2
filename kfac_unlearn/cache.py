"""
kfac_unlearn.cache
==================
Fast in-memory cache for Hessian-vector products, gradients, and inverted
K-FAC factors. Mirrors vLLM's block table — a lookup layer that lets you skip
expensive HDF5 reads and matrix inversions for repeat / nearby deletion requests.

Backends behind a common `HVPCache` interface:
  - InMemoryCache : dependency-free dict + monotonic-clock TTL. **Default** — a
                    reviewer can run every experiment with zero external services.
  - RedisHVPCache : process-shared Redis backend for multi-process deployments.

Key schema (identical across backends):
  hvp:{layer_name}:{z_hash}    TTL 3600s  (1h)   — full Δθ for (layer, data point)
  grad:{z_id}                  TTL 1800s (30m)   — raw gradient vector
  hinv:{layer_name}:{step}     TTL 7200s  (2h)   — A⁻¹ and G⁻¹ (most expensive)

The `hinv` TTL is longest because A⁻¹, G⁻¹ are reusable across many different
data-point deletions at the same checkpoint step.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np

from kfac_unlearn.serialization import _pack_array, _unpack_array, _z_hash

try:
    import redis
except ImportError:  # redis is an optional extra
    redis = None


REDIS_TTL = {
    "hvp": 3600,
    "grad": 1800,
    "hinv": 7200,
}


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────────────────────────────────────

class HVPCache(ABC):
    """
    Cache contract used by KFACUnlearner. Any backend implementing these
    methods is a drop-in replacement.
    """

    @abstractmethod
    def get_hvp(self, layer_name: str, z: np.ndarray) -> Optional[np.ndarray]: ...

    @abstractmethod
    def set_hvp(self, layer_name: str, z: np.ndarray, delta_theta: np.ndarray) -> None: ...

    @abstractmethod
    def get_grad(self, z_id: str) -> Optional[np.ndarray]: ...

    @abstractmethod
    def set_grad(self, z_id: str, grad: np.ndarray) -> None: ...

    @abstractmethod
    def get_hinv(
        self, layer_name: str, step: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]: ...

    @abstractmethod
    def set_hinv(
        self, layer_name: str, step: int, A_inv: np.ndarray, G_inv: np.ndarray
    ) -> None: ...

    @abstractmethod
    def invalidate_layer(self, layer_name: str) -> int: ...

    @abstractmethod
    def cache_stats(self) -> Dict: ...


# ─────────────────────────────────────────────────────────────────────────────
# In-memory backend (default, dependency-free)
# ─────────────────────────────────────────────────────────────────────────────

class InMemoryCache(HVPCache):
    """
    Dependency-free fallback. A dict with monotonic-clock TTL expiry that
    mirrors Redis semantics, so behaviour is identical to RedisHVPCache in tests.

    Stores numpy arrays directly (no pack/unpack needed in-process) but reuses
    the same `_z_hash` and key strings for parity with the Redis backend.
    """

    def __init__(self, ttl_overrides: Optional[Dict[str, int]] = None):
        self._store: Dict[str, Tuple[object, float]] = {}  # key -> (value, expiry_ts)
        self._ttl = {**REDIS_TTL, **(ttl_overrides or {})}

    # ── internal expiry ──────────────────────────────────────────────────

    def _now(self) -> float:
        return time.monotonic()

    def _get(self, key: str):
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if self._now() >= expiry:
            del self._store[key]
            return None
        return value

    def _set(self, key: str, value: object, ttl: int) -> None:
        self._store[key] = (value, self._now() + ttl)

    def _purge_expired(self) -> None:
        now = self._now()
        for k in [k for k, (_, exp) in self._store.items() if now >= exp]:
            del self._store[k]

    # ── HVP results ───────────────────────────────────────────────────────

    def get_hvp(self, layer_name: str, z: np.ndarray) -> Optional[np.ndarray]:
        return self._get(f"hvp:{layer_name}:{_z_hash(z)}")

    def set_hvp(self, layer_name: str, z: np.ndarray, delta_theta: np.ndarray) -> None:
        self._set(f"hvp:{layer_name}:{_z_hash(z)}", np.asarray(delta_theta), self._ttl["hvp"])

    # ── Per-point gradients ───────────────────────────────────────────────

    def get_grad(self, z_id: str) -> Optional[np.ndarray]:
        return self._get(f"grad:{z_id}")

    def set_grad(self, z_id: str, grad: np.ndarray) -> None:
        self._set(f"grad:{z_id}", np.asarray(grad), self._ttl["grad"])

    # ── Inverted K-FAC factors ────────────────────────────────────────────

    def get_hinv(
        self, layer_name: str, step: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        val = self._get(f"hinv:{layer_name}:{step}")
        return None if val is None else (val[0], val[1])

    def set_hinv(
        self, layer_name: str, step: int, A_inv: np.ndarray, G_inv: np.ndarray
    ) -> None:
        self._set(
            f"hinv:{layer_name}:{step}",
            (np.asarray(A_inv), np.asarray(G_inv)),
            self._ttl["hinv"],
        )

    # ── Utility ───────────────────────────────────────────────────────────

    def invalidate_layer(self, layer_name: str) -> int:
        prefixes = (f"hvp:{layer_name}:", f"hinv:{layer_name}:")
        keys = [k for k in self._store if k.startswith(prefixes)]
        for k in keys:
            del self._store[k]
        return len(keys)

    def cache_stats(self) -> Dict:
        self._purge_expired()
        return {
            "hvp_keys": sum(k.startswith("hvp:") for k in self._store),
            "grad_keys": sum(k.startswith("grad:") for k in self._store),
            "hinv_keys": sum(k.startswith("hinv:") for k in self._store),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Redis backend (optional extra)
# ─────────────────────────────────────────────────────────────────────────────

class RedisHVPCache(HVPCache):
    """
    Fast in-memory cache for Hessian-vector products and gradients backed by a
    live Redis server. Process-shared, so multiple unlearning workers hit the
    same cache. Requires `pip install kfac-unlearn[redis]`.

    Usage
    -----
    cache = RedisHVPCache(host="localhost", port=6379)
    cache.set_hvp("conv1", z_array, delta_theta_conv1)
    result = cache.get_hvp("conv1", z_array)   # None on cache miss
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl_overrides: Optional[Dict[str, int]] = None,
    ):
        if redis is None:
            raise ImportError(
                "RedisHVPCache requires the redis client. "
                "Install it with: pip install kfac-unlearn[redis]"
            )
        self._client = redis.Redis(
            host=host, port=port, db=db, password=password,
            decode_responses=False,   # we store raw bytes
        )
        self._ttl = {**REDIS_TTL, **(ttl_overrides or {})}

    # ── HVP (H⁻¹ · gradient) results ─────────────────────────────────────

    def get_hvp(self, layer_name: str, z: np.ndarray) -> Optional[np.ndarray]:
        """Retrieve a cached Δθ vector for (layer, data point z). None on miss."""
        key = f"hvp:{layer_name}:{_z_hash(z)}"
        raw = self._client.get(key)
        return _unpack_array(raw) if raw is not None else None

    def set_hvp(
        self, layer_name: str, z: np.ndarray, delta_theta: np.ndarray
    ) -> None:
        """Cache Δθ for (layer, z) with TTL."""
        key = f"hvp:{layer_name}:{_z_hash(z)}"
        self._client.setex(key, self._ttl["hvp"], _pack_array(delta_theta))

    # ── Per-point gradients ───────────────────────────────────────────────

    def get_grad(self, z_id: str) -> Optional[np.ndarray]:
        """Retrieve a cached gradient vector for data point z_id."""
        raw = self._client.get(f"grad:{z_id}")
        return _unpack_array(raw) if raw is not None else None

    def set_grad(self, z_id: str, grad: np.ndarray) -> None:
        """Cache gradient vector for data point z_id."""
        self._client.setex(f"grad:{z_id}", self._ttl["grad"], _pack_array(grad))

    # ── Inverted K-FAC factors ────────────────────────────────────────────

    def get_hinv(
        self, layer_name: str, step: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Retrieve cached (A⁻¹, G⁻¹) for a layer + checkpoint step.

        Matrix inversion is O(d³) so caching the result across multiple
        unlearning requests for the same checkpoint is a significant win.
        """
        key = f"hinv:{layer_name}:{step}"
        raw = self._client.get(key)
        if raw is None:
            return None
        # stored as two packed arrays concatenated with a 4-byte split marker
        split = int.from_bytes(raw[:4], "little")
        A_inv = _unpack_array(raw[4: 4 + split])
        G_inv = _unpack_array(raw[4 + split:])
        return A_inv, G_inv

    def set_hinv(
        self, layer_name: str, step: int, A_inv: np.ndarray, G_inv: np.ndarray
    ) -> None:
        """Cache (A⁻¹, G⁻¹) for a layer + checkpoint step."""
        key = f"hinv:{layer_name}:{step}"
        packed_A = _pack_array(A_inv)
        packed_G = _pack_array(G_inv)
        split_marker = len(packed_A).to_bytes(4, "little")
        self._client.setex(
            key, self._ttl["hinv"], split_marker + packed_A + packed_G
        )

    # ── Utility ───────────────────────────────────────────────────────────

    def invalidate_layer(self, layer_name: str) -> int:
        """Delete all cache entries for a layer. Returns number of keys deleted."""
        patterns = [
            f"hvp:{layer_name}:*",
            f"hinv:{layer_name}:*",
        ]
        deleted = 0
        for pat in patterns:
            keys = self._client.keys(pat)
            if keys:
                deleted += self._client.delete(*keys)
        return deleted

    def cache_stats(self) -> Dict:
        """Return counts of cached keys by type."""
        return {
            "hvp_keys": len(self._client.keys("hvp:*")),
            "grad_keys": len(self._client.keys("grad:*")),
            "hinv_keys": len(self._client.keys("hinv:*")),
        }

    @property
    def client(self):
        """Expose raw Redis client for advanced use."""
        return self._client


# ─────────────────────────────────────────────────────────────────────────────
# Backend factory
# ─────────────────────────────────────────────────────────────────────────────

def make_cache(cfg: Optional[Dict] = None) -> HVPCache:
    """
    Construct a cache backend from a config dict (e.g. parsed from
    configs/storage.yaml). Defaults to InMemoryCache.

    cfg example:
        {"backend": "redis", "host": "localhost", "port": 6379,
         "ttl_overrides": {"hvp": 7200}}
    """
    cfg = cfg or {}
    backend = cfg.get("backend", "inmemory").lower()
    ttl_overrides = cfg.get("ttl_overrides")

    if backend in ("inmemory", "memory", "local"):
        return InMemoryCache(ttl_overrides=ttl_overrides)
    if backend == "redis":
        return RedisHVPCache(
            host=cfg.get("host", "localhost"),
            port=cfg.get("port", 6379),
            db=cfg.get("db", 0),
            password=cfg.get("password"),
            ttl_overrides=ttl_overrides,
        )
    raise ValueError(f"Unknown cache backend: {backend!r}")
