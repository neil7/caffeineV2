"""
kfac_unlearn.serialization
===========================
Stable, pickle-free serialisation for numpy arrays going into a cache backend,
plus a content hash used to key cache entries by data point.

`_pack_array` / `_unpack_array` embed shape + dtype in a JSON header so arrays
round-trip safely across Python and numpy versions (pickle does not). The
43-byte overhead per array is negligible.
"""

import hashlib
import json

import numpy as np


def _z_hash(z: np.ndarray) -> str:
    """Stable hash of a numpy array — used as part of the cache key."""
    return hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()[:16]


def _pack_array(arr: np.ndarray) -> bytes:
    """Serialise a numpy array to bytes, embedding shape + dtype metadata."""
    meta = json.dumps({"shape": list(arr.shape), "dtype": str(arr.dtype)}).encode()
    meta_len = len(meta).to_bytes(4, "little")
    return meta_len + meta + np.ascontiguousarray(arr).tobytes()


def _unpack_array(data: bytes) -> np.ndarray:
    """Deserialise bytes produced by _pack_array."""
    meta_len = int.from_bytes(data[:4], "little")
    meta = json.loads(data[4: 4 + meta_len])
    raw = data[4 + meta_len:]
    return np.frombuffer(raw, dtype=meta["dtype"]).reshape(meta["shape"]).copy()
