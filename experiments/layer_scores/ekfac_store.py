"""
experiments.layer_scores.ekfac_store
======================================
Sibling to kfac_unlearn.storage.HDF5KFACStore (not modified — its save_kfac/
load_kfac API is hardcoded to exactly an (A, G) pair, no generic "arbitrary
named array" method). EK-FAC needs Q_A, Q_G, Lambda, a, gamma per layer per
checkpoint step, so this module is its own small store, mirroring the same
key-schema conventions (`_step_key`, chunking) but writing into its own file.

Key schema: /ckpt_{step}/{layer}/{Q_A, Q_G, Lambda, a, gamma}
"""

from typing import Dict, List, Tuple

import h5py
import numpy as np


def _step_key(step: int) -> str:
    return f"ckpt_{step}"


def _chunk_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(min(d, 256) for d in shape)


class EKFACStore:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def save_eigen(
        self,
        layer_name: str,
        step: int,
        Q_A: np.ndarray,
        Q_G: np.ndarray,
        Lambda: np.ndarray,
        a: np.ndarray,
        gamma: np.ndarray,
        overwrite: bool = False,
    ) -> None:
        if "/" in layer_name:
            raise ValueError(f"layer_name must not contain '/': {layer_name!r}")
        arrays = {"Q_A": Q_A, "Q_G": Q_G, "Lambda": Lambda, "a": a, "gamma": gamma}
        with h5py.File(self.filepath, "a") as f:
            grp_path = f"{_step_key(step)}/{layer_name}"
            if grp_path in f:
                if not overwrite:
                    raise ValueError(f"{grp_path} already exists (overwrite=False)")
                del f[grp_path]
            grp = f.require_group(grp_path)
            for name, arr in arrays.items():
                arr32 = np.asarray(arr, dtype=np.float32)
                grp.create_dataset(
                    name, data=arr32,
                    chunks=_chunk_shape(arr32.shape) if arr32.ndim else None,
                    compression="gzip", compression_opts=4,
                )

    def load_eigen(self, layer_name: str, step: int) -> Dict[str, np.ndarray]:
        with h5py.File(self.filepath, "r") as f:
            grp_path = f"{_step_key(step)}/{layer_name}"
            if grp_path not in f:
                raise KeyError(grp_path)
            grp = f[grp_path]
            return {k: grp[k][()] for k in ("Q_A", "Q_G", "Lambda", "a", "gamma")}

    def list_steps(self) -> List[int]:
        with h5py.File(self.filepath, "r") as f:
            return sorted(int(k.split("_", 1)[1]) for k in f.keys() if k.startswith("ckpt_"))

    def list_layers(self, step: int) -> List[str]:
        with h5py.File(self.filepath, "r") as f:
            key = _step_key(step)
            if key not in f:
                return []
            return list(f[key].keys())
