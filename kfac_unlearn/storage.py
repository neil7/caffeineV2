"""
kfac_unlearn.storage
====================
Persistent, chunked HDF5 store for K-FAC factor matrices.

File layout inside the .h5 store:

  /conv1
      /A/step_5000    ← numpy array, float32, chunked + gzip-compressed
      /A/step_10000
      /G/step_5000
      /G/step_10000
  /layer2
      /A/step_5000
      /G/step_5000
  /_meta
      layer_dims       ← JSON attr: {"conv1": [64, 64], "layer2": [128, 128]}
      checkpoint_steps ← JSON attr: [5000, 10000, 15000]

Key construction:  f"{layer_name}/{factor}/{step_key}"  e.g. "conv1/A/step_5000"

Only two matrices are read per unlearning request — the "page in only what you
need" property analogous to vLLM's PagedAttention.
"""

import json
from contextlib import contextmanager
from typing import Dict, Tuple

import h5py
import numpy as np


def _step_key(step: int) -> str:
    """Convert a training step integer to a valid HDF5 group name."""
    return f"step_{step}"


def _chunk_shape(matrix_shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    Choose HDF5 chunk dimensions so that:
      - A single row-slice reads efficiently (for partial Hessian loads)
      - Chunks stay under ~1MB each
    Rule: chunk rows = min(d, 256), chunk cols = min(d, 256)
    """
    r, c = matrix_shape
    return (min(r, 256), min(c, 256))


class HDF5KFACStore:
    """
    Persistent, chunked store for K-FAC factor matrices.

    Each layer's Kronecker factors A and G are stored independently,
    so a single-layer unlearning request reads only two small matrices
    rather than the entire checkpoint.

    Usage
    -----
    store = HDF5KFACStore("kfac_factors.h5")

    # During / after training — save factors
    store.save_kfac("conv1", step=5000, A=a_matrix, G=g_matrix)

    # During unlearning — load factors for one layer
    A, G = store.load_kfac("conv1", step=5000)

    # Find the nearest available checkpoint to an arbitrary step
    step = store.nearest_checkpoint("conv1", target_step=4800)
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._ensure_meta_groups()

    # ── internal helpers ──────────────────────────────────────────────────

    def _ensure_meta_groups(self):
        with h5py.File(self.filepath, "a") as f:
            f.require_group("_meta")

    @contextmanager
    def _open(self, mode: str = "r"):
        with h5py.File(self.filepath, mode) as f:
            yield f

    def _factor_path(self, layer: str, factor: str, step: int) -> str:
        """Build the full HDF5 dataset path for a factor."""
        return f"{layer}/{factor}/{_step_key(step)}"

    # ── public API ────────────────────────────────────────────────────────

    def save_kfac(
        self,
        layer_name: str,
        step: int,
        A: np.ndarray,
        G: np.ndarray,
        overwrite: bool = False,
    ) -> None:
        """
        Save K-FAC factors A and G for a layer at a training checkpoint.

        Parameters
        ----------
        layer_name : str   e.g. "conv1", "layer4.1", "fc2"
        step       : int   training step at which factors were computed
        A          : 2-D float32 array  — input activation covariance
        G          : 2-D float32 array  — output gradient covariance
        overwrite  : bool  if False, raises if dataset already exists
        """
        A = np.asarray(A, dtype=np.float32)
        G = np.asarray(G, dtype=np.float32)

        with self._open("a") as f:
            for factor_name, matrix in [("A", A), ("G", G)]:
                path = self._factor_path(layer_name, factor_name, step)

                if path in f:
                    if not overwrite:
                        raise ValueError(
                            f"Dataset '{path}' already exists. "
                            "Pass overwrite=True to replace it."
                        )
                    del f[path]

                chunks = _chunk_shape(matrix.shape)
                f.create_dataset(
                    path,
                    data=matrix,
                    chunks=chunks,
                    compression="gzip",
                    compression_opts=4,   # balanced speed / size
                )

            # update metadata
            meta = self._read_meta(f)
            meta["layer_dims"][layer_name] = list(A.shape)
            if step not in meta["checkpoint_steps"]:
                meta["checkpoint_steps"].append(step)
                meta["checkpoint_steps"].sort()
            self._write_meta(f, meta)

    def load_kfac(
        self, layer_name: str, step: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load K-FAC factors for one layer at one checkpoint step.

        Returns
        -------
        (A, G) as float32 numpy arrays.

        Only two matrices are read from disk — no other layer data is touched.
        This is the "page in only what you need" property analogous to vLLM.
        """
        with self._open("r") as f:
            path_A = self._factor_path(layer_name, "A", step)
            path_G = self._factor_path(layer_name, "G", step)

            if path_A not in f:
                raise KeyError(
                    f"No A factor for layer '{layer_name}' at step {step}. "
                    f"Available: {self.list_steps(layer_name)}"
                )
            if path_G not in f:
                raise KeyError(f"No G factor for layer '{layer_name}' at step {step}.")

            A = f[path_A][:]   # reads only this dataset's chunks
            G = f[path_G][:]
        return A, G

    def load_kfac_inverse(
        self, layer_name: str, step: int, damping: float = 1e-4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load A and G and return their inverses directly.

        Damping (λI) is added before inversion to handle near-singular matrices —
        this is the standard fix from Koh & Liang's non-convex extension.
        """
        A, G = self.load_kfac(layer_name, step)
        dA = A.shape[0]
        dG = G.shape[0]
        A_inv = np.linalg.inv(A + damping * np.eye(dA, dtype=np.float32))
        G_inv = np.linalg.inv(G + damping * np.eye(dG, dtype=np.float32))
        return A_inv, G_inv

    def list_steps(self, layer_name: str) -> list:
        """Return sorted list of checkpoint steps available for a layer."""
        with self._open("r") as f:
            if layer_name not in f:
                return []
            if "A" not in f[layer_name]:
                return []
            return sorted(
                int(k.replace("step_", ""))
                for k in f[layer_name]["A"].keys()
            )

    def nearest_checkpoint(self, layer_name: str, target_step: int) -> int:
        """
        Return the stored checkpoint step closest to target_step.

        Useful when an unlearning request arrives with an arbitrary step
        number — pick the nearest saved checkpoint rather than failing.
        """
        steps = self.list_steps(layer_name)
        if not steps:
            raise KeyError(f"No checkpoints found for layer '{layer_name}'.")
        return min(steps, key=lambda s: abs(s - target_step))

    def list_layers(self) -> list:
        """Return all layer names that have stored factors."""
        with self._open("r") as f:
            return [k for k in f.keys() if k != "_meta"]

    def storage_summary(self) -> Dict:
        """
        Return a dict showing disk usage per layer — useful for debugging
        and deciding which checkpoints to prune.
        """
        summary = {}
        with self._open("r") as f:
            for layer in self.list_layers():
                layer_bytes = 0
                for factor in ["A", "G"]:
                    if factor in f[layer]:
                        for step_key in f[layer][factor]:
                            ds = f[layer][factor][step_key]
                            layer_bytes += ds.id.get_storage_size()
                summary[layer] = {
                    "steps": self.list_steps(layer),
                    "disk_kb": round(layer_bytes / 1024, 1),
                }
        return summary

    # ── metadata helpers ─────────────────────────────────────────────────

    def _read_meta(self, f: h5py.File) -> dict:
        meta = {"layer_dims": {}, "checkpoint_steps": []}
        if "layer_dims" in f["_meta"].attrs:
            meta["layer_dims"] = json.loads(f["_meta"].attrs["layer_dims"])
        if "checkpoint_steps" in f["_meta"].attrs:
            meta["checkpoint_steps"] = json.loads(f["_meta"].attrs["checkpoint_steps"])
        return meta

    def _write_meta(self, f: h5py.File, meta: dict) -> None:
        f["_meta"].attrs["layer_dims"] = json.dumps(meta["layer_dims"])
        f["_meta"].attrs["checkpoint_steps"] = json.dumps(meta["checkpoint_steps"])
