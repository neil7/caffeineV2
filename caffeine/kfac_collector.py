"""
caffeine.kfac_collector
=======================
Collect K-FAC factors A and G during training via forward/backward hooks, then
export float32 numpy matrices into an HDF5KFACStore.

For a layer y = Wx:
    A = E[x xᵀ]      input activation covariance   (d_in × d_in)
    G = E[∇y ∇yᵀ]    output-gradient covariance     (d_out × d_out)

so the per-layer Hessian block H_W ≈ A ⊗ G and the Newton step is
vec(G⁻¹ ∇W A⁻ᵀ) — `kfac_newton_step` sees only 2D matrices.

Conv2d:
    A : F.unfold (im2col) the layer input → (N*L, C_in*kH*kW) patch matrix; the
        column order matches weight.reshape(C_out, -1) so factors line up.
    G : output-grad covariance summed over spatial positions.

Batch-N scaling (subtle, load-bearing): PyTorch backward grad_output is mean-
reduced over the batch when the loss uses reduction='mean'. We multiply the
captured output grad by the batch size N so G is an honest E[∇y∇yᵀ] of per-
example gradients rather than 1/N too small. A global rescale of G cancels in a
Pearson correlation but NOT after damping or in absolute Δθ norms — so the
scaling matters for faithful magnitudes.

NLP: call set_token_mask(name, mask) so pad tokens are excluded from A/G.

Usage
-----
    collector = KFACCollector(model)
    with collector:                       # hooks live only inside the block
        for x, y in collect_loader:
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)   # reduction='mean'
            loss.backward()
    collector.save_to_store(store, step=global_step)
    collector.reset()
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from caffeine.layer_registry import instrumented_layers


class KFACCollector:
    """
    Accumulate A and G for every Linear / Conv2d layer over one or more
    collection passes, then dump to an HDF5KFACStore.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None,
        include_bias: bool = False,
    ):
        self.model = model
        self.include_bias = include_bias

        targets = instrumented_layers(model)
        if layer_names is not None:
            wanted = set(layer_names)
            targets = [(n, m) for n, m in targets if n in wanted]
        self._targets: List[Tuple[str, nn.Module]] = targets

        # accumulators
        self._A_sum: Dict[str, torch.Tensor] = {}
        self._G_sum: Dict[str, torch.Tensor] = {}
        self._count: Dict[str, int] = {}
        # per-forward attention mask (NLP), keyed by layer name
        self._mask: Dict[str, torch.Tensor] = {}

        self._handles: list = []

    # ── context manager ───────────────────────────────────────────────────

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *exc):
        self.detach()
        return False

    # ── attach / detach hooks ─────────────────────────────────────────────

    def attach(self) -> None:
        if self._handles:
            return
        for name, mod in self._targets:
            self._handles.append(mod.register_forward_hook(self._fwd_hook(name)))
            self._handles.append(mod.register_full_backward_hook(self._bwd_hook(name)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # backwards-compatible alias
    def remove(self) -> None:
        self.detach()

    def reset(self) -> None:
        """Zero all accumulators — start a fresh collection window."""
        self._A_sum.clear()
        self._G_sum.clear()
        self._count.clear()
        self._mask.clear()

    def set_token_mask(self, name: str, mask: torch.Tensor) -> None:
        """
        Optional: provide a (N, seq) attention mask for an NLP layer so pad
        tokens are dropped before accumulating A/G. Call before backward.
        """
        self._mask[name] = mask

    # ── hooks ─────────────────────────────────────────────────────────────

    def _fwd_hook(self, name: str):
        def hook(module, inputs, output):
            with torch.no_grad():
                x = inputs[0].detach()
                if isinstance(module, nn.Conv2d):
                    a = self._conv_input_patches(module, x)         # (M, d_in)
                else:  # Linear
                    a = x.reshape(-1, x.shape[-1])                  # (M, d_in)
                    a = self._apply_mask(name, a, x)
                    if self.include_bias:
                        ones = torch.ones(a.shape[0], 1, device=a.device, dtype=a.dtype)
                        a = torch.cat([a, ones], dim=1)
                a = a.float()
                A = a.t() @ a                                       # (d_in, d_in)
                self._accumulate("A", name, A, a.shape[0])
        return hook

    def _bwd_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            with torch.no_grad():
                go = grad_output[0].detach()
                # undo loss reduction='mean' over the batch
                N = go.shape[0]
                go = go * float(N)
                if isinstance(module, nn.Conv2d):
                    # (N, C_out, H', W') -> (N*H'*W', C_out): sum over spatial
                    g = go.permute(0, 2, 3, 1).reshape(-1, go.shape[1])
                else:  # Linear
                    g = go.reshape(-1, go.shape[-1])
                    g = self._apply_mask(name, g, go)
                g = g.float()
                G = g.t() @ g                                       # (d_out, d_out)
                self._accumulate("G", name, G, g.shape[0])
        return hook

    # ── helpers ───────────────────────────────────────────────────────────

    def _conv_input_patches(self, module: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        """im2col the conv input → (N*L, C_in*kH*kW), matching weight.reshape(C_out, -1)."""
        patches = F.unfold(
            x,
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        )  # (N, C_in*kH*kW, L)
        N, K, L = patches.shape
        return patches.permute(0, 2, 1).reshape(N * L, K)

    def _apply_mask(self, name: str, flat: torch.Tensor, orig: torch.Tensor) -> torch.Tensor:
        """Drop pad-token rows for NLP layers when a mask was supplied."""
        mask = self._mask.get(name)
        if mask is None:
            return flat
        # orig is (N, seq, d); mask is (N, seq) -> flatten to (N*seq,)
        if orig.dim() == 3 and tuple(mask.shape[:2]) == tuple(orig.shape[:2]):
            sel = mask.reshape(-1).bool()
            return flat[sel]
        return flat

    def _accumulate(self, which: str, name: str, mat: torch.Tensor, n: int) -> None:
        store = self._A_sum if which == "A" else self._G_sum
        if name not in store:
            store[name] = mat.clone()
            if which == "A":
                self._count[name] = 0
        else:
            store[name] += mat
        if which == "A":
            self._count[name] += n

    # ── export ────────────────────────────────────────────────────────────

    def factors(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (A, G) as float32 numpy averages for one layer."""
        if name not in self._A_sum or name not in self._G_sum:
            raise KeyError(f"No complete factors accumulated for layer '{name}'.")
        count = max(self._count[name], 1)
        A = (self._A_sum[name] / count).cpu().numpy().astype(np.float32)
        G = (self._G_sum[name] / count).cpu().numpy().astype(np.float32)
        return A, G

    def save_to_store(self, store, step: int, overwrite: bool = True) -> List[str]:
        """
        Save every accumulated layer's (A, G) into an HDF5KFACStore at `step`.
        Returns the list of layer names written.
        """
        written = []
        for name, _ in self._targets:
            if name in self._A_sum and name in self._G_sum:
                A, G = self.factors(name)
                store.save_kfac(name, step, A, G, overwrite=overwrite)
                written.append(name)
        return written

    # backwards-compatible alias
    def save(self, store, step: int, overwrite: bool = True) -> List[str]:
        return self.save_to_store(store, step, overwrite=overwrite)
