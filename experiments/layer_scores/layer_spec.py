"""
experiments.layer_scores.layer_spec
=====================================
Model-agnostic layer registry + activation/grad-output reshape rules for
EK-FAC fitting (Ground Rule #6: written against a layer-module list + a
reshape function so a future Pythia run can reuse it; only Conv2d/Linear
targets are implemented now).

Reuses caffeine.layer_registry.instrumented_layers (unmodified) rather than
reimplementing module discovery.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from caffeine.layer_registry import instrumented_layers, weight_to_2d


@dataclass
class LayerSpec:
    name: str
    module: nn.Module
    d_out: int
    d_in: int  # includes the +1 bias column when has_bias
    has_bias: bool


def build_layer_specs(model: nn.Module, layer_names: Optional[List[str]] = None) -> List[LayerSpec]:
    """[(name, module), ...] in network order -> LayerSpec list, bias-augmented d_in."""
    targets = instrumented_layers(model)
    if layer_names is not None:
        wanted = set(layer_names)
        targets = [(n, m) for n, m in targets if n in wanted]

    specs = []
    for name, mod in targets:
        has_bias = mod.bias is not None
        w2d = weight_to_2d(mod, mod.weight)
        d_out, d_in = w2d.shape
        specs.append(LayerSpec(name=name, module=mod, d_out=d_out,
                                d_in=d_in + (1 if has_bias else 0),
                                has_bias=has_bias))
    return specs


def conv_input_patches(module: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """im2col: (N, C_in, H, W) -> (N*L, C_in*kH*kW), matching weight.reshape(C_out, -1)."""
    patches = F.unfold(
        x, kernel_size=module.kernel_size, dilation=module.dilation,
        padding=module.padding, stride=module.stride,
    )  # (N, C_in*kH*kW, L)
    N, K, L = patches.shape
    return patches.permute(0, 2, 1).reshape(N * L, K)


def reshape_activations(spec: LayerSpec, x: torch.Tensor) -> torch.Tensor:
    """Unfolded/flattened, bias-augmented input activations -> (M, d_in)."""
    if isinstance(spec.module, nn.Conv2d):
        a = conv_input_patches(spec.module, x)
    else:
        a = x.reshape(-1, x.shape[-1])
    if spec.has_bias:
        ones = torch.ones(a.shape[0], 1, device=a.device, dtype=a.dtype)
        a = torch.cat([a, ones], dim=1)
    return a


def reshape_grad_output(spec: LayerSpec, go: torch.Tensor) -> torch.Tensor:
    """Output-gradient, per spatial position for conv -> (M, d_out)."""
    if isinstance(spec.module, nn.Conv2d):
        return go.permute(0, 2, 3, 1).reshape(-1, go.shape[1])
    return go.reshape(-1, go.shape[-1])


def grad_W_from_patches(spec: LayerSpec, a: torch.Tensor, g: torch.Tensor, n: int) -> torch.Tensor:
    """
    Per-example weight gradient: grad_W = sum_t g_t a_t^T, summed over the n
    spatial/sequence positions belonging to one example (a, g are the flattened
    (M, d_in)/(M, d_out) activations/grad-outputs for a single example; M=n for
    linear layers, M=n*H'*W' for conv layers already summed per-example here by
    the caller passing one example's rows at a time).
    """
    return g.t() @ a  # (d_out, d_in)
