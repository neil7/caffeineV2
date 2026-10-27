"""
caffeine
========
PyTorch glue that connects real models to the numpy K-FAC unlearning engine in
`kfac_unlearn`.

  layer_registry  — which modules to instrument + the 2D reshape rules
  kfac_collector  — forward/backward hooks accumulating A and G into HDF5
  grad_extract    — per-layer gradient + weight extraction for a forget point
"""

from caffeine.layer_registry import (
    INSTRUMENTED_TYPES,
    instrumented_layers,
    weight_to_2d,
    weight_grad_2d,
)
from caffeine.kfac_collector import KFACCollector
from caffeine.grad_extract import (
    extract_grad_W_per_layer,
    model_weights_per_layer,
    write_back_weights,
)

__all__ = [
    "INSTRUMENTED_TYPES",
    "instrumented_layers",
    "weight_to_2d",
    "weight_grad_2d",
    "KFACCollector",
    "extract_grad_W_per_layer",
    "model_weights_per_layer",
    "write_back_weights",
]
