"""
experiments.common
==================
Shared training / collection helpers used by both modality train scripts.
"""

import json
import os
from typing import Callable, Dict, List

import torch

from caffeine.kfac_collector import KFACCollector


@torch.no_grad()
def accuracy(model, loader, device: str, forward: Callable = None) -> float:
    """Top-1 accuracy over a loader. `forward(model, batch) -> (logits, labels)`."""
    model.eval()
    correct = total = 0
    for batch in loader:
        logits, labels = (forward or _default_forward)(model, batch, device)
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def _default_forward(model, batch, device):
    """Vision default: batch is (images, labels)."""
    x, y = batch
    x, y = x.to(device), y.to(device)
    return model(x), y


def collect_and_save_factors(
    model,
    collect_loader,
    store,
    step: int,
    device: str,
    loss_fn: Callable,
    forward: Callable = None,
    max_batches: int = 50,
) -> List[str]:
    """
    Run a K-FAC collection pass over `collect_loader` (eval mode for stable BN
    statistics) and save A/G into the HDF5 store at `step`.
    """
    model.eval()
    collector = KFACCollector(model)
    with collector:
        for i, batch in enumerate(collect_loader):
            if i >= max_batches:
                break
            model.zero_grad(set_to_none=True)
            logits, labels = (forward or _default_forward)(model, batch, device)
            loss = loss_fn(logits, labels)
            loss.backward()
    written = collector.save_to_store(store, step=step, overwrite=True)
    collector.reset()
    return written


def write_run_meta(
    path: str,
    layer_names: List[str],
    n_train: int,
    steps: List[int],
    damping: float,
    extra: Dict = None,
) -> None:
    """
    Persist the metadata the unlearn/eval harness needs: the 1/n normalization
    count, instrumented layers, checkpoint steps, and damping used.
    """
    meta = {
        "layer_names": layer_names,
        "n_train": n_train,
        "checkpoint_steps": steps,
        "damping": damping,
    }
    if extra:
        meta.update(extra)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def load_run_meta(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)
