"""
benchmarks.retrain_baseline
=========================
Gold-standard unlearning: retrain a fresh model on the retain set only (the
forget points never seen). This is the reference the influence-function update
is measured against (CLAUDE.md metric 1: forget accuracy close to retrained).

Also exposes `actual_loss_diffs`, the influence-release `test_retraining`
quantity: for each candidate train point, retrain leaving it out and record the
change in test loss. Used to validate predicted influence (Pearson correlation).
"""

from typing import Callable, List

import numpy as np
import torch


def retrain_on_retain(
    make_model: Callable,
    retain_loader,
    device: str,
    epochs: int,
    lr: float,
    forward: Callable,
    weight_decay: float = 5e-4,
):
    """Train a fresh model from scratch on the retain set only. Returns the model."""
    model = make_model().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        for batch in retain_loader:
            opt.zero_grad()
            logits, labels = forward(model, batch, device)
            criterion(logits, labels).backward()
            opt.step()
    return model


@torch.no_grad()
def _test_loss(model, test_z, test_y, criterion, device):
    model.eval()
    return float(criterion(model(test_z.to(device)), test_y.to(device)).item())


def actual_loss_diffs(
    base_model,
    make_model: Callable,
    full_train_dataset,
    candidate_idx: List[int],
    test_z, test_y,
    device: str,
    retrain_steps: int,
    lr: float,
    forward: Callable,
) -> np.ndarray:
    """
    For each candidate train index: build a loader that excludes it, retrain a
    fresh copy for `retrain_steps`, and record (retrained_test_loss − base_test_loss).
    Mirrors influence-release experiments.test_retraining.

    NOTE: this is expensive (one retrain per candidate). Use a small candidate
    set for the sanity check, as Koh & Liang do.
    """
    criterion = torch.nn.CrossEntropyLoss()
    base_loss = _test_loss(base_model, test_z, test_y, criterion, device)

    diffs = np.zeros(len(candidate_idx))
    n = len(full_train_dataset)
    for c, idx in enumerate(candidate_idx):
        keep = [i for i in range(n) if i != idx]
        subset = torch.utils.data.Subset(full_train_dataset, keep)
        loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True)

        model = make_model().to(device)
        model.load_state_dict(base_model.state_dict())   # warm-start from base
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        model.train()
        seen = 0
        while seen < retrain_steps:
            for batch in loader:
                if seen >= retrain_steps:
                    break
                opt.zero_grad()
                logits, labels = forward(model, batch, device)
                criterion(logits, labels).backward()
                opt.step()
                seen += 1
        diffs[c] = _test_loss(model, test_z, test_y, criterion, device) - base_loss
    return diffs
