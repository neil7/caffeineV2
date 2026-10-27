"""
data.splits
==========
Forget / retain / test split convention shared by both modalities.

    train = retain ∪ forget   (disjoint)
    test  = held-out test set (never touched)

Two forget modes (seed 17 mirrors influence-release `test_retraining`):
    "random" — a random subset of training indices
    "class"  — every training index of one label (whole-class forgetting)
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class UnlearnSplit:
    retain_idx: np.ndarray
    forget_idx: np.ndarray
    test_idx: np.ndarray


def make_split(
    n_train: int,
    n_test: int,
    forget_size: int = 100,
    mode: str = "random",
    forget_class: int = None,
    labels: np.ndarray = None,
    seed: int = 17,
) -> UnlearnSplit:
    """
    Build a split over training indices [0, n_train) and test indices
    [0, n_test).

    mode="random" : forget `forget_size` random training points.
    mode="class"  : forget every training point with label `forget_class`
                    (requires `labels`, the per-train-index label array).
    """
    rng = np.random.default_rng(seed)
    all_train = np.arange(n_train)

    if mode == "random":
        forget_idx = rng.choice(all_train, size=forget_size, replace=False)
    elif mode == "class":
        if labels is None or forget_class is None:
            raise ValueError("mode='class' needs labels and forget_class.")
        forget_idx = all_train[np.asarray(labels) == forget_class]
    else:
        raise ValueError(f"Unknown forget mode: {mode!r}")

    forget_set = set(forget_idx.tolist())
    retain_idx = np.array([i for i in all_train if i not in forget_set])
    test_idx = np.arange(n_test)

    return UnlearnSplit(
        retain_idx=retain_idx,
        forget_idx=np.asarray(forget_idx),
        test_idx=test_idx,
    )
