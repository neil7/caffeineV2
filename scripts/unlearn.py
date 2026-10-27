"""
scripts/unlearn.py
=================
Run an unlearning request against a trained checkpoint.

    python scripts/unlearn.py --config configs/resnet18_cifar10.yaml

Loads the latest checkpoint + run_meta.json, builds the forget split, removes
the forget points via K-FAC, and prints timing + influence norm.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kfac_unlearn import HDF5KFACStore, KFACUnlearner, make_cache
from experiments.common import load_run_meta
from experiments.unlearn import unlearn_set


def _load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _build_model(cfg, device):
    if cfg["modality"] == "resnet18_cifar10":
        from models.resnet18 import resnet18_cifar
        return resnet18_cifar(num_classes=cfg["model"]["num_classes"]).to(device)
    if cfg["modality"] == "distilbert_amazon":
        from models.distilbert import distilbert_amazon
        model, _ = distilbert_amazon(num_labels=cfg["model"]["num_labels"])
        return model.to(device)
    raise ValueError(f"Unknown modality: {cfg['modality']!r}")


def main():
    p = argparse.ArgumentParser(description="K-FAC unlearning request")
    p.add_argument("--config", required=True)
    p.add_argument("--storage-config", default="configs/storage.yaml")
    args = p.parse_args()

    cfg = _load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir = cfg["storage"]["ckpt_dir"]
    meta = load_run_meta(os.path.join(ckpt_dir, "run_meta.json"))
    last_step = meta["checkpoint_steps"][-1]

    model = _build_model(cfg, device)
    state = torch.load(os.path.join(ckpt_dir, f"step_{last_step}.pt"), map_location=device)
    model.load_state_dict(state["model"])

    store = HDF5KFACStore(cfg["storage"]["h5"])
    cache = make_cache(_load_yaml(args.storage_config))
    unlearner = KFACUnlearner(store, cache, damping=cfg["unlearn"]["damping"])

    # NOTE: assemble (z, target, z_id) forget batches from your data loader per
    # cfg['split']; left as a hook because it depends on the chosen modality.
    raise SystemExit(
        "Build the forget batches for your modality (see experiments/unlearn.py "
        "unlearn_set), then call unlearn_set(model, unlearner, forget_batches, "
        "criterion, checkpoint_step=%d, scale=1/%d)." % (last_step, meta["n_train"])
    )


if __name__ == "__main__":
    main()
