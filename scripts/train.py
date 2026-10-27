"""
scripts/train.py
===============
Thin CLI: read a YAML config and dispatch to the right modality train script.

    python scripts/train.py --config configs/resnet18_cifar10.yaml
    python scripts/train.py --config configs/distilbert_amazon.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    p = argparse.ArgumentParser(description="Train a modality + dump K-FAC factors")
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = _load_yaml(args.config)
    modality = cfg.get("modality")

    if modality == "resnet18_cifar10":
        from experiments.train_resnet_cifar import train as run
        ns = _resnet_namespace(cfg)
    elif modality == "distilbert_amazon":
        from experiments.train_distilbert_amazon import train as run
        ns = _distilbert_namespace(cfg)
    else:
        raise ValueError(f"Unknown modality in config: {modality!r}")
    run(ns)


def _resnet_namespace(cfg):
    return argparse.Namespace(
        epochs=cfg["train"]["epochs"],
        batch_size=cfg["data"]["batch_size"],
        lr=cfg["train"]["lr"],
        damping=cfg["unlearn"]["damping"],
        checkpoint_every=cfg["train"]["checkpoint_every"],
        collect_batches=cfg["train"]["collect_batches"],
        data_root=cfg["data"]["root"],
        h5=cfg["storage"]["h5"],
        ckpt_dir=cfg["storage"]["ckpt_dir"],
        device=None,
    )


def _distilbert_namespace(cfg):
    return argparse.Namespace(
        steps=cfg["train"]["steps"],
        batch_size=cfg["data"]["batch_size"],
        lr=cfg["train"]["lr"],
        damping=cfg["unlearn"]["damping"],
        checkpoint_every=cfg["train"]["checkpoint_every"],
        collect_batches=cfg["train"]["collect_batches"],
        subset_train=cfg["data"]["subset_train"],
        subset_test=cfg["data"]["subset_test"],
        h5=cfg["storage"]["h5"],
        ckpt_dir=cfg["storage"]["ckpt_dir"],
        device=None,
    )


if __name__ == "__main__":
    main()
