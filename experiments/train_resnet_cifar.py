"""
experiments.train_resnet_cifar
=============================
Train ResNet-18 on CIFAR-10, periodically checkpointing weights and dumping
K-FAC factors (A, G) into an HDF5KFACStore.

Run:
    python -m experiments.train_resnet_cifar --epochs 100 --h5 kfac_factors_resnet.h5
"""

import argparse
import os

import torch
import torch.nn as nn

from kfac_unlearn import HDF5KFACStore
from models.resnet18 import resnet18_cifar, kfac_layers
from data.cifar10 import cifar10_loaders
from experiments.common import (
    accuracy,
    collect_and_save_factors,
    write_run_meta,
)


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, train_ds = cifar10_loaders(
        root=args.data_root, batch_size=args.batch_size
    )
    # un-augmented loader for factor collection (stable statistics)
    collect_loader, _, _ = cifar10_loaders(
        root=args.data_root, batch_size=args.batch_size, augment=False
    )

    model = resnet18_cifar(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    store = HDF5KFACStore(args.h5)
    steps_per_epoch = len(train_loader)
    global_step = 0
    saved_steps = []

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            global_step += 1
        sched.step()

        if (epoch + 1) % args.checkpoint_every == 0 or (epoch + 1) == args.epochs:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "step": global_step},
                os.path.join(args.ckpt_dir, f"step_{global_step}.pt"),
            )
            written = collect_and_save_factors(
                model, collect_loader, store, step=global_step,
                device=device, loss_fn=criterion, max_batches=args.collect_batches,
            )
            saved_steps.append(global_step)
            acc = accuracy(model, test_loader, device)
            print(f"[epoch {epoch+1}] step {global_step}  test_acc={acc:.4f}  "
                  f"layers_saved={len(written)}")

    write_run_meta(
        os.path.join(args.ckpt_dir, "run_meta.json"),
        layer_names=kfac_layers(model),
        n_train=len(train_ds),
        steps=saved_steps,
        damping=args.damping,
        extra={"modality": "resnet18_cifar10"},
    )


def main():
    p = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-10 + K-FAC factors")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--damping", type=float, default=1e-2)   # CNN value (Koh CNN: 1e-2)
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--collect-batches", type=int, default=50)
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--h5", default="kfac_factors_resnet.h5")
    p.add_argument("--ckpt-dir", default="checkpoints/resnet_cifar")
    p.add_argument("--device", default=None)
    train(p.parse_args())


if __name__ == "__main__":
    main()
