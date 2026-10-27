"""
experiments.train_distilbert_amazon
==================================
Fine-tune DistilBERT on amazon_reviews_multi (5-star), periodically saving K-FAC
factors for the transformer/FFN/head Linear layers into an HDF5KFACStore.

Run:
    python -m experiments.train_distilbert_amazon --steps 3000 --h5 kfac_factors_distilbert.h5
"""

import argparse
import os

import torch

from kfac_unlearn import HDF5KFACStore
from models.distilbert import distilbert_amazon, kfac_layers
from data.amazon import amazon_loaders
from caffeine.kfac_collector import KFACCollector
from experiments.common import write_run_meta


def _hf_forward(model, batch, device):
    """DistilBERT forward: batch has input_ids / attention_mask / labels."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits, labels


def _collect(model, loader, store, step, device, criterion, layer_names, max_batches):
    model.eval()
    collector = KFACCollector(model, layer_names=layer_names)
    with collector:
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            model.zero_grad(set_to_none=True)
            logits, labels = _hf_forward(model, batch, device)
            # mask-aware: drop pad tokens from A/G accumulation
            for name in layer_names:
                collector.set_token_mask(name, batch["attention_mask"].to(device))
            criterion(logits, labels).backward()
    written = collector.save_to_store(store, step=step, overwrite=True)
    collector.reset()
    return written


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = distilbert_amazon(num_labels=5)
    model.to(device)

    train_loader, test_loader, train_ds = amazon_loaders(
        tokenizer, batch_size=args.batch_size,
        subset_train=args.subset_train, subset_test=args.subset_test,
    )
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    layer_names = kfac_layers(model)

    store = HDF5KFACStore(args.h5)
    saved_steps = []
    step = 0
    model.train()
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break
            opt.zero_grad()
            logits, labels = _hf_forward(model, batch, device)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
            step += 1

            if step % args.checkpoint_every == 0 or step == args.steps:
                os.makedirs(args.ckpt_dir, exist_ok=True)
                torch.save(
                    {"model": model.state_dict(), "step": step},
                    os.path.join(args.ckpt_dir, f"step_{step}.pt"),
                )
                written = _collect(
                    model, train_loader, store, step, device, criterion,
                    layer_names, args.collect_batches,
                )
                saved_steps.append(step)
                print(f"[step {step}] loss={loss.item():.4f} layers_saved={len(written)}")
                model.train()

    write_run_meta(
        os.path.join(args.ckpt_dir, "run_meta.json"),
        layer_names=layer_names,
        n_train=len(train_ds),
        steps=saved_steps,
        damping=args.damping,
        extra={"modality": "distilbert_amazon"},
    )


def main():
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT on Amazon reviews + K-FAC factors")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--damping", type=float, default=1e-2)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--collect-batches", type=int, default=50)
    p.add_argument("--subset-train", type=int, default=50_000)
    p.add_argument("--subset-test", type=int, default=10_000)
    p.add_argument("--h5", default="kfac_factors_distilbert.h5")
    p.add_argument("--ckpt-dir", default="checkpoints/distilbert_amazon")
    p.add_argument("--device", default=None)
    train(p.parse_args())


if __name__ == "__main__":
    main()
