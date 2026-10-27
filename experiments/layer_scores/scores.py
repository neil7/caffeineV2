"""
experiments.layer_scores.scores
==================================
Per-(query, layer, alpha) EK-FAC scores: forget gradient V_l, score s_l,
rank-r sketch bounds. Writes scores.parquet (E1/E2 input) and leaves the
rank-r sweep to e3_bounds.py (bounds.parquet).
"""

import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from caffeine.grad_extract import extract_grad_W_per_layer
from experiments.layer_scores.ekfac_math import (
    full_bytes,
    layer_score,
    rotate_to_eigenbasis,
)
from experiments.layer_scores.layer_spec import LayerSpec, build_layer_specs
from experiments.layer_scores.queries import Query, load_queries

BN_PARAM_SUBSTRINGS = ("bn", "downsample.1")  # torchvision BasicBlock BN module names


def forget_gradient(
    model: nn.Module, dataset, indices: List[int], specs: List[LayerSpec], device
) -> Dict[str, np.ndarray]:
    """
    V_l = sum_{z in query} grad_W_l(z), true labels, model.eval(), no
    augmentation (caller must pass a no-augmentation dataset).

    Reuses caffeine.grad_extract.extract_grad_W_per_layer (single-point, not
    modified) once per point and accumulates — the spec's forget gradient is
    a sum over the query set, not a new primitive worth adding to grad_extract.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    spec_by_name = {s.name: s for s in specs}
    V: Dict[str, np.ndarray] = {s.name: np.zeros((s.d_out, s.d_in), dtype=np.float64) for s in specs}
    bn_grad_sq_sum = 0.0
    for i in indices:
        x, y = dataset[i]
        x = x.unsqueeze(0).to(device)
        target = torch.tensor([y], device=device)
        grads = extract_grad_W_per_layer(model, x, target, criterion)
        for name, g in grads.items():
            if name not in V:
                continue
            spec = spec_by_name[name]
            g64 = g.astype(np.float64)
            if spec.has_bias:
                bias_grad = spec.module.bias.grad.detach().cpu().numpy().astype(np.float64).reshape(-1, 1)
                g64 = np.concatenate([g64, bias_grad], axis=1)
            V[name] += g64
        bn_grad_sq_sum += sum(
            float((p.grad.double() ** 2).sum())
            for n, p in model.named_parameters()
            if p.grad is not None and any(s in n for s in BN_PARAM_SUBSTRINGS)
        )
    return V, bn_grad_sq_sum


def score_query(
    V: Dict[str, np.ndarray], eigs: Dict[str, dict], alpha: float
) -> Dict[str, float]:
    """Per-layer s_l for one query at one alpha (per-layer damping convention)."""
    out = {}
    for name, V_l in V.items():
        e = eigs[name]
        Q_A, Q_G, Lambda = e["Q_A"], e["Q_G"], e["Lambda"]
        lam = float(alpha * np.mean(Lambda))
        C = rotate_to_eigenbasis(V_l.astype(np.float32), Q_A, Q_G)
        out[name] = layer_score(C, Lambda, lam)
    return out


def build_scores_table(
    model: nn.Module,
    dataset,
    queries: List[Query],
    specs: List[LayerSpec],
    eigs: Dict[str, dict],
    alphas: List[float],
    device,
) -> pd.DataFrame:
    rows = []
    bytes_l = {s.name: full_bytes(s.d_in, s.d_out) for s in specs}
    for q in queries:
        V, bn_sq = forget_gradient(model, dataset, q.indices, specs, device)
        v_fro2_total = sum(float(np.sum(v ** 2)) for v in V.values())
        for alpha in alphas:
            s = score_query(V, eigs, alpha)
            s_total = sum(s.values())
            for name in V:
                rows.append({
                    "query_id": q.id, "query_type": q.type, "query_size": q.size,
                    "alpha": alpha, "layer": name,
                    "v_fro": float(np.linalg.norm(V[name])),
                    "s_l": s[name],
                    "s_share": s[name] / s_total if s_total > 0 else 0.0,
                    "bytes_l": bytes_l[name],
                    "bn_grad_norm_share": (bn_sq / (bn_sq + v_fro2_total)) if (bn_sq + v_fro2_total) > 0 else 0.0,
                })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Compute per-(query, layer, alpha) EK-FAC scores")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--factors-h5", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--out", default="results/layer_scores/scores.parquet")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cpu")
    args = p.parse_args()

    from data.cifar10 import cifar10_loaders
    from models.resnet18 import resnet18_cifar
    from experiments.layer_scores.ekfac_fit import resolve_device
    from experiments.layer_scores.ekfac_store import EKFACStore

    device = resolve_device(args.device)
    model = resnet18_cifar(num_classes=10).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    _, _, train_ds_aug = cifar10_loaders(root=args.data_root, augment=False)
    specs = build_layer_specs(model)

    store = EKFACStore(args.factors_h5)
    eigs = {s.name: store.load_eigen(s.name, args.step) for s in specs}

    queries = load_queries(args.queries)
    df = build_scores_table(model, train_ds_aug, queries, specs, eigs, args.alphas, device)
    df.to_parquet(args.out)
    print(f"wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
