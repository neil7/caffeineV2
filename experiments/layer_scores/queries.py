"""
experiments.layer_scores.queries
===================================
Forget-query generation: R (random), C (class), M (memorization proxy),
N (neighborhood) — exact taxonomy and counts from the spec's "Step 3: Forget
queries" section (~150 queries total). Fixed seeds throughout; membership is
written to queries.json.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Query:
    id: str
    type: str  # "R" | "C" | "M" | "N"
    size: int
    seed: int
    indices: List[int]
    meta: Optional[dict] = None


def generate_random_queries(
    n_train: int, sizes=(1, 10, 100, 1000), draws_per_size: int = 15, seed_base: int = 1000
) -> List[Query]:
    queries = []
    for size in sizes:
        for d in range(draws_per_size):
            seed = seed_base + size * 100 + d
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_train, size=size, replace=False).tolist()
            queries.append(Query(id=f"R_s{size}_d{d}", type="R", size=size, seed=seed, indices=idx))
    return queries


def generate_class_queries(
    labels: np.ndarray, forget_pct: float = 0.10, seeds=(2000, 2001)
) -> List[Query]:
    queries = []
    classes = sorted(set(labels.tolist()))
    for c in classes:
        idx = np.where(labels == c)[0].tolist()
        queries.append(Query(id=f"C_full_{c}", type="C", size=len(idx), seed=-1, indices=idx,
                              meta={"class": c, "mode": "full"}))
    for c in classes:
        class_idx = np.where(labels == c)[0]
        n_take = max(1, int(round(forget_pct * len(class_idx))))
        for seed in seeds:
            rng = np.random.default_rng(seed + c)
            idx = rng.choice(class_idx, size=n_take, replace=False).tolist()
            queries.append(Query(id=f"C_pct{int(forget_pct*100)}_{c}_seed{seed}", type="C",
                                  size=n_take, seed=seed + c, indices=idx,
                                  meta={"class": c, "mode": f"{int(forget_pct*100)}pct"}))
    return queries


def generate_memorization_queries(
    per_example_loss: np.ndarray,
    k_list=(10, 100),
    n_random_draws: int = 18,
    random_draw_size: int = 100,
    top_pct: float = 0.01,
    seed_base: int = 3000,
) -> List[Query]:
    """
    Proxy for memorization: rank by per-example training loss (true labels,
    trained model), NOT true leave-out memorization scores
    (TODO(neil) if true memorization scoring via leave-out retraining is wanted).
    """
    order = np.argsort(per_example_loss)[::-1]  # highest loss first
    queries = []
    for k in k_list:
        idx = order[:k].tolist()
        queries.append(Query(id=f"M_top{k}", type="M", size=k, seed=-1, indices=idx,
                              meta={"mode": "top_loss", "proxy": True}))
    n_top1pct = max(1, int(round(top_pct * len(per_example_loss))))
    top_pool = order[:n_top1pct]
    for d in range(n_random_draws):
        seed = seed_base + d
        rng = np.random.default_rng(seed)
        idx = rng.choice(top_pool, size=min(random_draw_size, len(top_pool)), replace=False).tolist()
        queries.append(Query(id=f"M_rand_top1pct_d{d}", type="M", size=len(idx), seed=seed,
                              indices=idx, meta={"mode": "random_from_top1pct", "proxy": True}))
    return queries


def generate_neighborhood_queries(
    features: np.ndarray, k_list=(10, 100), seeds_per_k: int = 20, seed_base: int = 4000
) -> List[Query]:
    """
    features: (n_train, d) penultimate-layer features. Seed points chosen with
    a fixed-seed RNG; neighbors = k nearest by Euclidean distance.
    """
    n = features.shape[0]
    norms = np.sum(features ** 2, axis=1)
    queries = []
    for k in k_list:
        for s in range(seeds_per_k):
            seed = seed_base + k * 1000 + s
            rng = np.random.default_rng(seed)
            seed_point = int(rng.integers(0, n))
            d2 = norms + norms[seed_point] - 2 * features @ features[seed_point]
            nn_idx = np.argsort(d2)[: k + 1]
            nn_idx = [int(i) for i in nn_idx if i != seed_point][:k]
            queries.append(Query(id=f"N_k{k}_seed{seed}", type="N", size=len(nn_idx), seed=seed,
                                  indices=nn_idx, meta={"k": k, "seed_point": seed_point}))
    return queries


def save_queries(queries: List[Query], path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(q) for q in queries], f)


def load_queries(path: str) -> List[Query]:
    with open(path) as f:
        data = json.load(f)
    return [Query(**d) for d in data]


def main():
    p = argparse.ArgumentParser(description="Generate R/C/M/N forget queries, ~150 total")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--out", default="results/layer_scores/queries.json")
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cpu")
    args = p.parse_args()

    import torch
    from data.cifar10 import cifar10_loaders
    from models.resnet18 import resnet18_cifar
    from experiments.layer_scores.ekfac_fit import resolve_device

    device = resolve_device(args.device)
    train_loader, _, train_ds = cifar10_loaders(root=args.data_root, batch_size=256, augment=False)
    labels = np.array(train_ds.targets)

    model = resnet18_cifar(num_classes=10).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    losses = np.zeros(len(train_ds), dtype=np.float64)
    features = np.zeros((len(train_ds), 512), dtype=np.float32)

    feat_cache = {}

    def avgpool_hook(module, inputs, output):
        feat_cache["feat"] = output.detach().flatten(1)

    h = model.avgpool.register_forward_hook(avgpool_hook)
    with torch.no_grad():
        offset = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses[offset:offset + x.shape[0]] = criterion(logits, y).cpu().numpy()
            features[offset:offset + x.shape[0]] = feat_cache["feat"].cpu().numpy()
            offset += x.shape[0]
    h.remove()

    queries: List[Query] = []
    queries += generate_random_queries(len(train_ds))
    queries += generate_class_queries(labels)
    queries += generate_memorization_queries(losses)
    queries += generate_neighborhood_queries(features)

    print(f"generated {len(queries)} queries "
          f"(R={sum(q.type=='R' for q in queries)}, C={sum(q.type=='C' for q in queries)}, "
          f"M={sum(q.type=='M' for q in queries)}, N={sum(q.type=='N' for q in queries)})")
    save_queries(queries, args.out)


if __name__ == "__main__":
    main()
