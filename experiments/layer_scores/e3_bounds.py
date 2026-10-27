"""
experiments.layer_scores.e3_bounds
=====================================
E3: rank-r sketch bound tightness (U/L, m_out fraction) + a sound fetch
planner vs. an oracle, with the soundness assertion from the spec.
"""

import argparse
from typing import Dict, List

import numpy as np
import pandas as pd

from experiments.layer_scores.ekfac_math import rank_r_sketch
from experiments.layer_scores.ekfac_store import EKFACStore
from experiments.layer_scores.layer_spec import build_layer_specs
from experiments.layer_scores.queries import load_queries
from experiments.layer_scores.scores import forget_gradient

R_LIST = [0, 8, 16, 32, 64, 128, 256]
EPS_LIST = [0.01, 0.05, 0.10]


def compute_bounds_table(
    model, dataset, queries, specs, eigs: Dict[str, dict], alpha: float, device
) -> pd.DataFrame:
    rows = []
    for q in queries:
        V, _ = forget_gradient(model, dataset, q.indices, specs, device)
        for s in specs:
            name = s.name
            e = eigs[name]
            Q_A, Q_G, Lambda = e["Q_A"], e["Q_G"], e["Lambda"]
            lam = float(alpha * np.mean(Lambda))
            v_fro2 = float(np.sum(V[name].astype(np.float64) ** 2))
            for r in R_LIST:
                res = rank_r_sketch(V[name].astype(np.float32), Q_A, Q_G, Lambda, lam, r)
                rows.append({
                    "query_id": q.id, "query_type": q.type, "layer": name, "r": r,
                    "L": res["L"], "U": res["U"],
                    "U_over_L": res["U"] / res["L"] if res["L"] > 0 else float("inf"),
                    "m_out_frac": res["m_out"] / v_fro2 if v_fro2 > 0 else 0.0,
                    "sketch_bytes": res["sketch_bytes"],
                })
    return pd.DataFrame(rows)


def sound_planner(bounds_r: pd.DataFrame, eps: float) -> set:
    """
    bounds_r: one row per layer at a fixed r, with columns L, U, sketch_bytes
    (here reused as the "bytes_l" denominator per spec: sort by U/bytes_l asc,
    add to skip set K while sum(U) <= eps * sum(L)).
    """
    total_L = bounds_r["L"].sum()
    ranked = bounds_r.assign(ratio=bounds_r["U"] / bounds_r["sketch_bytes"].clip(lower=1)).sort_values("ratio")
    K, cum_U = set(), 0.0
    for _, row in ranked.iterrows():
        if cum_U + row["U"] <= eps * total_L:
            K.add(row["layer"])
            cum_U += row["U"]
        else:
            break
    return K


def oracle_planner(scores_q: pd.DataFrame, eps: float) -> set:
    total_s = scores_q["s_l"].sum()
    ranked = scores_q.assign(ratio=scores_q["s_l"] / scores_q["bytes_l"]).sort_values("ratio")
    K, cum_s = set(), 0.0
    for _, row in ranked.iterrows():
        if cum_s + row["s_l"] <= eps * total_s:
            K.add(row["layer"])
            cum_s += row["s_l"]
        else:
            break
    return K


def main():
    p = argparse.ArgumentParser(description="E3: bound tightness + sound planner vs oracle")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--factors-h5", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--scores", required=True, help="scores.parquet, for the oracle comparison")
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--out-dir", default="results/layer_scores/e3")
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cpu")
    p.add_argument("--step", type=int, default=0)
    args = p.parse_args()

    import torch
    from data.cifar10 import cifar10_loaders
    from models.resnet18 import resnet18_cifar
    from experiments.layer_scores.ekfac_fit import resolve_device

    device = resolve_device(args.device)
    model = resnet18_cifar(num_classes=10).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    _, _, train_ds = cifar10_loaders(root=args.data_root, augment=False)
    specs = build_layer_specs(model)
    store = EKFACStore(args.factors_h5)
    eigs = {s.name: store.load_eigen(s.name, args.step) for s in specs}
    queries = load_queries(args.queries)
    scores_df = pd.read_parquet(args.scores)
    scores_df = scores_df[scores_df["alpha"] == args.alpha]

    bounds = compute_bounds_table(model, train_ds, queries, specs, eigs, args.alpha, device)
    bounds.to_parquet(f"{args.out_dir}/bounds.parquet")

    total_bytes_l = scores_df.drop_duplicates("layer").set_index("layer")["bytes_l"].sum()
    total_sketch_bytes_by_r = bounds.drop_duplicates(["layer", "r"]).groupby("r")["sketch_bytes"].sum()

    for violation_count, (query_id, r, eps) in enumerate(
        [(q.id, r, eps) for q in queries for r in R_LIST for eps in EPS_LIST]
    ):
        bounds_r = bounds[(bounds["query_id"] == query_id) & (bounds["r"] == r)]
        scores_q = scores_df[scores_df["query_id"] == query_id]
        K = sound_planner(bounds_r, eps)
        s_in_K = scores_q[scores_q["layer"].isin(K)]["s_l"].sum()
        total_s = scores_q["s_l"].sum()
        assert s_in_K <= eps * total_s + 1e-9, (
            f"SOUNDNESS VIOLATION: query={query_id} r={r} eps={eps} "
            f"sum(s in K)={s_in_K} > eps*sum(s)={eps*total_s}"
        )

    report_rows = []
    for eps in EPS_LIST:
        for r in R_LIST:
            sound_bytes, oracle_bytes = 0.0, 0.0
            for q in queries:
                bounds_r = bounds[(bounds["query_id"] == q.id) & (bounds["r"] == r)]
                scores_q = scores_df[scores_df["query_id"] == q.id]
                K_sound = sound_planner(bounds_r, eps)
                K_oracle = oracle_planner(scores_q, eps)
                bytes_l = scores_q.drop_duplicates("layer").set_index("layer")["bytes_l"]
                sound_bytes += bytes_l[~bytes_l.index.isin(K_sound)].sum()
                oracle_bytes += bytes_l[~bytes_l.index.isin(K_oracle)].sum()
            ratio = sound_bytes / oracle_bytes if oracle_bytes > 0 else float("nan")
            resident_frac = total_sketch_bytes_by_r.get(r, float("nan")) / total_bytes_l
            report_rows.append({"eps": eps, "r": r, "sound_oracle_ratio": ratio, "resident_sketch_frac": resident_frac})

    report = pd.DataFrame(report_rows)
    report.to_csv(f"{args.out_dir}/e3_summary.csv", index=False)
    print(report.to_string())
    print("soundness assertion: all (query, r, eps) combinations passed")


if __name__ == "__main__":
    main()
