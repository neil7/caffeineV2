"""
experiments.layer_scores.e2_variation
========================================
E2: hot-set overlap (Jaccard) within/across query types, static-pinning
baseline (train/test split + leave-one-type-out).
"""

import argparse
from itertools import combinations
from typing import Dict, List

import numpy as np
import pandas as pd

from experiments.layer_scores.e1_concentration import greedy_hot_set


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def hot_sets_by_query(df: pd.DataFrame, alpha: float, c: float) -> Dict[str, set]:
    sub = df[df["alpha"] == alpha]
    return {qid: set(greedy_hot_set(g, c)) for qid, g in sub.groupby("query_id")}


def pairwise_jaccard(hot_sets: Dict[str, set], query_type_of: Dict[str, str]) -> pd.DataFrame:
    rows = []
    ids = list(hot_sets.keys())
    for i, j in combinations(ids, 2):
        rows.append({
            "q1": i, "q2": j, "type1": query_type_of[i], "type2": query_type_of[j],
            "jaccard": jaccard(hot_sets[i], hot_sets[j]),
        })
    return pd.DataFrame(rows)


def static_pinning_baseline(
    df: pd.DataFrame, alpha: float, c: float, seed: int = 17
) -> Dict[str, float]:
    sub = df[df["alpha"] == alpha]
    query_ids = sub["query_id"].unique()
    query_type = sub.drop_duplicates("query_id").set_index("query_id")["query_type"]

    rng = np.random.default_rng(seed)
    train_ids, test_ids = [], []
    for qtype, grp in query_type.groupby(query_type):
        ids = grp.index.to_numpy()
        rng.shuffle(ids)
        half = len(ids) // 2
        train_ids += ids[:half].tolist()
        test_ids += ids[half:].tolist()

    layers = sub["layer"].unique()
    bytes_l = sub.drop_duplicates("layer").set_index("layer")["bytes_l"]

    train = sub[sub["query_id"].isin(train_ids)]
    mean_share = train.groupby("layer")["s_share"].mean()
    ranked = (mean_share / bytes_l.reindex(mean_share.index)).sort_values(ascending=False)

    S_static, cum = [], 0.0
    frac_reaching = 0.0
    for frac_target in [0.95]:
        for layer in ranked.index:
            S_static.append(layer)
            covered = train[train["layer"].isin(S_static)].groupby("query_id")["s_share"].sum()
            frac_reaching = float((covered >= c).mean())
            if frac_reaching >= frac_target:
                break

    S_static_set = set(S_static)
    bytes_static = bytes_l.loc[list(S_static_set)].sum()

    test = sub[sub["query_id"].isin(test_ids)]
    covered_test = test[test["layer"].isin(S_static_set)].groupby("query_id")["s_share"].sum()
    frac_test_reach = float((covered_test >= c).mean())

    from experiments.layer_scores.e1_concentration import bytes_q
    per_query_bytes = test.groupby("query_id").apply(lambda g: bytes_q(g, c))
    mean_per_query_bytes = float(per_query_bytes.mean()) * bytes_l.sum()
    R = bytes_static / mean_per_query_bytes if mean_per_query_bytes > 0 else float("nan")

    return {
        "bytes_static": float(bytes_static),
        "S_static": sorted(S_static_set),
        "frac_test_reach_c": frac_test_reach,
        "R": R,
    }


def leave_one_type_out(df: pd.DataFrame, alpha: float, c: float) -> pd.DataFrame:
    sub = df[df["alpha"] == alpha]
    types = sub["query_type"].unique()
    rows = []
    for held_out in types:
        train = sub[sub["query_type"] != held_out]
        test = sub[sub["query_type"] == held_out]
        bytes_l = sub.drop_duplicates("layer").set_index("layer")["bytes_l"]
        mean_share = train.groupby("layer")["s_share"].mean()
        ranked = (mean_share / bytes_l.reindex(mean_share.index)).sort_values(ascending=False)

        S, covered_frac = [], 0.0
        for layer in ranked.index:
            S.append(layer)
            covered = train[train["layer"].isin(S)].groupby("query_id")["s_share"].sum()
            if float((covered >= c).mean()) >= 0.95:
                break
        S_set = set(S)
        bytes_static = bytes_l.loc[list(S_set)].sum()
        covered_test = test[test["layer"].isin(S_set)].groupby("query_id")["s_share"].sum()
        frac_test_reach = float((covered_test >= c).mean()) if len(covered_test) else float("nan")

        from experiments.layer_scores.e1_concentration import bytes_q
        per_query_bytes = test.groupby("query_id").apply(lambda g: bytes_q(g, c))
        mean_per_query_bytes = float(per_query_bytes.mean()) * bytes_l.sum() if len(per_query_bytes) else float("nan")
        R = bytes_static / mean_per_query_bytes if mean_per_query_bytes and mean_per_query_bytes > 0 else float("nan")

        rows.append({"held_out_type": held_out, "frac_test_reach_c": frac_test_reach, "R": R})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="E2: hot-set overlap + static-pinning baseline")
    p.add_argument("--scores", required=True)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--c", type=float, default=0.95)
    p.add_argument("--out-dir", default="results/layer_scores/e2")
    args = p.parse_args()

    df = pd.read_parquet(args.scores)
    hot_sets = hot_sets_by_query(df, args.alpha, args.c)
    query_type_of = df.drop_duplicates("query_id").set_index("query_id")["query_type"].to_dict()
    jac = pairwise_jaccard(hot_sets, query_type_of)
    jac.to_parquet(f"{args.out_dir}/jaccard.parquet")

    baseline = static_pinning_baseline(df, args.alpha, args.c)
    print("static pinning baseline:", {k: v for k, v in baseline.items() if k != "S_static"})

    loto = leave_one_type_out(df, args.alpha, args.c)
    loto.to_csv(f"{args.out_dir}/leave_one_type_out.csv", index=False)
    print(loto.to_string())


if __name__ == "__main__":
    main()
