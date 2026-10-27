"""
experiments.layer_scores.e1_concentration
============================================
E1: for each query, how many layers / bytes are needed to capture
90/95/99% of the update. Figures E1a-E1d + median/p90 table.
"""

import argparse
from typing import List

import numpy as np
import pandas as pd


def k_q(layer_s: pd.Series, c: float) -> int:
    """# layers needed, sorted by s_l descending, to reach coverage c."""
    s_sorted = layer_s.sort_values(ascending=False).to_numpy()
    total = s_sorted.sum()
    if total <= 0:
        return len(s_sorted)
    cum = np.cumsum(s_sorted) / total
    return int(np.searchsorted(cum, c) + 1)


def bytes_q(df_query: pd.DataFrame, c: float) -> float:
    """
    Greedily add layers by s_l/bytes_l until coverage c reached; return the
    result as a fraction of total bytes.
    """
    total_s = df_query["s_l"].sum()
    total_bytes = df_query["bytes_l"].sum()
    if total_s <= 0 or total_bytes <= 0:
        return 1.0
    ranked = df_query.assign(ratio=df_query["s_l"] / df_query["bytes_l"]).sort_values(
        "ratio", ascending=False
    )
    cum_s, cum_bytes = 0.0, 0.0
    for _, row in ranked.iterrows():
        cum_s += row["s_l"]
        cum_bytes += row["bytes_l"]
        if cum_s / total_s >= c:
            break
    return cum_bytes / total_bytes


def greedy_hot_set(df_query: pd.DataFrame, c: float) -> List[str]:
    """Same greedy byte-ordered selection as bytes_q, but returns the layer-name set."""
    total_s = df_query["s_l"].sum()
    if total_s <= 0:
        return df_query["layer"].tolist()
    ranked = df_query.assign(ratio=df_query["s_l"] / df_query["bytes_l"]).sort_values(
        "ratio", ascending=False
    )
    chosen, cum_s = [], 0.0
    for _, row in ranked.iterrows():
        chosen.append(row["layer"])
        cum_s += row["s_l"]
        if cum_s / total_s >= c:
            break
    return chosen


def compute_e1_tables(df: pd.DataFrame, alpha: float, coverages=(0.90, 0.95, 0.99)) -> pd.DataFrame:
    sub = df[df["alpha"] == alpha]
    rows = []
    for (qid, qtype), g in sub.groupby(["query_id", "query_type"]):
        for c in coverages:
            rows.append({
                "query_id": qid, "query_type": qtype, "c": c,
                "k_q": k_q(g.set_index("layer")["s_l"], c),
                "bytes_q": bytes_q(g, c),
                "fc_share": float(g.loc[g["layer"] == "fc", "s_share"].sum()),
                "bn_share": float(g["bn_grad_norm_share"].iloc[0]),
            })
    return pd.DataFrame(rows)


def summary_table(e1: pd.DataFrame) -> pd.DataFrame:
    return (
        e1.groupby(["query_type", "c"])
        .agg(k_q_median=("k_q", "median"), k_q_p90=("k_q", lambda x: x.quantile(0.9)),
             bytes_q_median=("bytes_q", "median"), bytes_q_p90=("bytes_q", lambda x: x.quantile(0.9)))
        .reset_index()
    )


def make_figures(e1: pd.DataFrame, df: pd.DataFrame, alpha: float, out_dir: str) -> None:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots()
    for qtype, g in e1[e1["c"] == 0.95].groupby("query_type"):
        vals = np.sort(g["k_q"].to_numpy())
        ax.plot(vals, np.linspace(0, 1, len(vals)), label=qtype)
    ax.set_xlabel("k_q(0.95)"); ax.set_ylabel("CDF"); ax.legend(); ax.set_title("E1a: layer count CDF")
    fig.savefig(f"{out_dir}/e1a_k_cdf.pdf"); plt.close(fig)

    fig, ax = plt.subplots()
    for qtype, g in e1[e1["c"] == 0.95].groupby("query_type"):
        vals = np.sort(g["bytes_q"].to_numpy())
        ax.plot(vals, np.linspace(0, 1, len(vals)), label=qtype)
    ax.set_xlabel("bytes_q(0.95)"); ax.set_ylabel("CDF"); ax.legend(); ax.set_title("E1b: bytes fraction CDF")
    fig.savefig(f"{out_dir}/e1b_bytes_cdf.pdf"); plt.close(fig)

    sub = df[df["alpha"] == alpha].copy()
    sub["query_type"] = sub["query_id"].map(e1.drop_duplicates("query_id").set_index("query_id")["query_type"])
    pivot = sub.pivot_table(index="query_id", columns="layer", values="s_share", fill_value=0.0)
    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.1)))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=90, fontsize=5)
    ax.set_title("E1c: s_l / sum(s) heatmap")
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(f"{out_dir}/e1c_heatmap.pdf"); plt.close(fig)

    fig, ax = plt.subplots()
    e1.boxplot(column="fc_share", by="query_type", ax=ax)
    ax.set_title("E1d: fc share by query type"); plt.suptitle("")
    fig.savefig(f"{out_dir}/e1d_fc_share.pdf"); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="E1 concentration: how many layers/bytes per query")
    p.add_argument("--scores", required=True)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--out-dir", default="results/layer_scores/e1")
    args = p.parse_args()

    df = pd.read_parquet(args.scores)
    e1 = compute_e1_tables(df, args.alpha)
    e1.to_parquet(f"{args.out_dir}/e1_per_query.parquet")
    summary = summary_table(e1)
    summary.to_csv(f"{args.out_dir}/e1_summary.csv", index=False)
    print(summary.to_string())
    make_figures(e1, df, args.alpha, f"{args.out_dir}/figures")


if __name__ == "__main__":
    main()
