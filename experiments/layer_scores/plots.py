"""
experiments.layer_scores.plots
=================================
E3a-E3d figures (E1a-E1d live in e1_concentration.make_figures, next to the
tables they're derived from).
"""

import argparse
import os

import pandas as pd


def make_e3_figures(bounds: pd.DataFrame, summary: pd.DataFrame, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots()
    for layer, g in bounds.groupby("layer"):
        med = g.groupby("r")["U_over_L"].median()
        ax.plot(med.index, med.values, marker="o", markersize=2, alpha=0.5)
    ax.set_yscale("log"); ax.set_xlabel("r"); ax.set_ylabel("median U/L")
    ax.set_title("E3a: bound tightness vs r")
    fig.savefig(f"{out_dir}/e3a_U_over_L.pdf"); plt.close(fig)

    fig, ax = plt.subplots()
    for layer, g in bounds.groupby("layer"):
        med = g.groupby("r")["m_out_frac"].median()
        ax.plot(med.index, med.values, marker="o", markersize=2, alpha=0.5)
    ax.set_xlabel("r"); ax.set_ylabel("median m_out fraction")
    ax.set_title("E3b: out-of-block mass vs r")
    fig.savefig(f"{out_dir}/e3b_m_out.pdf"); plt.close(fig)

    fig, ax = plt.subplots()
    for eps, g in summary.groupby("eps"):
        g = g.sort_values("r")
        ax.plot(g["r"], g["sound_oracle_ratio"], marker="o", label=f"eps={eps}")
    ax.set_xlabel("r"); ax.set_ylabel("sound/oracle bytes ratio"); ax.legend()
    ax.set_title("E3c: sound vs oracle bytes")
    fig.savefig(f"{out_dir}/e3c_sound_oracle.pdf"); plt.close(fig)

    fig, ax = plt.subplots()
    resident = summary.drop_duplicates("r").sort_values("r")
    ax.plot(resident["r"], resident["resident_sketch_frac"], marker="o")
    ax.set_xlabel("r"); ax.set_ylabel("resident sketch bytes / total bytes")
    ax.set_title("E3d: resident sketch size vs r")
    fig.savefig(f"{out_dir}/e3d_resident_size.pdf"); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Render E3 figures from bounds.parquet + e3_summary.csv")
    p.add_argument("--bounds", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--out-dir", default="results/layer_scores/e3/figures")
    args = p.parse_args()

    bounds = pd.read_parquet(args.bounds)
    summary = pd.read_csv(args.summary)
    make_e3_figures(bounds, summary, args.out_dir)


if __name__ == "__main__":
    main()
