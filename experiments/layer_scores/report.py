"""
experiments.layer_scores.report
==================================
Assemble REPORT.md from a completed run's result files. Every number in the
report is read from a file in the run's results directory (traceability,
Ground Rule #4) — nothing here recomputes a statistic from memory.
"""

import argparse
import os

import pandas as pd

T_CONC = 0.30  # TODO(neil): confirm
T_VAR = 1.5    # TODO(neil): confirm

TODO_LIST = [
    "ResNet checkpoint path / training run used (fill in from config.yaml).",
    "N_corr default for Lambda fitting (spec proposes 10k, unconfirmed).",
    "Damping alpha default (spec proposes 0.1 x mean(Lambda), attributed to Grosse et al., unconfirmed).",
    "T_conc (0.30) and T_var (1.5) quadrant thresholds, unconfirmed.",
    "SLURM partition, account, GPU type, conda env, data path.",
    "Whether true memorization (leave-out retraining) is wanted for query type M, "
    "beyond the per-example-loss proxy implemented.",
]


def classify_quadrant(median_bytes_q95: float, r_test: float) -> str:
    concentrated = median_bytes_q95 <= T_CONC
    variable = r_test >= T_VAR
    if concentrated and variable:
        return "concentrated, variable"
    if concentrated and not variable:
        return "concentrated, static"
    if not concentrated and variable:
        return "diffuse, variable"
    return "diffuse, static"


def build_report(run_dir: str) -> str:
    e1_summary = pd.read_csv(os.path.join(run_dir, "e1", "e1_summary.csv"))
    median_bytes_q95 = float(
        e1_summary.loc[e1_summary["c"] == 0.95, "bytes_q_median"].median()
    )

    loto_path = os.path.join(run_dir, "e2", "leave_one_type_out.csv")
    r_test = float("nan")
    if os.path.exists(loto_path):
        loto = pd.read_csv(loto_path)
        r_test = float(loto["R"].median())

    quadrant = classify_quadrant(median_bytes_q95, r_test)

    lines = [
        f"# Layer-scores report — {run_dir}",
        "",
        "## E1 summary (median/p90 of k_q, bytes_q per type x c)",
        "",
        "```",
        e1_summary.to_string(index=False),
        "```",
        "",
        "## Figures",
        "",
        "- figures/e1a_k_cdf.pdf, e1b_bytes_cdf.pdf, e1c_heatmap.pdf, e1d_fc_share.pdf",
        "- e3/figures/e3a_U_over_L.pdf, e3b_m_out.pdf, e3c_sound_oracle.pdf, e3d_resident_size.pdf",
        "",
        "## Quadrant classification",
        "",
        f"- median bytes_q(0.95) = {median_bytes_q95:.4f} (T_conc = {T_CONC})",
        f"- R(0.95) held-out median = {r_test:.4f} (T_var = {T_VAR})",
        f"- **Quadrant: ({quadrant})**",
        "",
        "## TODO(neil)",
        "",
        *[f"- {t}" for t in TODO_LIST],
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Assemble REPORT.md from a layer-scores run directory")
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()

    report = build_report(args.run_dir)
    out_path = os.path.join(args.run_dir, "REPORT.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
