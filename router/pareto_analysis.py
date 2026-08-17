"""
pareto_analysis.py

Pareto-frontier analysis of the core-experiment strategies on accuracy vs.
latency. Pure re-analysis of existing results -- no retraining, no new
experiments, no modification of any prior formula/training/gateway code.

Replaces the single-scalar utility formula as the comparison lens per the
methodology decision: cost is represented as an accuracy-vs-latency tradeoff
curve, NOT a weighted scalar that blends latency with memory/params.

Inputs:
  router/core_experiment_summary.csv  -- 7 strategies (accuracy, avg latency,
                                         p95 latency, tier distribution %)
  router/model_benchmarks.json        -- per-tier fast/balanced/heavy
                                         (avg latency, gpu_peak_inference_mb,
                                         num_params)

Output:
  Two separately-labeled stdout tables (strategy comparison + tier reference,
  explicitly NOT cross-comparable) and router/pareto_frontier.png (two-panel
  matplotlib plot, one panel per analysis).

Standalone tier accuracies are NOT stored as a file in this repo (the training
scripts log test_acc to MLflow only), so the known Section-3 values are used as
literals: fast=0.8812, balanced=0.9819, heavy=0.9850.

MEASUREMENT-BASIS CAVEAT:
  Strategy latency comes from the live core experiment (end-to-end gateway ms);
  tier latency comes from model_benchmarks.json (single-image CUDA inference).
  These two latency scales are NOT comparable, so Pareto dominance is computed
  SEPARATELY within each group and never across the two. The two results must
  be read as independent analyses.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SUMMARY_CSV = BASE_DIR / "router" / "core_experiment_summary.csv"
BENCHMARKS_JSON = BASE_DIR / "router" / "model_benchmarks.json"
OUT_PNG = BASE_DIR / "router" / "pareto_frontier.png"

# Standalone per-tier test accuracy (Section 3 known values; not stored as a
# file in this repo -- MLflow only).
TIER_ACCURACY = {"fast": 0.8812, "balanced": 0.9819, "heavy": 0.9850}
TIER_NAMES = ["fast", "balanced", "heavy"]

# Reference GPU budget assumption for the packing note (informational only,
# NOT used in the Pareto calculation).
GPU_BUDGET_MB = 8000


def load_summary():
    df = pd.read_csv(SUMMARY_CSV)
    print(f"Loaded {len(df)} strategies from {SUMMARY_CSV.name}:")
    print(f"  columns: {list(df.columns)}")
    return df


def load_benchmarks():
    with open(BENCHMARKS_JSON) as f:
        data = json.load(f)
    benchmarks = {entry["tier"]: entry for entry in data}
    print(f"Loaded {len(benchmarks)} tiers from {BENCHMARKS_JSON.name}:")
    for tier, e in benchmarks.items():
        print(f"  {tier:9s} avg_latency_ms={e['avg_latency_ms']:.3f} "
              f"gpu_peak_inference_mb={e['gpu_peak_inference_mb']:.2f}")
    return benchmarks


def strategy_gpu_peak_mb(row, benchmarks):
    """Weighted-average GPU peak memory for a strategy from its tier mix."""
    return (
        row["pct_fast"] / 100 * benchmarks["fast"]["gpu_peak_inference_mb"]
        + row["pct_balanced"] / 100 * benchmarks["balanced"]["gpu_peak_inference_mb"]
        + row["pct_heavy"] / 100 * benchmarks["heavy"]["gpu_peak_inference_mb"]
    )


def packing_note(gpu_peak_mb):
    """Feasibility label: how many replicas fit in GPU_BUDGET_MB (informational)."""
    n = int(GPU_BUDGET_MB // gpu_peak_mb) if gpu_peak_mb > 0 else 0
    return f"~{n} replica(s) in {GPU_BUDGET_MB}MB budget"


def pareto_status(points):
    """
    A point is dominated if some OTHER point (within this same group) has >=
    accuracy AND <= latency, with at least one strict inequality. Returns a
    list of booleans (True = Pareto-optimal).
    """
    status = []
    for i, (acc_i, lat_i) in enumerate(points):
        dominated = False
        for j, (acc_j, lat_j) in enumerate(points):
            if j == i:
                continue
            if acc_j >= acc_i and lat_j <= lat_i and (acc_j > acc_i or lat_j < lat_i):
                dominated = True
                break
        status.append(not dominated)
    return status


def print_table(title, basis, rows):
    """rows: list of (name, accuracy, latency_ms, is_pareto, packing_note)."""
    print("-" * 70)
    print(title)
    print(f"  Measurement basis: {basis}")
    print("-" * 70)
    header = f"{'point':26s} {'accuracy':>9s} {'latency_ms':>10s} {'pareto_status':>14s}  packing_note"
    print(header)
    print("-" * 70)
    for name, acc, lat, pareto, note in rows:
        status = "Pareto-optimal" if pareto else "dominated"
        print(f"{name:26s} {acc:9.4f} {lat:10.3f} {status:>14s}  {note}")
    print()


def draw_panel(ax, title, names, points, is_pareto, color, marker):
    """Scatter one analysis group on the given axis, with its own step-line."""
    for name, (acc, lat), pareto in zip(names, points, is_pareto):
        ax.scatter(lat, acc, s=110, marker=marker, color=color if pareto else "none",
                   edgecolors=color, label=None)
        ax.annotate(name, (lat, acc), textcoords="offset points", xytext=(8, 6), fontsize=9)

    pareto_pts = sorted([(n, a, l) for n, (a, l), p in zip(names, points, is_pareto) if p],
                        key=lambda x: x[2])  # ascending latency
    if len(pareto_pts) >= 2:
        xs = [p[2] for p in pareto_pts]
        ys = [p[1] for p in pareto_pts]
        step_x, step_y = [], []
        prev = None
        for x, y in zip(xs, ys):
            if prev is None:
                step_x.append(x)
                step_y.append(y)
            else:
                step_x.append(prev)
                step_y.append(y)  # horizontal
                step_x.append(x)
                step_y.append(y)  # vertical
            prev = x
        ax.plot(step_x, step_y, color="#2ca02c", linewidth=2, linestyle="--", alpha=0.8,
                label="Pareto frontier (step)")

    ax.set_xlabel("avg latency (ms)")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")


def main():
    print("=" * 70)
    print("PARETO-FRONTIER ANALYSIS: accuracy vs. latency")
    print("=" * 70)
    print(f"GPU packing assumption: reference budget = {GPU_BUDGET_MB}MB "
          f"(informational only, not part of the Pareto calc).\n")

    df = load_summary()
    benchmarks = load_benchmarks()

    # ---- Build the two independent point sets ----------------------------
    strat_names = list(df["strategy"])
    strat_points = [(row["accuracy"], row["avg_latency_ms"]) for _, row in df.iterrows()]
    strat_pareto = pareto_status(strat_points)

    tier_names = list(TIER_NAMES)
    tier_points = [(TIER_ACCURACY[t], benchmarks[t]["avg_latency_ms"]) for t in TIER_NAMES]
    tier_pareto = pareto_status(tier_points)

    # ---- Memory footprint (informational, per group) ----------------------
    strat_packing = []
    for name in strat_names:
        row = df[df["strategy"] == name].iloc[0]
        strat_packing.append(packing_note(strategy_gpu_peak_mb(row, benchmarks)))

    tier_packing = [packing_note(benchmarks[t]["gpu_peak_inference_mb"]) for t in TIER_NAMES]

    # ---- Print the two separate tables ------------------------------------
    print_table(
        "TABLE A -- STRATEGY COMPARISON (apples-to-apples, gateway-measured)",
        "end-to-end gateway latency (core_experiment_summary.csv)",
        list(zip(strat_names, [p[0] for p in strat_points], [p[1] for p in strat_points],
                 strat_pareto, strat_packing)),
    )

    print("!" * 70)
    print("! WARNING: latency values below use a DIFFERENT measurement basis")
    print("! (isolated single-image CUDA benchmark, not end-to-end gateway ms).")
    print("! The two tables are NOT comparable -- do not compare or blend points")
    print("! across TABLE A and TABLE B. Each table is its own Pareto analysis.")
    print("!" * 70)

    print_table(
        "TABLE B -- STANDALONE TIER REFERENCE (isolated benchmark, NOT comparable to TABLE A)",
        "isolated single-image CUDA inference latency (model_benchmarks.json)",
        list(zip(tier_names, [p[0] for p in tier_points], [p[1] for p in tier_points],
                 tier_pareto, tier_packing)),
    )

    # ---- Two-panel plot ------------------------------------------------------
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 7))

    draw_panel(
        ax_a,
        "Strategy comparison (gateway-measured, apples-to-apples)",
        strat_names, strat_points, strat_pareto,
        color="#1f77b4", marker="o",
    )
    draw_panel(
        ax_b,
        "Standalone tier reference (isolated CUDA benchmark -- not directly comparable to Panel A)",
        tier_names, tier_points, tier_pareto,
        color="#d62728", marker="s",
    )

    fig.suptitle("Pareto frontier (filled = Pareto-optimal, hollow = dominated)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print("-" * 70)
    print(f"Two-panel plot saved to {OUT_PNG}")


if __name__ == "__main__":
    main()