"""
compute_strategy_utility.py

Applies the SAME utility formula used to generate the router's training
labels -- utility = accuracy - lambda*latency_norm - mu*resource_norm --
to the live, measured results from the 5-strategy core experiment
(core_experiment_summary.csv), so we can see how each strategy scores
against the actual objective the router was trained to optimize.

IMPORTANT CAVEAT, read before trusting the output:
    Training-time labels normalized latency against the TRAINING SET's
    p95 latency (per generate_utility_labels.py). This script normalizes
    against the max latency observed among these 5 live strategies instead,
    since we don't have that exact training-time scale here. This means:
      - The RANKING between strategies below is meaningful and trustworthy.
      - The ABSOLUTE utility numbers are NOT directly comparable to
        training-time utility values -- don't put them side by side in
        the report without this caveat.
    If you have the exact p95 latency value used in training (check
    generate_utility_labels.py or utility_sweep_summary.csv), set
    LATENCY_NORM_OVERRIDE below to that number for an apples-to-apples
    comparison instead.

USAGE:
    python router/compute_strategy_utility.py
"""

import json
from pathlib import Path

import pandas as pd

# ---- CONFIG -------------------------------------------------------------
SUMMARY_CSV = "router/core_experiment_summary.csv"

# Utility weights -- MUST match whatever was used in generate_utility_labels.py.
# Update these if your actual training weights differ.
LAMBDA_LATENCY = 0.7
MU_RESOURCE = 0.3

# Static per-tier GPU peak memory (MB), from benchmark_models.py's
# gpu_peak_inference_mb -- used as the resource_cost basis, weighted by
# each strategy's real tier distribution.
TIER_GPU_MEMORY_MB = {
    "fast": 22.47,
    "balanced": 32.18,
    "heavy": 34.81,
}

# Set to a specific ms value (e.g. from training data) to normalize latency
# against that instead of the max observed among these 5 strategies.
LATENCY_NORM_OVERRIDE = None
# ---------------------------------------------------------------------------


def compute_resource_cost(row):
    """Weighted average GPU memory footprint based on this strategy's
    actual observed tier distribution (pct_fast/pct_balanced/pct_heavy)."""
    weighted = (
        row["pct_fast"] / 100 * TIER_GPU_MEMORY_MB["fast"]
        + row["pct_balanced"] / 100 * TIER_GPU_MEMORY_MB["balanced"]
        + row["pct_heavy"] / 100 * TIER_GPU_MEMORY_MB["heavy"]
    )
    return weighted


def main():
    df = pd.read_csv(SUMMARY_CSV)

    required_cols = {"strategy", "accuracy", "avg_latency_ms", "pct_fast", "pct_balanced", "pct_heavy"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{SUMMARY_CSV} is missing expected columns: {missing}. "
            f"Got columns: {list(df.columns)}. Adjust this script to match your actual CSV."
        )

    df["resource_cost_mb"] = df.apply(compute_resource_cost, axis=1)

    latency_norm_basis = LATENCY_NORM_OVERRIDE if LATENCY_NORM_OVERRIDE else df["avg_latency_ms"].max()
    resource_norm_basis = df["resource_cost_mb"].max()

    print(f"Latency normalized against: {latency_norm_basis:.2f}ms "
          f"({'OVERRIDE value' if LATENCY_NORM_OVERRIDE else 'max observed among these strategies'})")
    print(f"Resource normalized against: {resource_norm_basis:.2f}MB (max observed)")
    print(f"Weights: lambda(latency)={LAMBDA_LATENCY}, mu(resource)={MU_RESOURCE}\n")

    df["latency_norm"] = df["avg_latency_ms"] / latency_norm_basis
    df["resource_norm"] = df["resource_cost_mb"] / resource_norm_basis

    df["utility"] = (
        df["accuracy"]
        - LAMBDA_LATENCY * df["latency_norm"]
        - MU_RESOURCE * df["resource_norm"]
    )

    result = df[[
        "strategy", "accuracy", "avg_latency_ms", "resource_cost_mb",
        "latency_norm", "resource_norm", "utility"
    ]].sort_values("utility", ascending=False).reset_index(drop=True)

    result["accuracy"] = result["accuracy"].round(4)
    result["avg_latency_ms"] = result["avg_latency_ms"].round(2)
    result["resource_cost_mb"] = result["resource_cost_mb"].round(2)
    result["latency_norm"] = result["latency_norm"].round(4)
    result["resource_norm"] = result["resource_norm"].round(4)
    result["utility"] = result["utility"].round(4)

    print(result.to_string(index=False))

    out_path = Path("router/strategy_utility_comparison.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    best = result.iloc[0]
    print(f"\nBy this utility function, the highest-scoring strategy is: {best['strategy']} "
          f"(utility={best['utility']:.4f})")
    print("If this does NOT match ml_router, that's the core finding worth writing up: "
          "the router's own training objective, evaluated against its real deployed "
          "latency, favors a different strategy than the router itself.")


if __name__ == "__main__":
    main()
