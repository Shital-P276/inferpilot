"""
utility_lambda_sweep.py

Sweeps the utility formula's lambda (latency weight) across a range,
recomputing which strategy "wins" at each value, to find the crossover
point(s) where ml_router would become utility-optimal against the real,
live-measured costs from the core experiment.

This is the honest alternative to just picking new weights that make
ml_router win -- instead of choosing a destination, we show the whole
curve and let the reader see exactly how sensitive the conclusion is.

USAGE:
    python router/utility_lambda_sweep.py
"""

from pathlib import Path

import pandas as pd

SUMMARY_CSV = "router/core_experiment_summary.csv"

# Sweep lambda from near-0 (accuracy-only) to near-1 (latency-only).
# mu (resource weight) held fixed at the original training value -- only
# testing lambda sensitivity here since latency is the dimension actually
# in question. Sweep mu separately if resource weighting is also in doubt.
LAMBDA_RANGE = [round(x * 0.05, 2) for x in range(1, 20)]  # 0.05, 0.10, ..., 0.95
MU_RESOURCE = 0.3

TIER_GPU_MEMORY_MB = {"fast": 22.47, "balanced": 32.18, "heavy": 34.81}


def compute_resource_cost(row):
    return (
        row["pct_fast"] / 100 * TIER_GPU_MEMORY_MB["fast"]
        + row["pct_balanced"] / 100 * TIER_GPU_MEMORY_MB["balanced"]
        + row["pct_heavy"] / 100 * TIER_GPU_MEMORY_MB["heavy"]
    )


def main():
    df = pd.read_csv(SUMMARY_CSV)
    df["resource_cost_mb"] = df.apply(compute_resource_cost, axis=1)

    latency_norm_basis = df["avg_latency_ms"].max()
    resource_norm_basis = df["resource_cost_mb"].max()
    df["latency_norm"] = df["avg_latency_ms"] / latency_norm_basis
    df["resource_norm"] = df["resource_cost_mb"] / resource_norm_basis

    sweep_rows = []
    winners = []

    for lam in LAMBDA_RANGE:
        df["utility"] = df["accuracy"] - lam * df["latency_norm"] - MU_RESOURCE * df["resource_norm"]
        ranked = df.sort_values("utility", ascending=False).reset_index(drop=True)
        winner = ranked.iloc[0]["strategy"]
        winners.append(winner)

        row = {"lambda": lam, "winner": winner}
        for _, r in df.iterrows():
            row[f"utility_{r['strategy']}"] = round(r["utility"], 4)
        sweep_rows.append(row)

    sweep_df = pd.DataFrame(sweep_rows)
    print(sweep_df.to_string(index=False))

    out_path = Path("router/utility_lambda_sweep.csv")
    sweep_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Find crossover point(s) -- where the winner changes
    print("\n" + "=" * 60)
    print("CROSSOVER ANALYSIS")
    print("=" * 60)
    prev_winner = None
    for _, row in sweep_df.iterrows():
        if row["winner"] != prev_winner:
            print(f"  lambda={row['lambda']:.2f}: winner changes to '{row['winner']}'")
            prev_winner = row["winner"]

    if "ml_router" in winners:
        first_ml_router_win = sweep_df[sweep_df["winner"] == "ml_router"]["lambda"].min()
        print(f"\nml_router first becomes utility-optimal at lambda <= {first_ml_router_win:.2f}")
        print(f"Your original training lambda was 0.7 -- ml_router "
              f"{'DOES' if 0.7 <= first_ml_router_win else 'does NOT'} win at that weight.")
    else:
        print(f"\nml_router NEVER wins across the entire swept range "
              f"(lambda={LAMBDA_RANGE[0]} to {LAMBDA_RANGE[-1]}). "
              f"This is a strong statement: no reasonable latency-weighting "
              f"choice would have favored the router's real deployed behavior "
              f"over the simpler baselines, given the measured costs.")


if __name__ == "__main__":
    main()