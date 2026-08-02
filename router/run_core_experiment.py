"""
Core experiment: compares all 5 routing strategies (Always Fast, Always Heavy,
Round Robin, Rule-Based, ML Router) on ACCURACY using the held-out image set.

Sends each held-out image through /route?strategy=X for every strategy,
records routed_tier, predicted_class vs true_label, and gateway_latency_ms.

This measures per-strategy accuracy under REAL SEQUENTIAL conditions (no
concurrency) -- see monitoring/load_test_strategies.js for the separate
concurrent-load latency/throughput comparison. Together these two scripts
produce the full comparison table required by the project plan.

Output: router/core_experiment_results.csv, router/core_experiment_summary.csv
"""
import time
from pathlib import Path

import pandas as pd
import httpx
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
HELD_OUT_PATH = BASE_DIR / "router" / "held_out_images.csv"
RESULTS_PATH = BASE_DIR / "router" / "core_experiment_raw_results.csv"
SUMMARY_PATH = BASE_DIR / "router" / "core_experiment_summary.csv"

GATEWAY_URL = "http://localhost:8000/route"
STRATEGIES = ["always_fast", "always_heavy", "round_robin", "rule_based", "ml_router"]

TIMEOUT_S = 15.0


def load_held_out():
    df = pd.read_csv(HELD_OUT_PATH)
    print(f"Loaded {len(df)} held-out images")
    return df


def run_strategy(strategy: str, held_out_df: pd.DataFrame, client: httpx.Client):
    rows = []
    for _, row in tqdm(held_out_df.iterrows(), total=len(held_out_df), desc=strategy):
        image_path = BASE_DIR / row["image_path"]
        true_label = row["true_label"]

        try:
            with open(image_path, "rb") as f:
                resp = client.post(
                    GATEWAY_URL,
                    params={"strategy": strategy},
                    files={"file": (image_path.name, f, "image/jpeg")},
                    timeout=TIMEOUT_S,
                )
            resp.raise_for_status()
            result = resp.json()

            predicted_class = result.get("predicted_class")
            correct = int(predicted_class == true_label)

            rows.append({
                "strategy": strategy,
                "image_path": row["image_path"],
                "true_label": true_label,
                "predicted_class": predicted_class,
                "correct": correct,
                "routed_tier": result.get("routed_tier"),
                "gateway_latency_ms": result.get("gateway_latency_ms"),
            })
        except Exception as e:
            print(f"  ERROR on {image_path.name} ({strategy}): {e}")
            rows.append({
                "strategy": strategy,
                "image_path": row["image_path"],
                "true_label": true_label,
                "predicted_class": None,
                "correct": 0,
                "routed_tier": None,
                "gateway_latency_ms": None,
            })

    return rows


def main():
    held_out_df = load_held_out()
    all_rows = []

    with httpx.Client() as client:
        for strategy in STRATEGIES:
            print(f"\n{'='*60}\nRunning strategy: {strategy}\n{'='*60}")
            t0 = time.time()
            rows = run_strategy(strategy, held_out_df, client)
            elapsed = time.time() - t0
            print(f"  Completed {len(rows)} requests in {elapsed:.1f}s "
                  f"({len(rows)/elapsed:.2f} req/s sequential)")
            all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nRaw results written to {RESULTS_PATH}")

    # ---- Build summary table ----
    summary_rows = []
    for strategy in STRATEGIES:
        strat_df = results_df[results_df["strategy"] == strategy]
        valid = strat_df[strat_df["gateway_latency_ms"].notna()]

        tier_counts = strat_df["routed_tier"].value_counts(normalize=True) * 100

        summary_rows.append({
            "strategy": strategy,
            "n_requests": len(strat_df),
            "accuracy": round(strat_df["correct"].mean(), 4),
            "avg_latency_ms": round(valid["gateway_latency_ms"].mean(), 2) if len(valid) else None,
            "p95_latency_ms": round(valid["gateway_latency_ms"].quantile(0.95), 2) if len(valid) else None,
            "pct_fast": round(tier_counts.get("fast", 0), 1),
            "pct_balanced": round(tier_counts.get("balanced", 0), 1),
            "pct_heavy": round(tier_counts.get("heavy", 0), 1),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"\n{'='*60}\nSUMMARY (sequential, no concurrency)\n{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\nWritten to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()