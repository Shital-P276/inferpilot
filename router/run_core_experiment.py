"""
run_core_experiment.py

The core research experiment: runs all 5 routing strategies (ml_router,
always_fast, always_heavy, round_robin, rule_based) against the SAME
held-out test set, through the SAME live gateway, and produces the
comparison table that's the actual research result of this project.

BEFORE RUNNING:
    1. Gateway must be up: docker compose up -d (or however you run it),
       reachable at GATEWAY_URL below.
    2. HELD_OUT_CSV must point at your held_out_images.csv (the 250 images
       never touched during router training/tuning -- using anything else
       here would contaminate the comparison).
    3. That CSV must have a column with the image path and a column with
       the TRUE label (ground truth), so accuracy can actually be computed.
       Adjust IMAGE_PATH_COL / TRUE_LABEL_COL below if your column names differ.

USAGE:
    python router/run_core_experiment.py

OUTPUT:
    - router/core_experiment_raw_results.csv   (every single request logged)
    - router/core_experiment_summary.csv       (the actual comparison table)
    - Prints the summary table to stdout too
"""

import time
import csv
from pathlib import Path

import httpx
import pandas as pd

# ---- CONFIG -------------------------------------------------------------
GATEWAY_URL = "http://localhost:8000/route"
HELD_OUT_CSV = "router/held_out_images.csv"
IMAGE_PATH_COL = "image_path"   # <-- adjust to match your actual CSV column name
TRUE_LABEL_COL = "true_label"   # <-- adjust to match your actual CSV column name

STRATEGIES = ["ml_router", "always_fast", "always_heavy", "round_robin", "rule_based"]

REQUEST_TIMEOUT = 30.0
# ---------------------------------------------------------------------------


def load_held_out_set():
    df = pd.read_csv(HELD_OUT_CSV)
    if IMAGE_PATH_COL not in df.columns or TRUE_LABEL_COL not in df.columns:
        raise ValueError(
            f"Expected columns '{IMAGE_PATH_COL}' and '{TRUE_LABEL_COL}' in {HELD_OUT_CSV}, "
            f"but got columns: {list(df.columns)}. Fix IMAGE_PATH_COL/TRUE_LABEL_COL at the "
            f"top of this script to match your actual CSV."
        )
    return df


def call_gateway(client: httpx.Client, image_path: str, strategy: str):
    path = Path(image_path)
    if not path.exists():
        return {"error": f"image not found: {image_path}"}

    with open(path, "rb") as f:
        image_bytes = f.read()

    try:
        resp = client.post(
            GATEWAY_URL,
            params={"strategy": strategy},
            files={"file": (path.name, image_bytes, "image/jpeg")},
            timeout=REQUEST_TIMEOUT,
        )
    except httpx.RequestError as e:
        return {"error": f"request failed: {e}"}

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

    return resp.json()


def warm_up_services(client: httpx.Client, held_out: pd.DataFrame):
    """Send one throwaway request per tier before timing anything, so CUDA/
    cuDNN warm-up (which can take 10-20+ seconds on a freshly built/restarted
    container) doesn't eat into the first real request's latency, and doesn't
    make the script look 'stuck' with no output during that window."""
    print("Warming up services (this can take 10-30s on freshly rebuilt containers)...")
    sample_path = held_out.iloc[0][IMAGE_PATH_COL]
    path = Path(sample_path)
    if not path.exists():
        print(f"  WARNING: warm-up image not found at {sample_path}, skipping warm-up.")
        return
    with open(path, "rb") as f:
        image_bytes = f.read()

    for strategy in STRATEGIES:
        print(f"  warming up strategy={strategy} ...", end=" ", flush=True)
        try:
            resp = client.post(
                GATEWAY_URL,
                params={"strategy": strategy},
                files={"file": (path.name, image_bytes, "image/jpeg")},
                timeout=60.0,  # generous timeout, cold start can be slow
            )
            print(f"done ({resp.status_code})")
        except httpx.RequestError as e:
            print(f"failed ({e}) -- continuing anyway, real run may hit the same cold-start cost")
    print("Warm-up complete.\n")


def run_experiment():
    held_out = load_held_out_set()
    print(f"Loaded {len(held_out)} held-out images.")
    print(f"Running {len(STRATEGIES)} strategies x {len(held_out)} images "
          f"= {len(STRATEGIES) * len(held_out)} total requests. This will take a while.")

    raw_rows = []

    with httpx.Client() as client:
        warm_up_services(client, held_out)

        for strategy in STRATEGIES:
            print(f"\n--- Strategy: {strategy} ---")
            for i, row in held_out.iterrows():
                image_path = row[IMAGE_PATH_COL]
                true_label = row[TRUE_LABEL_COL]

                start = time.perf_counter()
                result = call_gateway(client, image_path, strategy)
                wall_latency_ms = (time.perf_counter() - start) * 1000

                if "error" in result:
                    raw_rows.append({
                        "strategy": strategy,
                        "image_path": image_path,
                        "true_label": true_label,
                        "predicted_class": None,
                        "routed_tier": None,
                        "correct": None,
                        "gateway_latency_ms": None,
                        "wall_latency_ms": round(wall_latency_ms, 2),
                        "error": result["error"],
                    })
                    if (i + 1) <= 3 or (i + 1) % 10 == 0:
                        print(f"  {i + 1}/{len(held_out)} ERROR: {result['error'][:80]}")
                    continue

                predicted_class = result.get("predicted_class")
                routed_tier = result.get("routed_tier")
                correct = (predicted_class == true_label) if predicted_class is not None else None
                timings = result.get("timings_ms", {}) or {}

                raw_rows.append({
                    "strategy": strategy,
                    "image_path": image_path,
                    "true_label": true_label,
                    "predicted_class": predicted_class,
                    "routed_tier": routed_tier,
                    "correct": correct,
                    "gateway_latency_ms": result.get("gateway_latency_ms"),
                    "wall_latency_ms": round(wall_latency_ms, 2),
                    "feature_extraction_ms": timings.get("feature_extraction_ms"),
                    "fast_call_ms": timings.get("fast_call_ms"),
                    "router_predict_ms": timings.get("router_predict_ms"),
                    "rule_decision_ms": timings.get("rule_decision_ms"),
                    "escalated_call_ms": timings.get("escalated_call_ms"),
                    "error": None,
                })

                if (i + 1) <= 3 or (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{len(held_out)} done (last: {routed_tier}, "
                          f"{result.get('gateway_latency_ms', '?')}ms)")

    raw_df = pd.DataFrame(raw_rows)
    raw_out_path = Path("router/core_experiment_raw_results.csv")
    raw_df.to_csv(raw_out_path, index=False)
    print(f"\nSaved raw per-request results to {raw_out_path}")

    return raw_df


def summarize(raw_df: pd.DataFrame):
    error_counts = raw_df[raw_df["error"].notna()].groupby("strategy").size()
    if len(error_counts) > 0:
        print("\nWARNING: some requests errored out, excluded from accuracy/latency stats:")
        print(error_counts)

    clean = raw_df[raw_df["error"].isna()].copy()

    summary_rows = []
    for strategy in STRATEGIES:
        subset = clean[clean["strategy"] == strategy]
        if len(subset) == 0:
            continue

        accuracy = subset["correct"].mean()
        avg_gateway_latency = subset["gateway_latency_ms"].mean()
        p95_gateway_latency = subset["gateway_latency_ms"].quantile(0.95)
        avg_wall_latency = subset["wall_latency_ms"].mean()

        tier_counts = subset["routed_tier"].value_counts(normalize=True).to_dict()
        tier_dist = ", ".join(f"{k}:{v:.0%}" for k, v in sorted(tier_counts.items()))

        def avg_or_none(col):
            if col not in subset.columns:
                return None
            vals = subset[col].dropna()
            return round(vals.mean(), 3) if len(vals) > 0 else None

        summary_rows.append({
            "strategy": strategy,
            "n_requests": len(subset),
            "accuracy": round(accuracy, 4),
            "avg_gateway_latency_ms": round(avg_gateway_latency, 2),
            "p95_gateway_latency_ms": round(p95_gateway_latency, 2),
            "avg_wall_latency_ms": round(avg_wall_latency, 2),
            "avg_feature_extraction_ms": avg_or_none("feature_extraction_ms"),
            "avg_fast_call_ms": avg_or_none("fast_call_ms"),
            "avg_router_predict_ms": avg_or_none("router_predict_ms"),
            "avg_rule_decision_ms": avg_or_none("rule_decision_ms"),
            "avg_escalated_call_ms": avg_or_none("escalated_call_ms"),
            "tier_distribution": tier_dist,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_out_path = Path("router/core_experiment_summary.csv")
    summary_df.to_csv(summary_out_path, index=False)

    print(f"\nSaved comparison table to {summary_out_path}\n")
    print(summary_df.to_string(index=False))

    return summary_df


if __name__ == "__main__":
    raw_df = run_experiment()
    summarize(raw_df)