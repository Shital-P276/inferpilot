"""
pull_prometheus_data.py

Pulls raw, precise data directly from Prometheus's HTTP API for research
comparison between routing strategies -- no eyeballing chart curves.

Since gateway_latency_ms and routing_decisions_total are already labeled
by "strategy" in gateway_service.py, this queries by that label directly
over a WIDE time window (default: last 2 hours), so it doesn't matter if
you don't know the exact start/end timestamps of each k6 run -- Prometheus
retains the history, and the label filter does the separating for you.

WHAT THIS PULLS (all metrics that exist in your stack, per gateway_service.py
and fast/balanced/heavy_service.py):
  1. gateway_latency_ms -- p50/p95/p99, per strategy, over time (the metric
     that actually matters for the cascade-vs-single-shot comparison)
  2. routing_decisions_total -- final tier distribution, per strategy
  3. inference_latency_ms -- per-tier model inference time (NOT strategy-
     labeled -- this is per SERVICE, i.e. fast/balanced/heavy, shared
     across whichever strategy called them)
  4. prediction_confidence -- per-tier confidence distribution
  5. predictions_total -- per-tier, per-class prediction counts
  6. http_request_duration_seconds -- the auto-instrumented endpoint
     latency (NOTE: this one is NOT strategy-labeled, since it comes from
     prometheus-fastapi-instrumentator's generic per-path/method
     instrumentation, not your custom metrics -- pulled as an overall
     reference series, can't be split by strategy)

USAGE:
    python monitoring/pull_prometheus_data.py
    python monitoring/pull_prometheus_data.py --hours 4
    python monitoring/pull_prometheus_data.py --strategies ml_router single_shot_router

OUTPUT:
    monitoring/prometheus_pulls/*.csv  -- one file per query
    monitoring/prometheus_pulls/summary_comparison.csv -- the headline table
"""

import argparse
import csv
import time
from pathlib import Path

import requests

PROMETHEUS_URL = "http://localhost:9090"
OUTPUT_DIR = Path("monitoring/prometheus_pulls")


def query_range(query: str, start: float, end: float, step: str = "5s"):
    """Runs a PromQL range query against the local Prometheus HTTP API."""
    resp = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query_range",
        params={"query": query, "start": start, "end": end, "step": step},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "success":
        raise RuntimeError(f"Prometheus query failed: {data}")
    return data["data"]["result"]


def query_instant(query: str):
    """Runs a PromQL instant query (current/latest value) against Prometheus."""
    resp = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": query},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "success":
        raise RuntimeError(f"Prometheus query failed: {data}")
    return data["data"]["result"]


def save_range_result_to_csv(result, out_path: Path, value_label: str):
    """Flattens a PromQL range_query result (one or more time series, each
    with its own label set) into a single CSV: timestamp, value, and all
    label key/value pairs as extra columns."""
    rows = []
    all_label_keys = set()
    for series in result:
        for label_key in series["metric"]:
            all_label_keys.add(label_key)

    for series in result:
        labels = series["metric"]
        for ts, val in series["values"]:
            row = {"timestamp": ts, value_label: val}
            for k in all_label_keys:
                row[k] = labels.get(k, "")
            rows.append(row)

    if not rows:
        print(f"  WARNING: no data returned for {out_path.name} -- skipping file.")
        return

    fieldnames = ["timestamp", value_label] + sorted(all_label_keys)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} rows -> {out_path}")


def pull_gateway_latency_percentiles(strategies, start, end):
    """p50/p95/p99 gateway_latency_ms over time, per strategy."""
    print("\n=== gateway_latency_ms percentiles (per strategy) ===")
    for strategy in strategies:
        for pct, pct_name in [(0.50, "p50"), (0.95, "p95"), (0.99, "p99")]:
            query = (
                f'histogram_quantile({pct}, '
                f'sum(rate(gateway_latency_ms_bucket{{strategy="{strategy}"}}[1m])) by (le))'
            )
            result = query_range(query, start, end)
            out_path = OUTPUT_DIR / f"gateway_latency_{strategy}_{pct_name}.csv"
            save_range_result_to_csv(result, out_path, f"latency_ms_{pct_name}")


def pull_routing_decisions(strategies, start, end):
    """Final tier distribution over time, per strategy."""
    print("\n=== routing_decisions_total (tier distribution per strategy) ===")
    for strategy in strategies:
        query = f'routing_decisions_total{{strategy="{strategy}"}}'
        result = query_range(query, start, end)
        out_path = OUTPUT_DIR / f"routing_decisions_{strategy}.csv"
        save_range_result_to_csv(result, out_path, "count")


def pull_inference_latency(start, end):
    """Per-tier model inference latency -- NOT strategy-labeled, shared
    across whichever strategy called that tier's service."""
    print("\n=== inference_latency_ms (per tier, all strategies combined) ===")
    for pct, pct_name in [(0.50, "p50"), (0.95, "p95")]:
        query = f'histogram_quantile({pct}, sum(rate(inference_latency_ms_bucket[1m])) by (le, tier))'
        result = query_range(query, start, end)
        out_path = OUTPUT_DIR / f"inference_latency_{pct_name}.csv"
        save_range_result_to_csv(result, out_path, f"latency_ms_{pct_name}")


def pull_prediction_confidence(start, end):
    print("\n=== prediction_confidence (per tier) ===")
    query = 'histogram_quantile(0.50, sum(rate(prediction_confidence_bucket[1m])) by (le, tier))'
    result = query_range(query, start, end)
    out_path = OUTPUT_DIR / "prediction_confidence_median.csv"
    save_range_result_to_csv(result, out_path, "confidence_median")


def pull_predictions_by_class(start, end):
    print("\n=== predictions_total (per tier, per class) ===")
    query = 'predictions_total'
    result = query_range(query, start, end)
    out_path = OUTPUT_DIR / "predictions_by_class.csv"
    save_range_result_to_csv(result, out_path, "count")


def pull_http_request_duration(start, end):
    """Auto-instrumented, NOT strategy-labeled -- overall /predict endpoint
    latency across whatever traffic hit it during the window, regardless
    of which strategy the gateway used internally."""
    print("\n=== http_request_duration_seconds (overall, NOT strategy-split) ===")
    query = 'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{handler="/predict"}[1m])) by (le))'
    try:
        result = query_range(query, start, end)
        out_path = OUTPUT_DIR / "http_request_duration_p95.csv"
        save_range_result_to_csv(result, out_path, "duration_seconds_p95")
    except Exception as e:
        print(f"  WARNING: query failed ({e}) -- the 'handler' label name may differ "
              f"in your actual instrumentation. Check available labels via Prometheus's "
              f"own UI (http://localhost:9090/graph) if this matters to you.")


def build_summary_comparison(strategies):
    """Pulls CURRENT (instant) summary stats per strategy -- peak/overall
    p95 latency and tier distribution -- into one small comparison table,
    the headline numbers for the report."""
    print("\n=== Building summary comparison table ===")
    rows = []
    for strategy in strategies:
        row = {"strategy": strategy}

        for pct, pct_name in [(0.50, "p50"), (0.95, "p95"), (0.99, "p99")]:
            query = (
                f'histogram_quantile({pct}, '
                f'sum(rate(gateway_latency_ms_bucket{{strategy="{strategy}"}}[5m])) by (le))'
            )
            result = query_instant(query)
            row[f"gateway_latency_{pct_name}_ms"] = round(float(result[0]["value"][1]), 2) if result else None

        tier_query = f'sum(routing_decisions_total{{strategy="{strategy}"}}) by (tier)'
        tier_result = query_instant(tier_query)
        total = sum(float(r["value"][1]) for r in tier_result)
        for r in tier_result:
            tier = r["metric"]["tier"]
            count = float(r["value"][1])
            row[f"pct_{tier}"] = round(count / total * 100, 1) if total > 0 else 0

        rows.append(row)

    if not rows:
        print("  No data to summarize.")
        return

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = ["strategy"] + sorted(all_keys - {"strategy"})

    out_path = OUTPUT_DIR / "summary_comparison.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved summary comparison to {out_path}")
    print("\n" + "=" * 70)
    for row in rows:
        print(f"\n{row['strategy']}:")
        for k, v in row.items():
            if k != "strategy":
                print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=2.0,
                         help="How many hours back to pull data from (default: 2)")
    parser.add_argument("--strategies", nargs="+",
                         default=["ml_router", "always_fast", "always_heavy",
                                  "round_robin", "rule_based", "single_shot_router"],
                         help="Which strategies to pull gateway_latency_ms / "
                              "routing_decisions_total for")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Confirm Prometheus is actually reachable before doing anything else
    try:
        requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=5)
    except Exception as e:
        raise RuntimeError(
            f"Could not reach Prometheus at {PROMETHEUS_URL} ({e}). "
            f"Confirm it's running: curl.exe http://localhost:9090/-/healthy"
        )

    end = time.time()
    start = end - (args.hours * 3600)
    print(f"Pulling data from the last {args.hours} hour(s) "
          f"({time.strftime('%H:%M:%S', time.localtime(start))} to "
          f"{time.strftime('%H:%M:%S', time.localtime(end))})")

    pull_gateway_latency_percentiles(args.strategies, start, end)
    pull_routing_decisions(args.strategies, start, end)
    pull_inference_latency(start, end)
    pull_prediction_confidence(start, end)
    pull_predictions_by_class(start, end)
    pull_http_request_duration(start, end)
    build_summary_comparison(args.strategies)

    print(f"\nAll data saved to {OUTPUT_DIR}/")
    print("Note: strategies/tiers with no traffic in the pulled window will show "
          "'no data' warnings above and won't have a CSV -- that's expected for "
          "any strategy you haven't actually load-tested yet.")


if __name__ == "__main__":
    main()