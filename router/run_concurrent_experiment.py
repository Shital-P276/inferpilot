"""
Concurrent experiment: combines run_core_experiment.py's ground-truth accuracy
scoring with k6-style concurrent load testing (5/25/50 VUs phases).

Every request in a real concurrent load test is scored against ground truth,
producing both accuracy AND real concurrent latency/throughput from the same
data in one script, one run.

Output: router/concurrent_experiment_raw_results.csv,
        router/concurrent_experiment_summary.csv
"""

import asyncio
import os
import random
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
HELD_OUT_PATH = BASE_DIR / "router" / "held_out_images.csv"
RAW_RESULTS_PATH = Path(os.environ.get(
    "CONCURRENT_EXPERIMENT_RAW_PATH",
    str(BASE_DIR / "router" / "concurrent_experiment_raw_results.csv"),
))
SUMMARY_PATH = Path(os.environ.get(
    "CONCURRENT_EXPERIMENT_SUMMARY_PATH",
    str(BASE_DIR / "router" / "concurrent_experiment_summary.csv"),
))

GATEWAY_URL = "http://localhost:8000/route"
STRATEGIES = [
    "always_fast",
    "always_heavy",
    "round_robin",
    "rule_based",
    "ml_router",
    "single_shot_router",
    "correctness_gate_router",
    "correctness_gate_v3_router",
    "correctness_gate_v3a_router",
    "hybrid_gate",
    "hybrid_dumb",
]

PHASES = [
    {"name": "light", "vus": 5, "duration_s": 30},
    {"name": "heavy", "vus": 25, "duration_s": 30},
    {"name": "burst", "vus": 50, "duration_s": 30},
]

TIMEOUT_S = 15.0


# ---------------------------------------------------------------------------
# SIMULATED_HEAVY_DELAY_MS guard
# ---------------------------------------------------------------------------

def _check_simulated_delay():
    val = int(os.environ.get("SIMULATED_HEAVY_DELAY_MS", "0"))
    if val != 0:
        print(
            f"\n{'!'*70}\n"
            f"WARNING: SIMULATED_HEAVY_DELAY_MS={val} (not 0).\n"
            f"This project's primary baseline uses real Heavy inference, not "
            f"simulated delay.\n"
            f"Set SIMULATED_HEAVY_DELAY_MS=0 before running for real results.\n"
            f"{'!'*70}\n"
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_held_out() -> pd.DataFrame:
    df = pd.read_csv(HELD_OUT_PATH)
    df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)
    print(f"Loaded {len(df)} held-out images")
    return df


def _read_image_bytes(relative_path: str) -> bytes | None:
    full = BASE_DIR / relative_path
    try:
        return full.read_bytes()
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Single async request
# ---------------------------------------------------------------------------

async def _send_one(
    client: httpx.AsyncClient,
    strategy: str,
    image_path: str,
    image_bytes: bytes,
    true_label: str,
    debug: bool = False,
) -> dict[str, Any]:
    try:
        resp = await client.post(
            GATEWAY_URL,
            params={"strategy": strategy},
            files={"file": (Path(image_path).name, image_bytes, "image/jpeg")},
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        result = resp.json()
        predicted_class = result.get("predicted_class")
        correct = int(predicted_class == true_label)
        row = {
            "strategy": strategy,
            "image_path": image_path,
            "true_label": true_label,
            "predicted_class": predicted_class,
            "correct": correct,
            "routed_tier": result.get("routed_tier"),
            "gateway_latency_ms": result.get("gateway_latency_ms"),
        }
        if debug:
            print(f"\n  [DEBUG] Gateway response keys: {sorted(result.keys())}")
            print(f"  [DEBUG] predicted_class = {predicted_class!r}")
            print(f"  [DEBUG] true_label     = {true_label!r}")
            print(f"  [DEBUG] correct        = {correct}")
            print(f"  [DEBUG] routed_tier    = {result.get('routed_tier')!r}")
            print(f"  [DEBUG] gateway_latency= {result.get('gateway_latency_ms')!r}")
            print(f"  [DEBUG] Full response  = {result}")
        return row
    except Exception as exc:
        if debug:
            print(f"\n  [DEBUG] REQUEST FAILED: {type(exc).__name__}: {exc}")
            print(f"  [DEBUG] image_path = {image_path}")
            print(f"  [DEBUG] true_label = {true_label}")
        return {
            "strategy": strategy,
            "image_path": image_path,
            "true_label": true_label,
            "predicted_class": None,
            "correct": 0,
            "routed_tier": None,
            "gateway_latency_ms": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# Single worker coroutine
# ---------------------------------------------------------------------------

async def _worker(
    worker_id: int,
    strategy: str,
    images: list[dict],
    stop_event: asyncio.Event,
    results: list,
    lock: asyncio.Lock,
    debug: bool = False,
):
    async with httpx.AsyncClient() as client:
        while not stop_event.is_set():
            chosen = random.choice(images)
            row = await _send_one(
                client,
                strategy,
                chosen["image_path"],
                chosen["image_bytes"],
                chosen["true_label"],
                debug=debug,
            )
            async with lock:
                results.append(row)
            debug = False


# ---------------------------------------------------------------------------
# Run one phase (constant VUs for duration_s)
# ---------------------------------------------------------------------------

async def _run_phase(
    strategy: str,
    phase: dict,
    images: list[dict],
    phase_label: str,
) -> list[dict]:
    vus = phase["vus"]
    duration_s = phase["duration_s"]
    print(f"  Phase {phase_label}: {vus} VUs for {duration_s}s ...")

    results: list[dict] = []
    lock = asyncio.Lock()
    stop_event = asyncio.Event()

    workers = [
        asyncio.create_task(_worker(
            i, strategy, images, stop_event, results, lock,
            debug=(i == 0),
        ))
        for i in range(vus)
    ]

    await asyncio.sleep(duration_s)
    stop_event.set()

    await asyncio.gather(*workers, return_exceptions=True)

    n_ok = sum(1 for r in results if "error" not in r)
    n_err = sum(1 for r in results if "error" in r)
    print(f"    -> {len(results)} total requests: {n_ok} succeeded, {n_err} failed")
    if n_err > 0:
        first_err = next(r for r in results if "error" in r)
        print(f"    -> First error: {first_err['error']}")
    return results


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def _compute_stats(rows: list[dict], label: str) -> dict:
    df = pd.DataFrame(rows)
    ok_mask = ~df["error"].fillna(False).astype(bool) if "error" in df.columns else pd.Series(True, index=df.index)
    valid = df[ok_mask & df["gateway_latency_ms"].notna()]
    tier_counts = df.loc[ok_mask, "routed_tier"].value_counts(normalize=True) * 100

    return {
        "strategy": rows[0]["strategy"] if rows else "",
        "concurrency_phase": label,
        "n_requests": int(ok_mask.sum()),
        "n_errors": int((~ok_mask).sum()),
        "accuracy": round(df.loc[ok_mask, "correct"].mean(), 4) if ok_mask.any() else 0,
        "avg_latency_ms": round(valid["gateway_latency_ms"].mean(), 2) if len(valid) else None,
        "p50_latency_ms": round(valid["gateway_latency_ms"].quantile(0.50), 2) if len(valid) else None,
        "p95_latency_ms": round(valid["gateway_latency_ms"].quantile(0.95), 2) if len(valid) else None,
        "p99_latency_ms": round(valid["gateway_latency_ms"].quantile(0.99), 2) if len(valid) else None,
        "throughput_rps": round(len(valid) / 30.0, 2) if len(valid) else 0,
        "pct_fast": round(tier_counts.get("fast", 0), 1),
        "pct_balanced": round(tier_counts.get("balanced", 0), 1),
        "pct_heavy": round(tier_counts.get("heavy", 0), 1),
    }


# ---------------------------------------------------------------------------
# Run all phases for one strategy
# ---------------------------------------------------------------------------

async def _run_strategy(
    strategy: str,
    held_out_df: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    print(f"\n{'='*60}\nStrategy: {strategy}\n{'='*60}")

    # Pre-load image bytes once
    images = []
    for _, row in held_out_df.iterrows():
        img_bytes = _read_image_bytes(row["image_path"])
        if img_bytes is not None:
            images.append({
                "image_path": row["image_path"],
                "true_label": row["true_label"],
                "image_bytes": img_bytes,
            })
    print(f"  Loaded {len(images)} images into memory")

    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    strategy_start = time.time()

    for phase in PHASES:
        phase_start = time.time()
        phase_rows = await _run_phase(strategy, phase, images, phase["name"])
        phase_elapsed = time.time() - phase_start
        all_rows.extend(phase_rows)

        stats = _compute_stats(phase_rows, phase["name"])
        stats["phase_elapsed_s"] = round(phase_elapsed, 2)
        summary_rows.append(stats)

        print(
            f"    accuracy={stats['accuracy']:.4f}  "
            f"avg_lat={stats['avg_latency_ms']}ms  "
            f"p95={stats['p95_latency_ms']}ms  "
            f"throughput={stats['throughput_rps']} req/s"
        )

    # Overall
    overall_stats = _compute_stats(all_rows, "overall")
    overall_stats["phase_elapsed_s"] = round(time.time() - strategy_start, 2)
    summary_rows.append(overall_stats)

    elapsed = time.time() - strategy_start
    print(f"  Strategy {strategy} done: {len(all_rows)} requests in {elapsed:.1f}s")

    return all_rows, summary_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main():
    _check_simulated_delay()

    held_out_df = load_held_out()
    all_raw: list[dict] = []
    all_summary: list[dict] = []

    for strategy in STRATEGIES:
        raw, summary = await _run_strategy(strategy, held_out_df)
        all_raw.extend(raw)
        all_summary.extend(summary)

    # Write raw per-request CSV (keep error column for diagnostics)
    raw_df = pd.DataFrame(all_raw)
    raw_df.to_csv(RAW_RESULTS_PATH, index=False)
    print(f"\nRaw results written to {RAW_RESULTS_PATH}")

    # Write summary CSV
    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"Summary written to {SUMMARY_PATH}")

    # Print final summary table
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    display_cols = [
        "strategy", "concurrency_phase", "n_requests", "n_errors", "accuracy",
        "avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
        "throughput_rps", "pct_fast", "pct_balanced", "pct_heavy",
    ]
    print(summary_df[display_cols].to_string(index=False))


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
