"""
router/run_chaos_experiment.py

Single-strategy chaos probe (copied pattern from run_concurrent_experiment.py,
standalone file so the main script is never edited for chaos runs).

Purpose: generate failure/fallback traces against a faulted gateway --
typically FAULT_INJECTION=heavy:1.0 (or a `docker stop` on a tier) -- and
record breaker-state snapshots over time for the resilience report.

Usage (PowerShell):
  $env:FAULT_INJECTION="heavy:1.0"; docker compose up --build gateway
  python router/run_chaos_experiment.py --strategy correctness_gate_v3_router
  Remove-Item Env:FAULT_INJECTION   # then rebuild to return to stock

Outputs (fresh names by default; never touches the baseline CSVs):
  router/chaos_experiment_raw.csv / .json override via env
  router/chaos_experiment_summary.csv
  router/chaos_health_trace.jsonl   (periodic /health snapshots)
"""

import argparse
import asyncio
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
HELD_OUT_PATH = BASE_DIR / "router" / "held_out_images.csv"
RAW_RESULTS_PATH = Path(os.environ.get(
    "CHAOS_EXPERIMENT_RAW_PATH",
    str(BASE_DIR / "router" / "chaos_experiment_raw.csv"),
))
SUMMARY_PATH = Path(os.environ.get(
    "CHAOS_EXPERIMENT_SUMMARY_PATH",
    str(BASE_DIR / "router" / "chaos_experiment_summary.csv"),
))
HEALTH_TRACE_PATH = Path(os.environ.get(
    "CHAOS_HEALTH_TRACE_PATH",
    str(BASE_DIR / "router" / "chaos_health_trace.jsonl"),
))

GATEWAY_URL = "http://localhost:8000/route"
HEALTH_URL = "http://localhost:8000/health"
TIMEOUT_S = 15.0


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


async def _send_one(client: httpx.AsyncClient, strategy: str, image_path: str,
                    image_bytes: bytes, true_label: str) -> dict[str, Any]:
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
        return {
            "strategy": strategy,
            "image_path": image_path,
            "true_label": true_label,
            "predicted_class": predicted_class,
            "correct": int(predicted_class == true_label),
            "routed_tier": result.get("routed_tier"),
            "fallback_from": result.get("fallback_from"),
            "degraded_routing": result.get("degraded_routing"),
            "gateway_latency_ms": result.get("gateway_latency_ms"),
        }
    except Exception as exc:
        return {
            "strategy": strategy,
            "image_path": image_path,
            "true_label": true_label,
            "predicted_class": None,
            "correct": 0,
            "routed_tier": None,
            "fallback_from": None,
            "degraded_routing": None,
            "gateway_latency_ms": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def _worker(worker_id: int, strategy: str, images: list[dict],
                  stop_event: asyncio.Event, results: list,
                  lock: asyncio.Lock):
    async with httpx.AsyncClient() as client:
        while not stop_event.is_set():
            chosen = random.choice(images)
            row = await _send_one(client, strategy, chosen["image_path"],
                                  chosen["image_bytes"], chosen["true_label"])
            async with lock:
                results.append(row)


async def _health_poller(stop_event: asyncio.Event, interval_s: float):
    """Append /health snapshots (breaker states) until stopped."""
    with open(HEALTH_TRACE_PATH, "w") as f:
        async with httpx.AsyncClient() as client:
            while not stop_event.is_set():
                try:
                    resp = await client.get(HEALTH_URL, timeout=5.0)
                    snap = resp.json() if resp.status_code == 200 else \
                        {"health_error": resp.status_code}
                except Exception as exc:
                    snap = {"health_error": f"{type(exc).__name__}: {exc}"}
                snap["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                f.write(json.dumps(snap) + "\n")
                f.flush()
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
                except asyncio.TimeoutError:
                    continue


async def _run_phase(strategy: str, phase: dict, images: list[dict]) -> list[dict]:
    vus, duration_s = phase["vus"], phase["duration_s"]
    print(f"  Phase {phase['name']}: {vus} VUs for {duration_s}s ...")
    results: list[dict] = []
    lock = asyncio.Lock()
    stop_event = asyncio.Event()
    workers = [asyncio.create_task(_worker(i, strategy, images, stop_event,
                                           results, lock))
               for i in range(vus)]
    await asyncio.sleep(duration_s)
    stop_event.set()
    await asyncio.gather(*workers, return_exceptions=True)
    n_ok = sum(1 for r in results if "error" not in r)
    n_err = sum(1 for r in results if "error" in r)
    n_fb = sum(1 for r in results
               if r.get("fallback_from") not in (None, ""))
    print(f"    -> {len(results)} requests: {n_ok} ok, {n_err} failed, "
          f"{n_fb} fallbacks")
    return results


def _compute_stats(rows: list[dict], label: str) -> dict:
    df = pd.DataFrame(rows)
    ok_mask = ~df["error"].fillna(False).astype(bool) \
        if "error" in df.columns else pd.Series(True, index=df.index)
    valid = df[ok_mask & df["gateway_latency_ms"].notna()]
    fb = df.loc[ok_mask, "fallback_from"]
    fb_count = int(fb.notna().sum()) if "fallback_from" in df.columns else 0
    tier_counts = df.loc[ok_mask, "routed_tier"].value_counts(normalize=True) * 100
    return {
        "strategy": rows[0]["strategy"] if rows else "",
        "concurrency_phase": label,
        "n_requests": int(ok_mask.sum()),
        "n_errors": int((~ok_mask).sum()),
        "n_fallbacks": fb_count,
        "accuracy": round(df.loc[ok_mask, "correct"].mean(), 4) if ok_mask.any() else 0,
        "avg_latency_ms": round(valid["gateway_latency_ms"].mean(), 2) if len(valid) else None,
        "p95_latency_ms": round(valid["gateway_latency_ms"].quantile(0.95), 2) if len(valid) else None,
        "throughput_rps": round(len(valid) / 30.0, 2) if len(valid) else 0,
        "pct_fast": round(tier_counts.get("fast", 0), 1),
        "pct_balanced": round(tier_counts.get("balanced", 0), 1),
        "pct_heavy": round(tier_counts.get("heavy", 0), 1),
    }


async def async_main(strategy: str, phases: list[dict], health_poll_s: float):
    held_out_df = load_held_out()
    images = []
    for _, row in held_out_df.iterrows():
        img_bytes = _read_image_bytes(row["image_path"])
        if img_bytes is not None:
            images.append({"image_path": row["image_path"],
                           "true_label": row["true_label"],
                           "image_bytes": img_bytes})
    print(f"Loaded {len(images)} images into memory")
    print(f"Strategy: {strategy} | health poll every {health_poll_s}s -> "
          f"{HEALTH_TRACE_PATH.name}")

    stop_health = asyncio.Event()
    poller = asyncio.create_task(_health_poller(stop_health, health_poll_s))

    all_raw, all_summary = [], []
    for phase in phases:
        phase_rows = await _run_phase(strategy, phase, images)
        all_raw.extend(phase_rows)
        stats = _compute_stats(phase_rows, phase["name"])
        all_summary.append(stats)
        print(f"    accuracy={stats['accuracy']:.4f} "
              f"fallbacks={stats['n_fallbacks']} "
              f"p95={stats['p95_latency_ms']}ms")

    stop_health.set()
    await asyncio.gather(poller, return_exceptions=True)

    overall = _compute_stats(all_raw, "overall")
    all_summary.append(overall)
    pd.DataFrame(all_raw).to_csv(RAW_RESULTS_PATH, index=False)
    print(f"\nRaw results written to {RAW_RESULTS_PATH}")
    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"Summary written to {SUMMARY_PATH}")
    print(f"Health trace written to {HEALTH_TRACE_PATH}")
    print(f"\n{summary_df.to_string(index=False)}")


def main():
    ap = argparse.ArgumentParser(description="Single-strategy chaos probe")
    ap.add_argument("--strategy", default="correctness_gate_v3_router")
    ap.add_argument("--vus", default="5,25,50",
                    help="Comma-separated VUs per light,heavy,burst phase")
    ap.add_argument("--duration-s", type=float, default=30.0)
    ap.add_argument("--health-poll-s", type=float, default=5.0)
    args = ap.parse_args()
    vus = [int(v) for v in args.vus.split(",")]
    names = ["light", "heavy", "burst"]
    phases = [{"name": n, "vus": v, "duration_s": args.duration_s}
              for n, v in zip(names, vus)]
    asyncio.run(async_main(args.strategy, phases, args.health_poll_s))


if __name__ == "__main__":
    main()
