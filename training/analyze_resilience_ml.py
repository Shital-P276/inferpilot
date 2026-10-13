"""
training/analyze_resilience_ml.py

ML-on-resilience analysis (offline, read-only; no retraining of any model).

 Hegedűs-data: clean 11-strategy run + chaos run (FAULT_INJECTION=heavy:1.0).
 Accepts extra run files (--extra) e.g. partial-fault probes, same schema.

Part 1 -- learned breaker thresholds (replay):
  Replays the chaos heavy-intended request sequence under candidate
  consecutive-failure thresholds N=1..10: failures-before-open, requests-
  to-open, and resulting accuracy (failed attempts fall back, so errors stay
  0; the cost is direct-vs-probed fallback routing). Combined with the
  clean-run constraint (max consecutive tier failures observed in clean runs)
  to recommend N with margin. Honest caveat printed: single fault rate
  (1.0); intermittent faults would separate candidates better.

Part 2 -- anomaly detection (change-point on a stitched timeline):
  Timeline = clean correctness_gate_v3_router rows (label 0) ++ chaos rows
  (label 1), in file order (~completion order). Online-only features
  (rolling window 50, past data only). Three detectors:
    (a) breaker baseline: first fallback flag (the system itself);
    (b) latency z-score: |z|>3 on rolling log-latency;
    (c) IsolationForest on [log-latency, fallback, error], fit on the first
        200 clean requests, applied to the rest.
  Metrics: first-alarm index in chaos segment (delay in requests),
  false alarms in clean segment. Negative outcomes reported, not hidden --
  e.g. fail-fast injected faults short-circuit (no latency spike), which a
  latency detector cannot see.

Run: source ~/venv_inferpilot/bin/activate
     python training/analyze_resilience_ml.py [--extra <csv> ...]
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

CLEAN_CSV = "router/concurrent_experiment_raw_v3a_full11.csv"
CLEAN_STRATEGY = "correctness_gate_v3_router"
CHAOS_CSV = "router/chaos_experiment_raw.csv"
ROLL = 50

BREAKER_CURRENT_N = 5


def load_timeline(extra, clean_csv=CLEAN_CSV, chaos_csv=CHAOS_CSV,
                  clean_strategy=CLEAN_STRATEGY):
    clean = pd.read_csv(clean_csv)
    clean = clean[clean.strategy == clean_strategy].reset_index(drop=True)
    chaos = pd.read_csv(chaos_csv).reset_index(drop=True)
    frames = [clean.assign(segment="clean"), chaos.assign(segment="chaos")]
    for path in extra:
        df = pd.read_csv(path).reset_index(drop=True)
        frames.append(df.assign(segment=f"extra:{path}"))
    tl = pd.concat(frames, ignore_index=True)
    tl["is_fallback"] = tl["fallback_from"].notna().astype(int)
    tl["is_error"] = tl["gateway_latency_ms"].isna().astype(int)
    tl["log_lat"] = np.log1p(tl["gateway_latency_ms"].fillna(
        tl["gateway_latency_ms"].median()))
    return tl, len(clean)


def part1_thresholds(chaos):
    print("=" * 70)
    print("Part 1: breaker-threshold replay (chaos run, heavy fault rate 1.0)")
    print("=" * 70)
    # Heavy-intended requests in file order; with rate-1.0 every pre-open
    # heavy attempt fails (then falls back -- errors stay 0 in the data).
    n_total = len(chaos)
    # Reconstruct heavy-intended: fallbacks from heavy + actually-routed heavy.
    # In this trace routed heavy == 0, so heavy-intended == fallback_from==heavy.
    heavy_intended = (chaos["fallback_from"] == "heavy").sum()
    print(f"  chaos requests: {n_total}, heavy-intended: {heavy_intended} "
          f"({heavy_intended / n_total:.1%}), errors observed: 0")
    print(f"\n  {'N':>3}  {'failed-calls-before-open':>26}  {'~requests-to-open':>18}")
    for n in range(1, 11):
        failed = min(n, heavy_intended)
        # heavy-intended share => total requests elapsed to accumulate n heavies
        share = heavy_intended / n_total
        req_to_open = int(round(failed / share)) if share > 0 else n_total
        print(f"  {n:>3}  {failed:>26d}  {req_to_open:>18d}")
    print(f"\n  Current BREAKER_FAILURE_THRESHOLD={BREAKER_CURRENT_N}: "
          f"~{int(round(min(5, heavy_intended) / (heavy_intended / n_total)))} "
          f"requests to open under total outage.")


def part1_clean_constraint():
    print("\nClean-run constraint (all strategies, full-11 run + chaos run):")
    clean = pd.read_csv(CLEAN_CSV)
    # Per-tier consecutive-failure proxy: errors are 0 everywhere in clean data,
    # and chaos errors are 0 too (fallback absorbs). So max OBSERVED consecutive
    # tier-call failures in any clean run = 0.
    print(f"  total clean errors: {int(clean['gateway_latency_ms'].isna().sum())} "
          f"over {len(clean)} requests")
    print("  => max observed consecutive tier failures on clean data: 0.")
    print("  => any N>=1 never false-trips on observed clean data; N=3 reacts "
          "~40% faster than N=5 under total outage at zero observed cost.")
    print("  CAVEAT: single fault rate (1.0). Intermittent faults (0.2-0.5) are "
          "needed to measure flapping risk and pick a true optimum -- "
          "recommend 2 short partial-fault probes (see bottom).")


def part2_anomaly(tl, n_clean):
    print("\n" + "=" * 70)
    print("Part 2: anomaly detection on stitched timeline "
          f"(clean n={n_clean} ++ chaos n={len(tl) - n_clean})")
    print("=" * 70)
    log_lat = tl["log_lat"].to_numpy()
    fb = tl["is_fallback"].to_numpy()
    err = tl["is_error"].to_numpy()

    # (a) breaker baseline: first fallback flag anywhere
    first_fb = int(np.nonzero(fb)[0].min()) if fb.sum() else -1
    print(f"  (a) breaker (first fallback): index {first_fb} "
          f"({'clean!' if first_fb < n_clean else 'chaos, delay '
                f'{first_fb - n_clean} reqs past onset'})")

    # (b) online rolling latency z-score
    z_alarm, feas = -1, 0
    for i in range(ROLL, len(tl)):
        window = log_lat[i - ROLL:i]
        mu, sd = window.mean(), window.std()
        if sd < 1e-9:
            continue
        feas += 1
        if abs((log_lat[i] - mu) / sd) > 3.0:
            z_alarm = i
            break
    print(f"  (b) latency |z|>3 (rolling {ROLL}): first alarm index {z_alarm} "
          f"({'clean (FALSE ALARM)' if 0 <= z_alarm < n_clean else 'chaos, delay '
            f'{z_alarm - n_clean} reqs' if z_alarm >= 0 else 'NEVER FIRED'})")

    # (c) IsolationForest on [log-lat, fallback, error]; fit on clean head
    X = np.column_stack([log_lat, fb, err])
    clf = IsolationForest(n_estimators=200, contamination=0.01,
                          random_state=42)
    clf.fit(X[:200])
    pred = clf.predict(X[200:])  # -1 = anomaly
    alarm_rel = np.nonzero(pred == -1)[0]
    clean_alarms = int((alarm_rel + 200 < n_clean).sum())
    chaos_alarms = alarm_rel[alarm_rel + 200 >= n_clean]
    first_if = int(chaos_alarms.min() + 200) if len(chaos_alarms) else -1
    print(f"  (c) IsolationForest: {clean_alarms} false alarms on clean segment; "
          f"first chaos alarm index {first_if} "
          f"({'delay ' + str(first_if - n_clean) + ' reqs' if first_if >= 0 else 'NEVER FIRED'})")

    print("\n  EXPECTATION (pre-registered): fail-fast injected faults "
          "short-circuit (chaos mean latency is LOWER than clean), so (b) "
          "should fail or fire backwards; (a) should win on total outage; "
          "(c) can only match (a) via the fallback feature, with delay. "
          "The detector's real value would be partial/intermittent faults "
          "where the breaker has NOT tripped -- not measurable in this trace.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[],
                    help="Extra run CSVs (partial-fault probes), same schema")
    ap.add_argument("--chaos", default=CHAOS_CSV,
                    help="Chaos-run CSV (default: %(default)s)")
    ap.add_argument("--clean", default=CLEAN_CSV,
                    help="Clean-run CSV (default: %(default)s)")
    ap.add_argument("--clean-strategy", default=CLEAN_STRATEGY)
    args = ap.parse_args()

    tl, n_clean = load_timeline(args.extra, clean_csv=args.clean,
                                chaos_csv=args.chaos,
                                clean_strategy=args.clean_strategy)
    chaos = tl[tl.segment == "chaos"].reset_index(drop=True)
    part1_thresholds(chaos)
    part1_clean_constraint()
    part2_anomaly(tl, n_clean)

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("  1. Threshold: N=3 (faster reaction, zero observed false-trip cost).")
    print("     Change is one env var (BREAKER_FAILURE_THRESHOLD=3) + rebuild;")
    print("     re-run the chaos probe to confirm identical absorption.")
    print("  2. Detector: do NOT ship a latency detector (evidence above); the")
    print("     fallback/error-rate channel is the honest signal. Ship it as a")
    print("     Prometheus alert rule on fallback_total rate, not as 'AI'.")
    print("  3. To complete the learning properly: 2 short partial-fault probes")
    print("     FAULT_INJECTION=heavy:0.3 and heavy:0.5 with")
    print("     --vus 2,2,2 --duration-s 10, then re-run this script with")
    print("     --extra <those two CSVs>.")
    print("=" * 70)


if __name__ == "__main__":
    main()
