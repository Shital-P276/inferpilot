"""
profile_router_predict.py

Standalone micro-benchmark for the router's predict() call, isolated from
all network/gateway overhead. Pinpoints exactly which sub-step inside
router_predict_ms (observed ~56ms in the live gateway) is actually slow:
DataFrame construction, the Random Forest's predict() itself, or the
label-decode step.

Run this LOCALLY (venv, not inside a container) -- it loads the pickle
directly and calls it in a tight loop with a fixed dummy row, so there's
no HTTP/Docker/GPU contention involved at all. This isolates the router's
own CPU-bound cost.

USAGE:
    python router/profile_router_predict.py
"""

import time
import pickle
import statistics

import pandas as pd

ROUTER_MODEL_PATH = "router/router_best_model.pkl"
N_ITERATIONS = 200
N_WARMUP = 20


def load_router():
    with open(ROUTER_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["label_encoder"], bundle["feature_cols"]


def build_dummy_row(feature_cols):
    """A representative single request's worth of features -- values don't
    matter for timing purposes, just need to hit every column the real
    pipeline expects."""
    dummy_values = {
        "width": 224,
        "height": 224,
        "file_size_kb": 85.3,
        "blur_score": 120.5,
        "brightness": 130.2,
        "edge_density": 0.045,
        "fast_confidence": 0.62,
        "load_stage": "light",
    }
    row = {k: dummy_values[k] for k in feature_cols if k in dummy_values}
    return row, feature_cols


def time_block(fn, n_iterations, n_warmup):
    for _ in range(n_warmup):
        fn()
    timings = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - t0) * 1000)
    return timings


def summarize(name, timings):
    print(f"\n{name}")
    print(f"  mean:   {statistics.mean(timings):.4f} ms")
    print(f"  median: {statistics.median(timings):.4f} ms")
    print(f"  p95:    {sorted(timings)[int(len(timings) * 0.95)]:.4f} ms")
    print(f"  min:    {min(timings):.4f} ms")
    print(f"  max:    {max(timings):.4f} ms")


def main():
    print(f"Loading router from {ROUTER_MODEL_PATH} ...")
    model, label_encoder, feature_cols = load_router()
    row_dict, cols = build_dummy_row(feature_cols)
    print(f"feature_cols: {feature_cols}")

    # ---- Step 1: DataFrame construction alone ----
    def build_df():
        return pd.DataFrame([row_dict], columns=cols)

    df_timings = time_block(build_df, N_ITERATIONS, N_WARMUP)
    summarize("1) DataFrame construction only", df_timings)

    # Pre-build one DataFrame to reuse for the next two isolated steps,
    # so their timings don't include DataFrame construction cost.
    fixed_df = pd.DataFrame([row_dict], columns=cols)

    # ---- Step 2: model.predict() alone (DataFrame already built) ----
    def predict_only():
        return model.predict(fixed_df)

    predict_timings = time_block(predict_only, N_ITERATIONS, N_WARMUP)
    summarize("2) model.predict() only (Pipeline: ColumnTransformer/OneHotEncoder + RandomForest)", predict_timings)

    fixed_label = model.predict(fixed_df)

    # ---- Step 3: label_encoder.inverse_transform() alone ----
    def decode_only():
        return label_encoder.inverse_transform(fixed_label)

    decode_timings = time_block(decode_only, N_ITERATIONS, N_WARMUP)
    summarize("3) label_encoder.inverse_transform() only", decode_timings)

    # ---- Step 4: the full sequence together, for a sanity-check total ----
    def full_sequence():
        df = pd.DataFrame([row_dict], columns=cols)
        label = model.predict(df)
        return label_encoder.inverse_transform(label)

    full_timings = time_block(full_sequence, N_ITERATIONS, N_WARMUP)
    summarize("4) Full sequence (DataFrame + predict + decode) -- should roughly equal 1+2+3", full_timings)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"DataFrame construction:  {statistics.mean(df_timings):8.4f} ms  ({statistics.mean(df_timings) / statistics.mean(full_timings) * 100:5.1f}% of total)")
    print(f"model.predict():         {statistics.mean(predict_timings):8.4f} ms  ({statistics.mean(predict_timings) / statistics.mean(full_timings) * 100:5.1f}% of total)")
    print(f"inverse_transform():     {statistics.mean(decode_timings):8.4f} ms  ({statistics.mean(decode_timings) / statistics.mean(full_timings) * 100:5.1f}% of total)")
    print(f"Full sequence:           {statistics.mean(full_timings):8.4f} ms")
    print(f"\n(Compare this 'Full sequence' number to the ~55.9ms 'avg_router_predict_ms' "
          f"seen in the live gateway experiment -- if this local number is much lower, "
          f"the extra cost in the gateway is coming from somewhere else, e.g. first-call "
          f"cold-start effects inside the container, or something in build_feature_row's "
          f"real feature values vs this dummy row.)")


if __name__ == "__main__":
    main()