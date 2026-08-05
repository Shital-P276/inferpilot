"""
benchmark_gateway_cpu_cost.py

Isolates and measures the two CPU-bound steps inside the gateway's ml_router
path, separately, using REAL images from your held-out set:

  1. extract_request_features() -- OpenCV Laplacian (blur) + Canny (edges)
  2. router_model.predict()     -- sklearn Pipeline (preprocess + RandomForest)

This settles which one (or both) is the actual source of ml_router's ~55-85ms
per-request overhead beyond a plain network call, without needing Docker or
the gateway running at all -- pure local timing, same pattern as
profile_router_predict.py and test_n_jobs_1.py.

USAGE:
    python router/benchmark_gateway_cpu_cost.py
"""

import io
import time
import pickle
import statistics
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROUTER_MODEL_PATH = "router/router_best_model.pkl"
HELD_OUT_CSV = "router/held_out_images.csv"
N_SAMPLE_IMAGES = 20   # real images to test across, for a representative average
N_REPEATS_PER_IMAGE = 10
N_WARMUP = 3


def extract_request_features(image_bytes: bytes, file_size_kb: float, max_dimension: int = 1024):
    """Verbatim copy of the gateway's feature extraction, so this benchmark
    measures the exact same code path -- keep in sync with
    serving/gateway_service.py if that function ever changes.

    CAP-BASED RESIZE: only downscales images LARGER than max_dimension on
    their long edge. This is deliberately conservative -- the router was
    trained on features computed from full-resolution images, so downscaling
    every image risks train/serve skew (different blur_score/edge_density
    values than the model learned from). Capping only the oversized tail
    fixes the worst p95/max latency outliers while leaving typical-sized
    images (below the cap) producing IDENTICAL feature values to training.
    """
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = pil_img.size  # report TRUE original dimensions as features, unchanged

    # Only resize the working copy used for blur/edge computation -- not the
    # reported width/height features themselves.
    long_edge = max(width, height)
    if long_edge > max_dimension:
        scale = max_dimension / long_edge
        new_size = (int(width * scale), int(height * scale))
        cv_source = pil_img.resize(new_size, Image.BILINEAR)
    else:
        cv_source = pil_img

    cv_img = cv2.cvtColor(np.array(cv_source), cv2.COLOR_RGB2GRAY)

    blur_score = cv2.Laplacian(cv_img, cv2.CV_64F).var()
    brightness = float(np.mean(cv_img))
    edges = cv2.Canny(cv_img, 100, 200)
    edge_density = float(np.sum(edges > 0)) / (cv_img.shape[0] * cv_img.shape[1])

    return {
        "width": width,
        "height": height,
        "file_size_kb": round(file_size_kb, 2),
        "blur_score": round(float(blur_score), 2),
        "brightness": round(brightness, 2),
        "edge_density": round(edge_density, 4),
    }


def load_router():
    with open(ROUTER_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["label_encoder"], bundle["feature_cols"]


def build_feature_row(request_features, fast_confidence, load_stage, feature_cols):
    row = {}
    for key in ["width", "height", "file_size_kb", "blur_score", "brightness", "edge_density"]:
        if key in feature_cols:
            row[key] = request_features[key]
    if "fast_confidence" in feature_cols:
        row["fast_confidence"] = fast_confidence
    if "load_stage" in feature_cols:
        row["load_stage"] = load_stage
    return pd.DataFrame([row], columns=feature_cols)


def main():
    print(f"Loading router from {ROUTER_MODEL_PATH} ...")
    model, label_encoder, feature_cols = load_router()

    print(f"Loading {N_SAMPLE_IMAGES} sample images from {HELD_OUT_CSV} ...")
    held_out = pd.read_csv(HELD_OUT_CSV)
    sample = held_out.sample(n=min(N_SAMPLE_IMAGES, len(held_out)), random_state=42)

    feature_extraction_timings = []
    router_predict_timings = []
    skipped = 0
    skew_check_rows = []

    for _, row in sample.iterrows():
        img_path = Path(row["image_path"])
        if not img_path.exists():
            skipped += 1
            continue

        with open(img_path, "rb") as f:
            image_bytes = f.read()
        file_size_kb = len(image_bytes) / 1024

        # Skew check: does this image actually get resized? If so, compare
        # feature values with and without the cap to quantify the tradeoff
        # for real, rather than assuming it's negligible.
        probe_img = Image.open(io.BytesIO(image_bytes))
        if max(probe_img.size) > 1024:
            capped = extract_request_features(image_bytes, file_size_kb, max_dimension=1024)
            uncapped = extract_request_features(image_bytes, file_size_kb, max_dimension=100000)
            skew_check_rows.append({
                "image": img_path.name,
                "original_size": probe_img.size,
                "blur_capped": capped["blur_score"],
                "blur_uncapped": uncapped["blur_score"],
                "blur_pct_diff": round(abs(capped["blur_score"] - uncapped["blur_score"]) / max(uncapped["blur_score"], 1e-6) * 100, 1),
                "edge_capped": capped["edge_density"],
                "edge_uncapped": uncapped["edge_density"],
                "edge_pct_diff": round(abs(capped["edge_density"] - uncapped["edge_density"]) / max(uncapped["edge_density"], 1e-6) * 100, 1),
            })

        # Warm-up (skip these timings -- first calls always slower)
        for _ in range(N_WARMUP):
            extract_request_features(image_bytes, file_size_kb)

        # Time feature extraction, repeated for stability
        for _ in range(N_REPEATS_PER_IMAGE):
            t0 = time.perf_counter()
            request_features = extract_request_features(image_bytes, file_size_kb)
            feature_extraction_timings.append((time.perf_counter() - t0) * 1000)

        # Build a representative feature row using this image's real values
        feature_row = build_feature_row(request_features, fast_confidence=0.65,
                                          load_stage="light", feature_cols=feature_cols)

        # Warm-up router predict too
        for _ in range(N_WARMUP):
            model.predict(feature_row)

        for _ in range(N_REPEATS_PER_IMAGE):
            t0 = time.perf_counter()
            model.predict(feature_row)
            router_predict_timings.append((time.perf_counter() - t0) * 1000)

    if skipped:
        print(f"WARNING: skipped {skipped} images (file not found on disk).")

    def summarize(name, timings):
        print(f"\n{name}  (n={len(timings)})")
        print(f"  mean:   {statistics.mean(timings):.4f} ms")
        print(f"  median: {statistics.median(timings):.4f} ms")
        print(f"  p95:    {sorted(timings)[int(len(timings) * 0.95)]:.4f} ms")
        print(f"  min:    {min(timings):.4f} ms")
        print(f"  max:    {max(timings):.4f} ms")

    summarize("extract_request_features() [OpenCV Laplacian + Canny]", feature_extraction_timings)
    summarize("router_model.predict() [sklearn Pipeline]", router_predict_timings)

    if skew_check_rows:
        print(f"\n{'=' * 60}")
        print(f"SKEW CHECK -- {len(skew_check_rows)} image(s) in this sample exceeded 1024px and got resized")
        print("=" * 60)
        for r in skew_check_rows:
            print(f"  {r['image']} (original {r['original_size']}): "
                  f"blur {r['blur_uncapped']}->{r['blur_capped']} ({r['blur_pct_diff']}% diff), "
                  f"edge_density {r['edge_uncapped']}->{r['edge_capped']} ({r['edge_pct_diff']}% diff)")
    else:
        print(f"\nNo images in this sample exceeded 1024px -- the resize cap never triggered, "
              f"so all feature values above are identical to the un-capped version anyway "
              f"(zero skew risk observed in this sample).")

    total_mean = statistics.mean(feature_extraction_timings) + statistics.mean(router_predict_timings)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    fe_mean = statistics.mean(feature_extraction_timings)
    rp_mean = statistics.mean(router_predict_timings)
    print(f"extract_request_features(): {fe_mean:8.4f} ms  ({fe_mean / total_mean * 100:5.1f}% of combined CPU cost)")
    print(f"router_model.predict():     {rp_mean:8.4f} ms  ({rp_mean / total_mean * 100:5.1f}% of combined CPU cost)")
    print(f"Combined:                   {total_mean:8.4f} ms")
    print(f"\nCompare this combined number to the observed ~55-85ms overhead gap between "
          f"ml_router/rule_based and always_fast in the live gateway experiment. If this "
          f"local number accounts for most of that gap, the CPU-bound hypothesis is confirmed. "
          f"If it's much lower, there's still an unexplained gap (container CPU constraints, "
          f"event-loop blocking, or something else) worth investigating further.")


if __name__ == "__main__":
    main()