"""
gateway_service.py

The live routing gateway. Sits in front of fast/balanced/heavy services.
Receives an image, extracts request-level features, always queries Fast
first for its confidence, feeds everything into the trained router model,
and forwards to whichever tier the router picks.

BEFORE RUNNING -- verify the one-hot column naming:
    Load router_best_model.pkl once, print feature_cols, and confirm the
    load_stage column names below (LOAD_STAGE_COLUMNS) match exactly what
    your train_router.py produced. If they don't match, predictions will
    be silently wrong -- this script will warn you loudly instead, but you
    still need to fix the mapping before trusting any output.

USAGE:
    uvicorn gateway_service:app --host 0.0.0.0 --port 8000
"""

import io
import time
import pickle
import threading
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import httpx
import torch
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

from training.models.fast_cnn import FastCNN

# ---- CONFIG -----------------------------------------------------------
ROUTER_MODEL_PATH = "router/router_best_model.pkl"

# All tiers, including Fast, are served as separate microservices and reached
# over HTTP so the gateway does not do local CPU inference for the model path.
SERVICE_URLS = {
    "fast": "http://fast:8001",
    "balanced": "http://balanced:8002",
    "heavy": "http://heavy:8003",
}

# Concurrency thresholds used to estimate current load stage.
# These are a practical proxy for the light/heavy/burst labels used during
# training (k6 VUs) -- tune these numbers against your own k6 stage
# definitions if the mapping feels off in practice.
LOAD_STAGE_THRESHOLDS = {
    "light": (0, 10),
    "heavy": (10, 40),
    "burst": (40, float("inf")),
}

# NOTE: load_stage is passed as a single raw string column ("light"/
# "heavy"/"burst") -- the trained pipeline has its own OneHotEncoder
# (via ColumnTransformer) that handles the encoding internally. Confirmed
# by inspecting feature_cols in router_best_model.pkl, which lists
# "load_stage" as one plain column, not pre-split dummy columns.
# -------------------------------------------------------------------------

VALID_STRATEGIES = {"ml_router", "always_fast", "always_heavy", "round_robin", "rule_based", "single_shot_router"}

# Rule-based baseline: hand-written threshold on Fast's confidence.
# This is the "what a human would hand-code without ML" comparison point --
# deliberately simple, single threshold, no load-awareness.
RULE_BASED_CONFIDENCE_THRESHOLD = 0.7
RULE_BASED_ESCALATE_TO = "heavy"

_round_robin_lock = threading.Lock()
_round_robin_counter = 0
_ROUND_ROBIN_ORDER = ["fast", "balanced", "heavy"]


def next_round_robin_tier():
    global _round_robin_counter
    with _round_robin_lock:
        tier = _ROUND_ROBIN_ORDER[_round_robin_counter % len(_ROUND_ROBIN_ORDER)]
        _round_robin_counter += 1
        return tier

app = FastAPI(title="InferPilot Routing Gateway")
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

ROUTING_DECISIONS = Counter(
    "routing_decisions_total", "Count of routing decisions by chosen tier and strategy", ["tier", "strategy"]
)
GATEWAY_LATENCY = Histogram(
    "gateway_latency_ms",
    "End-to-end gateway latency in ms",
    ["strategy"],
    buckets=[10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000],
)


# The gateway intentionally does not run the Fast model locally; it calls the
# deployed Fast service over HTTP so the performance comparison reflects the
# actual production architecture used for routing.

# ---- Load the trained SINGLE-SHOT router once at startup (optional, if present) ----
SINGLESHOT_MODEL_PATH = "router/router_singleshot_model.pkl"

try:
    with open(SINGLESHOT_MODEL_PATH, "rb") as f:
        singleshot_bundle = pickle.load(f)
    singleshot_model = singleshot_bundle["model"]
    singleshot_label_encoder = singleshot_bundle["label_encoder"]
    singleshot_feature_cols = singleshot_bundle["feature_cols"]
except FileNotFoundError:
    singleshot_model = None
    singleshot_label_encoder = None
    singleshot_feature_cols = []

if singleshot_model is not None:
    if "fast_confidence" in singleshot_feature_cols:
        print(f"[gateway] WARNING: single-shot router includes fast_confidence; that defeats the purpose of a true single-shot path.")
    if "load_stage" not in singleshot_feature_cols:
        print(f"[gateway] WARNING: expected 'load_stage' in singleshot_feature_cols, got {singleshot_feature_cols}")


async def call_tier(tier: str, file: UploadFile, image_bytes: bytes):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{SERVICE_URLS[tier]}/predict",
            files={"file": (file.filename, image_bytes, file.content_type)},
            timeout=10.0,
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"{tier} tier unavailable")
    return resp.json()

# Track in-flight requests for a crude live load estimate
_concurrent_lock = threading.Lock()
_concurrent_requests = 0


def _bump_concurrency(delta):
    global _concurrent_requests
    with _concurrent_lock:
        _concurrent_requests += delta
        return _concurrent_requests


def estimate_load_stage():
    current = _concurrent_requests
    for stage, (low, high) in LOAD_STAGE_THRESHOLDS.items():
        if low <= current < high:
            return stage
    return "burst"


# ---- Load the trained router once at startup ----
with open(ROUTER_MODEL_PATH, "rb") as f:
    router_bundle = pickle.load(f)

router_model = router_bundle["model"]
label_encoder = router_bundle["label_encoder"]
feature_cols = router_bundle["feature_cols"]

if "load_stage" not in feature_cols:
    print(
        f"[gateway] WARNING: expected a raw 'load_stage' column in feature_cols, "
        f"but got feature_cols={feature_cols}. Check whether your router pipeline "
        f"actually expects pre-encoded columns instead -- if so, revert to manual "
        f"one-hot encoding here."
    )
if "fast_confidence" not in feature_cols:
    print(
        f"[gateway] WARNING: expected 'fast_confidence' in feature_cols, "
        f"got feature_cols={feature_cols}. Confirm the exact name used in training."
    )


def extract_request_features(image_bytes: bytes, file_size_kb: float):
    """Mirrors whatever feature extraction generate_utility_labels.py used.
    Adjust if your training pipeline computed blur/brightness/edge_density
    differently (e.g. different color space or kernel size)."""
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = pil_img.size

    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)

    # Blur score: variance of Laplacian -- lower means blurrier
    blur_score = cv2.Laplacian(cv_img, cv2.CV_64F).var()

    # Brightness: mean pixel intensity
    brightness = float(np.mean(cv_img))

    # Edge density: fraction of pixels flagged as edges by Canny
    edges = cv2.Canny(cv_img, 100, 200)
    edge_density = float(np.sum(edges > 0)) / (width * height)

    return {
        "width": width,
        "height": height,
        "file_size_kb": round(file_size_kb, 2),
        "blur_score": round(float(blur_score), 2),
        "brightness": round(brightness, 2),
        "edge_density": round(edge_density, 4),
    }


def build_feature_row(request_features: dict, fast_confidence: float, load_stage: str):
    """Builds a single-row DataFrame matching feature_cols exactly.
    load_stage is passed as its raw string value -- the trained pipeline's
    own ColumnTransformer + OneHotEncoder handles the encoding internally,
    confirmed from the pickle's feature_cols (a single 'load_stage' column,
    not pre-split dummies)."""
    row = {}
    for key in ["width", "height", "file_size_kb", "blur_score", "brightness", "edge_density"]:
        if key in feature_cols:
            row[key] = request_features[key]

    if "fast_confidence" in feature_cols:
        row["fast_confidence"] = fast_confidence

    if "load_stage" in feature_cols:
        row["load_stage"] = load_stage

    return pd.DataFrame([row], columns=feature_cols)


def build_singleshot_feature_row(request_features: dict, load_stage: str):
    """Builds the feature row for the single-shot tier selector."""
    row = {}
    for key in ["width", "height", "file_size_kb", "blur_score", "brightness", "edge_density"]:
        if key in singleshot_feature_cols:
            row[key] = request_features[key]
    if "load_stage" in singleshot_feature_cols:
        row["load_stage"] = load_stage
    return pd.DataFrame([row], columns=singleshot_feature_cols)


@app.get("/health")
def health():
    return {"status": "ok", "concurrent_requests": _concurrent_requests}


@app.post("/route")
async def route(file: UploadFile = File(...), strategy: str = "ml_router"):
    if strategy not in VALID_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy '{strategy}'. Must be one of {sorted(VALID_STRATEGIES)}",
        )

    start = time.perf_counter()
    _bump_concurrency(1)
    timings_ms = {}

    try:
        image_bytes = await file.read()
        file_size_kb = len(image_bytes) / 1024

        t0 = time.perf_counter()
        request_features = extract_request_features(image_bytes, file_size_kb)
        timings_ms["feature_extraction_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        load_stage = estimate_load_stage()

        # Only run Fast when the strategy actually needs its output.
        # ml_router needs fast_confidence as a router feature; rule_based needs
        # it to evaluate the threshold. always_heavy/round_robin don't use it
        # at all -- running it for them would inflate their latency with a
        # lookup they'd never do in a real standalone deployment.
        # NOTE: Fast now runs IN-PROCESS (see run_fast_inference above), not
        # over HTTP -- this eliminates the ~40-50ms network round-trip that
        # was previously the dominant cost here for a sub-millisecond model.
        NEEDS_FAST_CONFIDENCE = {"ml_router", "rule_based"}

        fast_result = None
        fast_confidence = None

        if strategy in NEEDS_FAST_CONFIDENCE or strategy == "always_fast":
            t0 = time.perf_counter()
            fast_result = await call_tier("fast", file, image_bytes)
            timings_ms["fast_call_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            fast_confidence = fast_result["confidence"]

        # ---- Strategy-specific decision (this is the only part that differs) ----
        if strategy == "ml_router":
            t0 = time.perf_counter()
            feature_row = build_feature_row(request_features, fast_confidence, load_stage)
            predicted_label = router_model.predict(feature_row)[0]
            chosen_tier = label_encoder.inverse_transform([predicted_label])[0]
            timings_ms["router_predict_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        elif strategy == "always_fast":
            chosen_tier = "fast"

        elif strategy == "always_heavy":
            chosen_tier = "heavy"

        elif strategy == "round_robin":
            chosen_tier = next_round_robin_tier()

        elif strategy == "rule_based":
            # Hand-written threshold baseline: escalate only if Fast isn't
            # confident enough. No load-awareness, no learned weighting --
            # this is deliberately the simplest thing a human would hand-code.
            t0 = time.perf_counter()
            if fast_confidence < RULE_BASED_CONFIDENCE_THRESHOLD:
                chosen_tier = RULE_BASED_ESCALATE_TO
            else:
                chosen_tier = "fast"
            timings_ms["rule_decision_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        elif strategy == "single_shot_router":
            if singleshot_model is None:
                raise HTTPException(status_code=400, detail="single_shot_router model not available")
            t0 = time.perf_counter()
            feature_row = build_singleshot_feature_row(request_features, load_stage)
            predicted_label = singleshot_model.predict(feature_row)[0]
            chosen_tier = singleshot_label_encoder.inverse_transform([predicted_label])[0]
            timings_ms["router_predict_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        # ---------------------------------------------------------------------

        ROUTING_DECISIONS.labels(tier=chosen_tier, strategy=strategy).inc()

        if chosen_tier == "fast" and fast_result is not None:
            final_result = fast_result
            timings_ms["escalated_call_ms"] = 0.0
        elif chosen_tier == "fast":
            # Fast is always called through the service endpoint rather than running
            # the model in the gateway process, so all strategies use the same API path.
            t0 = time.perf_counter()
            final_result = await call_tier("fast", file, image_bytes)
            timings_ms["escalated_call_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        else:
            # balanced or heavy -- these remain separate microservices over HTTP.
            t0 = time.perf_counter()
            async with httpx.AsyncClient() as client:
                final_resp = await client.post(
                    f"{SERVICE_URLS[chosen_tier]}/predict",
                    files={"file": (file.filename, image_bytes, file.content_type)},
                    timeout=10.0,
                )
            timings_ms["escalated_call_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            if final_resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"{chosen_tier} tier unavailable")
            final_result = final_resp.json()

        elapsed_ms = (time.perf_counter() - start) * 1000
        GATEWAY_LATENCY.labels(strategy=strategy).observe(elapsed_ms)

        return {
            **final_result,
            "routed_tier": chosen_tier,
            "strategy": strategy,
            "fast_confidence_used_for_routing": fast_confidence,
            "estimated_load_stage": load_stage,
            "gateway_latency_ms": round(elapsed_ms, 2),
            "timings_ms": timings_ms,
        }

    finally:
        _bump_concurrency(-1)