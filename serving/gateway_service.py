"""
gateway_service.py

The live routing gateway. Sits in front of fast/balanced/heavy services.

Supports a `strategy` query param on /route:
    - "ml_router" (default) -- original behavior, unchanged
    - "always_fast" -- NEW, skips the router, always forwards to Fast

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
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# ---- CONFIG -----------------------------------------------------------
ROUTER_MODEL_PATH = "router/router_best_model.pkl"

SERVICE_URLS = {
    "fast": "http://fast:8001",
    "balanced": "http://balanced:8002",
    "heavy": "http://heavy:8003",
}

LOAD_STAGE_THRESHOLDS = {
    "light": (0, 10),
    "heavy": (10, 40),
    "burst": (40, float("inf")),
}
# -------------------------------------------------------------------------

app = FastAPI(title="InferPilot Routing Gateway")
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

ROUTING_DECISIONS = Counter(
    "routing_decisions_total", "Count of routing decisions by chosen tier", ["tier", "strategy"]
)
GATEWAY_LATENCY = Histogram(
    "gateway_latency_ms", "End-to-end gateway latency in ms", ["strategy"]
)

_concurrent_lock = threading.Lock()
_concurrent_requests = 0

# Round-robin state
_round_robin_lock = threading.Lock()
_round_robin_index = 0
_ROUND_ROBIN_TIERS = ["fast", "balanced", "heavy"]


def get_next_round_robin_tier():
    global _round_robin_index
    with _round_robin_lock:
        tier = _ROUND_ROBIN_TIERS[_round_robin_index % len(_ROUND_ROBIN_TIERS)]
        _round_robin_index += 1
        return tier

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

# Rule-based thresholds (from train_router.py's confidence-threshold baseline,
# tuned on train set, seed=42, reproducible — see handoff docs)
RULE_LOW_THRESH = 0.50
RULE_HIGH_THRESH = 0.66


def rule_based_tier(fast_confidence: float) -> str:
    if fast_confidence >= RULE_HIGH_THRESH:
        return "fast"
    elif fast_confidence >= RULE_LOW_THRESH:
        return "balanced"
    else:
        return "heavy"

# ---- Load the trained router once at startup ----
with open(ROUTER_MODEL_PATH, "rb") as f:
    router_bundle = pickle.load(f)

router_model = router_bundle["model"]
label_encoder = router_bundle["label_encoder"]
feature_cols = router_bundle["feature_cols"]

if "load_stage" not in feature_cols:
    print(f"[gateway] WARNING: expected a raw 'load_stage' column in feature_cols, got {feature_cols}")
if "fast_confidence" not in feature_cols:
    print(f"[gateway] WARNING: expected 'fast_confidence' in feature_cols, got {feature_cols}")


def extract_request_features(image_bytes: bytes, file_size_kb: float):
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = pil_img.size
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(cv_img, cv2.CV_64F).var()
    brightness = float(np.mean(cv_img))
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
    row = {}
    for key in ["width", "height", "file_size_kb", "blur_score", "brightness", "edge_density"]:
        if key in feature_cols:
            row[key] = request_features[key]
    if "fast_confidence" in feature_cols:
        row["fast_confidence"] = fast_confidence
    if "load_stage" in feature_cols:
        row["load_stage"] = load_stage
    return pd.DataFrame([row], columns=feature_cols)


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


@app.get("/health")
def health():
    return {"status": "ok", "concurrent_requests": _concurrent_requests}


@app.post("/route")
async def route(
    file: UploadFile = File(...),
    strategy: str = Query("ml_router", description="ml_router | always_fast | always_heavy | round_robin | rule_based"),
):
    start = time.perf_counter()
    _bump_concurrency(1)

    try:
        image_bytes = await file.read()
        file_size_kb = len(image_bytes) / 1024

        # ---- always_fast: skip everything, just forward to Fast ----
        if strategy == "always_fast":
            fast_result = await call_tier("fast", file, image_bytes)
            chosen_tier = "fast"

            ROUTING_DECISIONS.labels(tier=chosen_tier, strategy=strategy).inc()
            elapsed_ms = (time.perf_counter() - start) * 1000
            GATEWAY_LATENCY.labels(strategy=strategy).observe(elapsed_ms)

            return {
                **fast_result,
                "routed_tier": chosen_tier,
                "strategy": strategy,
                "gateway_latency_ms": round(elapsed_ms, 2),
            }

        # ---- always_heavy: skip everything, always forward to Heavy ----
        elif strategy == "always_heavy":
            heavy_result = await call_tier("heavy", file, image_bytes)
            chosen_tier = "heavy"

            ROUTING_DECISIONS.labels(tier=chosen_tier, strategy=strategy).inc()
            elapsed_ms = (time.perf_counter() - start) * 1000
            GATEWAY_LATENCY.labels(strategy=strategy).observe(elapsed_ms)

            return {
                **heavy_result,
                "routed_tier": chosen_tier,
                "strategy": strategy,
                "gateway_latency_ms": round(elapsed_ms, 2),
            }

        # ---- round_robin: cycles fast -> balanced -> heavy -> fast -> ... ----
        elif strategy == "round_robin":
            chosen_tier = get_next_round_robin_tier()
            result = await call_tier(chosen_tier, file, image_bytes)

            ROUTING_DECISIONS.labels(tier=chosen_tier, strategy=strategy).inc()
            elapsed_ms = (time.perf_counter() - start) * 1000
            GATEWAY_LATENCY.labels(strategy=strategy).observe(elapsed_ms)

            return {
                **result,
                "routed_tier": chosen_tier,
                "strategy": strategy,
                "gateway_latency_ms": round(elapsed_ms, 2),
            }

        # ---- rule_based: hand-written confidence thresholds, no learning ----
        elif strategy == "rule_based":
            fast_result = await call_tier("fast", file, image_bytes)
            fast_confidence = fast_result["confidence"]
            chosen_tier = rule_based_tier(fast_confidence)

            ROUTING_DECISIONS.labels(tier=chosen_tier, strategy=strategy).inc()

            if chosen_tier == "fast":
                final_result = fast_result
            else:
                final_result = await call_tier(chosen_tier, file, image_bytes)

            elapsed_ms = (time.perf_counter() - start) * 1000
            GATEWAY_LATENCY.labels(strategy=strategy).observe(elapsed_ms)

            return {
                **final_result,
                "routed_tier": chosen_tier,
                "strategy": strategy,
                "fast_confidence_used_for_routing": fast_confidence,
                "gateway_latency_ms": round(elapsed_ms, 2),
            }

        # ---- ml_router: original, unchanged behavior ----
        elif strategy == "ml_router":
            request_features = extract_request_features(image_bytes, file_size_kb)
            load_stage = estimate_load_stage()

            fast_result = await call_tier("fast", file, image_bytes)
            fast_confidence = fast_result["confidence"]

            feature_row = build_feature_row(request_features, fast_confidence, load_stage)
            predicted_label = router_model.predict(feature_row)[0]
            chosen_tier = label_encoder.inverse_transform([predicted_label])[0]

            ROUTING_DECISIONS.labels(tier=chosen_tier, strategy=strategy).inc()

            if chosen_tier == "fast":
                final_result = fast_result
            else:
                final_result = await call_tier(chosen_tier, file, image_bytes)

            elapsed_ms = (time.perf_counter() - start) * 1000
            GATEWAY_LATENCY.labels(strategy=strategy).observe(elapsed_ms)

            return {
                **final_result,
                "routed_tier": chosen_tier,
                "strategy": strategy,
                "fast_confidence_used_for_routing": fast_confidence,
                "estimated_load_stage": load_stage,
                "gateway_latency_ms": round(elapsed_ms, 2),
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")

    finally:
        _bump_concurrency(-1)