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
from fastapi import FastAPI, UploadFile, File, HTTPException
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

app = FastAPI(title="InferPilot Routing Gateway")
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

ROUTING_DECISIONS = Counter(
    "routing_decisions_total", "Count of routing decisions by chosen tier", ["tier"]
)
GATEWAY_LATENCY = Histogram(
    "gateway_latency_ms", "End-to-end gateway latency in ms"
)

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


@app.get("/health")
def health():
    return {"status": "ok", "concurrent_requests": _concurrent_requests}


@app.post("/route")
async def route(file: UploadFile = File(...)):
    start = time.perf_counter()
    _bump_concurrency(1)

    try:
        image_bytes = await file.read()
        file_size_kb = len(image_bytes) / 1024
        request_features = extract_request_features(image_bytes, file_size_kb)
        load_stage = estimate_load_stage()

        # Always call Fast first -- cascade design (see handoff doc, Path B)
        async with httpx.AsyncClient() as client:
            fast_resp = await client.post(
                f"{SERVICE_URLS['fast']}/predict",
                files={"file": (file.filename, image_bytes, file.content_type)},
                timeout=10.0,
            )
        if fast_resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Fast tier unavailable")

        fast_result = fast_resp.json()
        fast_confidence = fast_result["confidence"]

        feature_row = build_feature_row(request_features, fast_confidence, load_stage)
        predicted_label = router_model.predict(feature_row)[0]
        chosen_tier = label_encoder.inverse_transform([predicted_label])[0]

        ROUTING_DECISIONS.labels(tier=chosen_tier).inc()

        if chosen_tier == "fast":
            final_result = fast_result
        else:
            async with httpx.AsyncClient() as client:
                final_resp = await client.post(
                    f"{SERVICE_URLS[chosen_tier]}/predict",
                    files={"file": (file.filename, image_bytes, file.content_type)},
                    timeout=10.0,
                )
            if final_resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"{chosen_tier} tier unavailable")
            final_result = final_resp.json()

        elapsed_ms = (time.perf_counter() - start) * 1000
        GATEWAY_LATENCY.observe(elapsed_ms)

        return {
            **final_result,
            "routed_tier": chosen_tier,
            "fast_confidence_used_for_routing": fast_confidence,
            "estimated_load_stage": load_stage,
            "gateway_latency_ms": round(elapsed_ms, 2),
        }

    finally:
        _bump_concurrency(-1)
