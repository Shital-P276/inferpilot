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

import asyncio
import io
import os
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
from torchvision.transforms import v2
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

from training.models.fast_cnn import FastCNN
from training.models.correctness_gate import CorrectnessGateNet, CorrectnessGateNetV2b

# ---- CONFIG -----------------------------------------------------------
ROUTER_MODEL_PATH = "router/router_best_model.pkl"

# Single-shot Path A router: predicts tier directly from request features
# ONLY (no fast_confidence, no Fast pre-call) -- see train_router_singleshot.py.
# feature_cols confirmed from the pickle: ['width','height','file_size_kb',
# 'blur_score','brightness','edge_density','load_stage'] -- same 6 numeric +
# 1 categorical as the cascade router, minus fast_confidence.
SINGLESHOT_ROUTER_MODEL_PATH = "router/router_singleshot_model.pkl"

# CorrectnessGateNetV2b single-shot routing strategy: frozen MobileNetV3-Small
# backbone + 24K-param trainable head (0.7572 avg AUROC, up from V1's 0.7068).
# V1 (CorrectnessGateNet, correctness_gate_best.pt) is retained in the repo
# as the legacy baseline for comparison -- not loaded in production.
CORRECTNESS_GATE_CHECKPOINT_PATH = "training/checkpoints/correctness_gate_v2b_best.pt"
CORRECTNESS_GATE_LAMBDA = 0.20
CORRECTNESS_GATE_COSTS = {"fast": 0.480, "balanced": 0.947, "heavy": 0.986}
CORRECTNESS_GATE_TIERS = ["fast", "balanced", "heavy"]

SERVICE_URLS = {
    "balanced": "http://balanced:8002",
    "heavy": "http://heavy:8003",
}

# Simulated added latency for Heavy calls ONLY -- represents real cloud vision
# API latency for complex analysis (AWS Rekognition / Google Vision / Azure
# Vision, ~2-3s per published benchmarks, Aug 2026 session research) vs. Heavy's
# isolated-benchmark latency of ~6.29ms. The stress-test goal is to widen the
# absolute tier-latency gap to resemble production so routing's cost advantage
# can be measured fairly. Predictions/accuracy are NOT touched -- the sleep is
# applied AFTER Heavy's real HTTP response is received, only delaying the reply.
# Env override (e.g. SIMULATED_HEAVY_DELAY_MS=0 to disable) allows re-tests
# without a rebuild.
SIMULATED_HEAVY_DELAY_MS = int(os.environ.get("SIMULATED_HEAVY_DELAY_MS", "2500"))

FAST_CHECKPOINT_PATH = "training/checkpoints/fast_cnn_best.pt"
FAST_NUM_CLASSES = 28
FAST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FAST_CLASS_NAMES = [
    "Apple__Healthy", "Apple__Rotten",
    "Banana__Healthy", "Banana__Rotten",
    "Bellpepper__Healthy", "Bellpepper__Rotten",
    "Carrot__Healthy", "Carrot__Rotten",
    "Cucumber__Healthy", "Cucumber__Rotten",
    "Grape__Healthy", "Grape__Rotten",
    "Guava__Healthy", "Guava__Rotten",
    "Jujube__Healthy", "Jujube__Rotten",
    "Mango__Healthy", "Mango__Rotten",
    "Orange__Healthy", "Orange__Rotten",
    "Pomegranate__Healthy", "Pomegranate__Rotten",
    "Potato__Healthy", "Potato__Rotten",
    "Strawberry__Healthy", "Strawberry__Rotten",
    "Tomato__Healthy", "Tomato__Rotten",
]

_fast_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Same eval-time pipeline as train_correctness_gate.py (ToImage -> Resize(256)
# -> CenterCrop(224) -> ToDtype(scale) -> ImageNet Normalize).
_correctness_gate_preprocess = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_fast_model = None

LOAD_STAGE_THRESHOLDS = {
    "light": (0, 10),
    "heavy": (10, 40),
    "burst": (40, float("inf")),
}

VALID_STRATEGIES = {"ml_router", "always_fast", "always_heavy", "round_robin", "rule_based", "single_shot_router", "correctness_gate_router"}

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
    "gateway_latency_ms", "End-to-end gateway latency in ms", ["strategy"],
    buckets=[10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000,
             7500, 10000, 15000, 20000, 30000]
)


@app.on_event("startup")
def load_fast_model_on_startup():
    global _fast_model

    if not Path(FAST_CHECKPOINT_PATH).exists():
        raise RuntimeError(f"Fast checkpoint not found at {FAST_CHECKPOINT_PATH}")
    if len(FAST_CLASS_NAMES) != FAST_NUM_CLASSES:
        raise RuntimeError(
            f"FAST_CLASS_NAMES has {len(FAST_CLASS_NAMES)} entries but "
            f"FAST_NUM_CLASSES={FAST_NUM_CLASSES}. Fix the mismatch."
        )

    model = FastCNN(num_classes=FAST_NUM_CLASSES)
    checkpoint = torch.load(FAST_CHECKPOINT_PATH, map_location=FAST_DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(FAST_DEVICE)
    model.eval()

    _fast_model = model
    print(f"[gateway] Fast model loaded in-process on {FAST_DEVICE}")


def run_fast_inference(image_bytes: bytes) -> dict:
    if _fast_model is None:
        raise HTTPException(status_code=503, detail="Fast model not loaded yet")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = _fast_preprocess(image).unsqueeze(0).to(FAST_DEVICE)

    start = time.perf_counter()
    with torch.no_grad():
        logits = _fast_model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probs, dim=1)
    if FAST_DEVICE.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "tier": "fast",
        "predicted_class": FAST_CLASS_NAMES[predicted_idx.item()],
        "confidence": round(confidence.item(), 4),
        "latency_ms": round(latency_ms, 3),
    }

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

# ---- Load the single-shot Path A router (separate pickle, separate feature
# set -- no fast_confidence). Confirmed via inspection of the actual pickle:
# feature_cols = ['width','height','file_size_kb','blur_score','brightness',
# 'edge_density','load_stage']; classes = ['balanced','fast','heavy'].
with open(SINGLESHOT_ROUTER_MODEL_PATH, "rb") as f:
    router_singleshot_bundle = pickle.load(f)

singleshot_router_model = router_singleshot_bundle["model"]
singleshot_label_encoder = router_singleshot_bundle["label_encoder"]
singleshot_feature_cols = router_singleshot_bundle["feature_cols"]

if "fast_confidence" in singleshot_feature_cols:
    print(
        f"[gateway] WARNING: single-shot router's feature_cols unexpectedly "
        f"contains 'fast_confidence' -- that defeats the point of single-shot "
        f"(no cascade, no Fast pre-call). got feature_cols={singleshot_feature_cols}. "
        f"Check whether you loaded the wrong pickle."
    )

# ---- Load the CorrectnessGateNetV2b single-shot router (frozen
# MobileNetV3-Small backbone + 24K-param trainable head). Predicts P(correct)
# per tier directly from raw image pixels. V1 (CorrectnessGateNet) is retained
# in correctness_gate.py as legacy baseline, not loaded here.
if not Path(CORRECTNESS_GATE_CHECKPOINT_PATH).exists():
    raise RuntimeError(f"CorrectnessGateNetV2b checkpoint not found at {CORRECTNESS_GATE_CHECKPOINT_PATH}")

_gate_checkpoint = torch.load(CORRECTNESS_GATE_CHECKPOINT_PATH, map_location=FAST_DEVICE)
_gate_state_dict = _gate_checkpoint.get("model_state_dict", _gate_checkpoint) if isinstance(_gate_checkpoint, dict) else _gate_checkpoint
correctness_gate_model = CorrectnessGateNetV2b(num_tiers=3)
correctness_gate_model.load_state_dict(_gate_state_dict)
correctness_gate_model.to(FAST_DEVICE)
correctness_gate_model.eval()
print(f"[gateway] CorrectnessGateNetV2b loaded in-process on {FAST_DEVICE}")


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


def build_singleshot_feature_row(request_features: dict, load_stage: str):
    """Same shape as build_feature_row, but against singleshot_feature_cols
    and with no fast_confidence input at all -- there's no Fast result to
    draw it from, by design."""
    row = {}
    for key in ["width", "height", "file_size_kb", "blur_score", "brightness", "edge_density"]:
        if key in singleshot_feature_cols:
            row[key] = request_features[key]

    if "load_stage" in singleshot_feature_cols:
        row["load_stage"] = load_stage

    return pd.DataFrame([row], columns=singleshot_feature_cols)


def route_correctness_gate(image_bytes: bytes) -> str:
    """CorrectnessGateNet single-shot routing decision from raw image pixels.

    Preprocesses the image with the training eval pipeline, runs one CNN
    forward pass to get P(correct) for each tier (ordered fast, balanced,
    heavy), then picks argmax over tiers of (P_correct_tier - LAMBDA*cost_tier).
    """
    if correctness_gate_model is None:
        raise HTTPException(status_code=503, detail="CorrectnessGateNet not loaded yet")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = _correctness_gate_preprocess(image).unsqueeze(0).to(FAST_DEVICE)

    with torch.no_grad():
        tier_correctness = correctness_gate_model(input_tensor)[0]  # [3] -> (fast, balanced, heavy)

    scores = [
        tier_correctness[i].item() - CORRECTNESS_GATE_LAMBDA * CORRECTNESS_GATE_COSTS[tier]
        for i, tier in enumerate(CORRECTNESS_GATE_TIERS)
    ]
    chosen_tier = CORRECTNESS_GATE_TIERS[int(max(range(len(scores)), key=scores.__getitem__))]
    return chosen_tier


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

        # single_shot_router / correctness_gate_router deliberately excluded:
        # neither ever needs Fast's confidence -- that's the whole architectural
        # point (one hop, no cascade). Only ml_router/rule_based (cascade
        # strategies) and always_fast (needs the actual Fast answer) trigger
        # the Fast call.
        NEEDS_FAST_CONFIDENCE = {"ml_router", "rule_based"}

        fast_result = None
        fast_confidence = None

        if strategy in NEEDS_FAST_CONFIDENCE or strategy == "always_fast":
            t0 = time.perf_counter()
            fast_result = run_fast_inference(image_bytes)
            timings_ms["fast_call_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            fast_confidence = fast_result["confidence"]

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
            t0 = time.perf_counter()
            if fast_confidence < RULE_BASED_CONFIDENCE_THRESHOLD:
                chosen_tier = RULE_BASED_ESCALATE_TO
            else:
                chosen_tier = "fast"
            timings_ms["rule_decision_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        elif strategy == "single_shot_router":
            # Path A: predict tier directly from request features alone --
            # no Fast pre-call, exactly one network hop downstream (to
            # whichever tier gets picked). fast_result/fast_confidence stay
            # None here, same as always_heavy/round_robin.
            t0 = time.perf_counter()
            feature_row = build_singleshot_feature_row(request_features, load_stage)
            predicted_label = singleshot_router_model.predict(feature_row)[0]
            chosen_tier = singleshot_label_encoder.inverse_transform([predicted_label])[0]
            timings_ms["router_predict_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        elif strategy == "correctness_gate_router":
            # Path B single-shot: one CNN forward pass on the raw image (no
            # Fast pre-call, no hand-engineered feature row). Picks the tier
            # maximizing (P_correct_tier - LAMBDA*cost_tier).
            t0 = time.perf_counter()
            chosen_tier = route_correctness_gate(image_bytes)
            timings_ms["router_predict_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        ROUTING_DECISIONS.labels(tier=chosen_tier, strategy=strategy).inc()

        if chosen_tier == "fast" and fast_result is not None:
            final_result = fast_result
            timings_ms["escalated_call_ms"] = 0.0
        elif chosen_tier == "fast":
            t0 = time.perf_counter()
            final_result = run_fast_inference(image_bytes)
            timings_ms["escalated_call_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        else:
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

            # Simulated cloud-vision latency for Heavy ONLY: real prediction is
            # already in final_result, so accuracy is untouched -- this only delays
            # the reply to widen the tier-latency gap for the stress test. Applies
            # to every strategy that actually dispatches to Heavy (all converge
            # here); never delays fast/balanced.
            if chosen_tier == "heavy" and SIMULATED_HEAVY_DELAY_MS > 0:
                await asyncio.sleep(SIMULATED_HEAVY_DELAY_MS / 1000)
                timings_ms["simulated_heavy_delay_ms"] = SIMULATED_HEAVY_DELAY_MS

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