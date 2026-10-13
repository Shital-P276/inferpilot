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
from torchvision.transforms import v2
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

from training.models.correctness_gate import CorrectnessGateNet, CorrectnessGateNetV2b, CorrectnessGateNetV3

from serving.resilience import (
    BREAKERS,
    BULKHEAD,
    FALLBACKS,
    TierUnavailable,
    breaker_snapshot,
    fallback_chain,
    maybe_inject_fault,
    FAULT_TIER,
    FAULT_RATE,
)


async def call_tier(tier: str, image_bytes: bytes, filename, content_type):
    """Single guarded call to a tier service: breaker + bulkhead + fault
    injection. Returns (result_json, elapsed_ms). Raises TierUnavailable when
    the tier is breaker-blocked, saturated, fault-injected, erroring, or
    unreachable -- every such outcome also records a breaker failure so chaos
    traces trip the breaker realistically."""
    if not BREAKERS[tier].allow_request():
        raise TierUnavailable(f"breaker {BREAKERS[tier].state} for tier={tier}")
    async with BULKHEAD.slot(tier):
        try:
            maybe_inject_fault(tier)
        except TierUnavailable:
            BREAKERS[tier].record_failure()
            raise
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{SERVICE_URLS[tier]}/predict",
                    files={"file": (filename, image_bytes, content_type)},
                    timeout=10.0,
                )
        except Exception as exc:
            BREAKERS[tier].record_failure()
            raise TierUnavailable(f"tier={tier} unreachable: {type(exc).__name__}") from exc
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        if resp.status_code != 200:
            BREAKERS[tier].record_failure()
            raise TierUnavailable(f"tier={tier} returned {resp.status_code}")
        BREAKERS[tier].record_success()
        return resp.json(), elapsed_ms

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
# CorrectnessGateNetV3: same backbone with last block unfrozen + dual heads
# (correctness + confidence auxiliary, 0.7724 avg AUROC). Loaded alongside V2b
# — additive, not a replacement; V2b kept as fallback/comparison.
CORRECTNESS_GATE_V3_CHECKPOINT_PATH = "training/checkpoints/correctness_gate_v3_best.pt"
# CorrectnessGateNetV3a: ablation winner (unfrozen backbone + single correctness
# head, 0.7810 test avg AUROC vs V3's 0.7724). Same V3 architecture with
# dual_head=False -- forward returns one [3] tensor, no tuple unpack.
# Checkpoint dict is {"model_state_dict","val_avg_auroc","flags"} (no
# class_names); loader uses .get() so this is fine. Additive, like V3.
CORRECTNESS_GATE_V3A_CHECKPOINT_PATH = "training/checkpoints/correctness_gate_v3a_best.pt"
# TODO: CORRECTNESS_GATE_LAMBDA = 0.20 was tuned against the OLD cost landscape
# (0.480-0.986, ~2x range). The new iWildCam landscape is 0.0367-1.0 (~27x range)
# after recomputing from router/model_benchmarks.json. 0.20 has NOT been
# re-validated against the new costs and likely needs re-sweeping (lambda
# sweep over correctness_gate with new costs) before trusting
# hybrid_gate/correctness_gate_router behavior under the new architecture.
CORRECTNESS_GATE_LAMBDA = 0.20
# Recomputed from new router/model_benchmarks.json using the same
# 0.7*gpu_ratio + 0.3*param_ratio formula as
# training/train_correctness_gate.py::load_resource_norm()
# (old values were {"fast": 0.480, "balanced": 0.947, "heavy": 0.986}).
CORRECTNESS_GATE_COSTS = {"fast": 0.0367, "balanced": 0.1515, "heavy": 1.0000}
CORRECTNESS_GATE_TIERS = ["fast", "balanced", "heavy"]
# Device for CorrectnessGateNetV2b: "cuda" (default, same as before) or "cpu".
# CPU mode frees GPU cycles for Balanced/Heavy inference under concurrent load.
# k6 load test (Aug 2026): CPU matches GPU latency (p50 ~611ms) while improving
# throughput (13.4 -> 14.0 req/s) and reducing GPU contention (75.7% -> 57.3%).
CORRECTNESS_GATE_DEVICE = torch.device(os.environ.get("CORRECTNESS_GATE_DEVICE", "cpu"))

SERVICE_URLS = {
    "fast": "http://fast:8001",
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
SIMULATED_HEAVY_DELAY_MS = int(os.environ.get("SIMULATED_HEAVY_DELAY_MS", "0"))

# Same eval-time pipeline as train_correctness_gate.py (ToImage -> Resize(256)
# -> CenterCrop(224) -> ToDtype(scale) -> ImageNet Normalize).
_correctness_gate_preprocess = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

LOAD_STAGE_THRESHOLDS = {
    "light": (0, 10),
    "heavy": (10, 40),
    "burst": (40, float("inf")),
}

VALID_STRATEGIES = {"ml_router", "always_fast", "always_heavy", "round_robin", "rule_based", "single_shot_router", "correctness_gate_router", "correctness_gate_v3_router", "correctness_gate_v3a_router", "hybrid_gate", "hybrid_dumb"}

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
# CORRECTNESS_GATE_DEVICE env var controls GPU vs CPU placement.
if not Path(CORRECTNESS_GATE_CHECKPOINT_PATH).exists():
    raise RuntimeError(f"CorrectnessGateNetV2b checkpoint not found at {CORRECTNESS_GATE_CHECKPOINT_PATH}")

_gate_checkpoint = torch.load(CORRECTNESS_GATE_CHECKPOINT_PATH, map_location=CORRECTNESS_GATE_DEVICE)
_gate_state_dict = _gate_checkpoint.get("model_state_dict", _gate_checkpoint) if isinstance(_gate_checkpoint, dict) else _gate_checkpoint
correctness_gate_model = CorrectnessGateNetV2b(num_tiers=3)
correctness_gate_model.load_state_dict(_gate_state_dict)
correctness_gate_model.to(CORRECTNESS_GATE_DEVICE)
correctness_gate_model.eval()
print(f"[gateway] CorrectnessGateNetV2b loaded on {CORRECTNESS_GATE_DEVICE}")

# ---- Load CorrectnessGateNetV3 alongside V2b (additive, not replacement)
# Same CPU-inside-gateway design rule as V2b (CORRECTNESS_GATE_DEVICE=cpu by default).
# V3 uses correctness_head for routing only; confidence_head is auxiliary/training-only
# per diagnose_v3_confidence_head.py (not reliable for decisions) — ignored here.
if not Path(CORRECTNESS_GATE_V3_CHECKPOINT_PATH).exists():
    raise RuntimeError(f"CorrectnessGateNetV3 checkpoint not found at {CORRECTNESS_GATE_V3_CHECKPOINT_PATH}")

_gate_v3_checkpoint = torch.load(CORRECTNESS_GATE_V3_CHECKPOINT_PATH, map_location=CORRECTNESS_GATE_DEVICE)
_gate_v3_state_dict = _gate_v3_checkpoint.get("model_state_dict", _gate_v3_checkpoint) if isinstance(_gate_v3_checkpoint, dict) else _gate_v3_checkpoint
correctness_gate_v3_model = CorrectnessGateNetV3(num_tiers=3)
correctness_gate_v3_model.load_state_dict(_gate_v3_state_dict)
correctness_gate_v3_model.to(CORRECTNESS_GATE_DEVICE)
correctness_gate_v3_model.eval()
print(f"[gateway] CorrectnessGateNetV3 loaded on {CORRECTNESS_GATE_DEVICE}")

# ---- Load CorrectnessGateNetV3a alongside V3 (additive, not replacement)
# Ablation winner: unfrozen last backbone block + SINGLE correctness head
# (no confidence_head at all -- dual_head=False, so forward returns one [3]
# tensor, not a tuple). Same CPU placement, lambda, and costs as V2b/V3.
if not Path(CORRECTNESS_GATE_V3A_CHECKPOINT_PATH).exists():
    raise RuntimeError(f"CorrectnessGateNetV3a checkpoint not found at {CORRECTNESS_GATE_V3A_CHECKPOINT_PATH}")

_gate_v3a_checkpoint = torch.load(CORRECTNESS_GATE_V3A_CHECKPOINT_PATH, map_location=CORRECTNESS_GATE_DEVICE)
_gate_v3a_state_dict = _gate_v3a_checkpoint.get("model_state_dict", _gate_v3a_checkpoint) if isinstance(_gate_v3a_checkpoint, dict) else _gate_v3a_checkpoint
correctness_gate_v3a_model = CorrectnessGateNetV3(num_tiers=3, unfreeze_last_block=True, dual_head=False)
correctness_gate_v3a_model.load_state_dict(_gate_v3a_state_dict)
correctness_gate_v3a_model.to(CORRECTNESS_GATE_DEVICE)
correctness_gate_v3a_model.eval()
print(f"[gateway] CorrectnessGateNetV3a loaded on {CORRECTNESS_GATE_DEVICE}")


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
    input_tensor = _correctness_gate_preprocess(image).unsqueeze(0).to(CORRECTNESS_GATE_DEVICE)

    with torch.no_grad():
        tier_correctness = correctness_gate_model(input_tensor)[0]  # [3] -> (fast, balanced, heavy)

    scores = [
        tier_correctness[i].item() - CORRECTNESS_GATE_LAMBDA * CORRECTNESS_GATE_COSTS[tier]
        for i, tier in enumerate(CORRECTNESS_GATE_TIERS)
    ]
    chosen_tier = CORRECTNESS_GATE_TIERS[int(max(range(len(scores)), key=scores.__getitem__))]
    return chosen_tier


def route_correctness_gate_v3(image_bytes: bytes) -> str:
    """CorrectnessGateNetV3 single-shot routing (correctness_head only).

    Same pattern as route_correctness_gate (V2b) — preprocess + forward,
    argmax over (P_correct - LAMBDA*cost). V3's confidence_head is ignored
    entirely for routing (auxiliary-training-only per diagnose_v3_confidence_head.py).
    Uses same CORRECTNESS_GATE_LAMBDA/costs and CPU placement as V2b.
    """
    if correctness_gate_v3_model is None:
        raise HTTPException(status_code=503, detail="CorrectnessGateNetV3 not loaded yet")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = _correctness_gate_preprocess(image).unsqueeze(0).to(CORRECTNESS_GATE_DEVICE)

    with torch.no_grad():
        tier_correctness, _ = correctness_gate_v3_model(input_tensor)  # ignore confidence_head
        tier_correctness = tier_correctness[0]  # [3] -> (fast, balanced, heavy)

    scores = [
        tier_correctness[i].item() - CORRECTNESS_GATE_LAMBDA * CORRECTNESS_GATE_COSTS[tier]
        for i, tier in enumerate(CORRECTNESS_GATE_TIERS)
    ]
    chosen_tier = CORRECTNESS_GATE_TIERS[int(max(range(len(scores)), key=scores.__getitem__))]
    return chosen_tier


def route_correctness_gate_v3a(image_bytes: bytes) -> str:
    """CorrectnessGateNetV3a single-shot routing (ablation winner).

    Same pattern as route_correctness_gate_v3, but the model has NO
    confidence_head (dual_head=False), so forward returns a single [3]
    correctness tensor -- no tuple unpack. Same CORRECTNESS_GATE_LAMBDA,
    costs, and CPU placement as V2b/V3.
    """
    if correctness_gate_v3a_model is None:
        raise HTTPException(status_code=503, detail="CorrectnessGateNetV3a not loaded yet")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = _correctness_gate_preprocess(image).unsqueeze(0).to(CORRECTNESS_GATE_DEVICE)

    with torch.no_grad():
        tier_correctness = correctness_gate_v3a_model(input_tensor)[0]  # [3] -> (fast, balanced, heavy)

    scores = [
        tier_correctness[i].item() - CORRECTNESS_GATE_LAMBDA * CORRECTNESS_GATE_COSTS[tier]
        for i, tier in enumerate(CORRECTNESS_GATE_TIERS)
    ]
    chosen_tier = CORRECTNESS_GATE_TIERS[int(max(range(len(scores)), key=scores.__getitem__))]
    return chosen_tier


def route_hybrid_gate(image_bytes: bytes, fast_confidence: float) -> str:
    """Hybrid routing: rule_based default, correctness_gate overrides in uncertain band.

    If fast_confidence is OUTSIDE [0.60, 0.80]: use rule_based logic unchanged
    (Fast if >= 0.7, Heavy if < 0.7).
    If fast_confidence is INSIDE [0.60, 0.80]: delegate to correctness_gate
    (V2b model) for this request, overriding rule_based.
    """
    if fast_confidence < 0.60 or fast_confidence > 0.80:
        # Outside uncertain band — rule_based decision
        if fast_confidence < RULE_BASED_CONFIDENCE_THRESHOLD:
            return RULE_BASED_ESCALATE_TO
        return "fast"
    else:
        # Inside uncertain band — let correctness gate decide
        return route_correctness_gate(image_bytes)


def route_hybrid_dumb(image_bytes: bytes, fast_confidence: float) -> str:
    """Scientific control for hybrid_gate: same trigger, dumb escalation.

    Uses the EXACT SAME confidence band check as hybrid_gate ([0.60, 0.80]).
    Outside the band: rule_based logic unchanged (identical to hybrid_gate).
    Inside the band: deterministic "heavy" -- no model call, no per-image
    intelligence. Isolates whether hybrid_gate's gain comes from genuine
    per-image routing intelligence or merely from escalating uncertain cases.
    """
    if fast_confidence < 0.60 or fast_confidence > 0.80:
        # Outside uncertain band — rule_based decision (same as hybrid_gate)
        if fast_confidence < RULE_BASED_CONFIDENCE_THRESHOLD:
            return RULE_BASED_ESCALATE_TO
        return "fast"
    else:
        # Inside uncertain band — dumb escalation (no model call)
        return "heavy"


@app.get("/health")
def health():
    return {"status": "ok", "concurrent_requests": _concurrent_requests,
            "breakers": breaker_snapshot(),
            "fault_injection": {"tier": FAULT_TIER, "rate": FAULT_RATE}}


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
        NEEDS_FAST_CONFIDENCE = {"ml_router", "rule_based", "hybrid_gate", "hybrid_dumb"}

        fast_result = None
        fast_confidence = None
        degraded_routing = None  # set when Fast is down and a cascade strategy degrades
        active_strategy = strategy  # dispatch chain uses this; may degrade below

        if strategy in NEEDS_FAST_CONFIDENCE or strategy == "always_fast":
            t0 = time.perf_counter()
            try:
                fast_result, fast_ms = await call_tier(
                    "fast", image_bytes, file.filename, file.content_type)
                timings_ms["fast_call_ms"] = fast_ms
            except TierUnavailable as exc:
                timings_ms["fast_call_ms"] = round((time.perf_counter() - t0) * 1000, 3)
                timings_ms["fast_unavailable_reason"] = str(exc)
                if strategy == "always_fast":
                    # Fast is terminal in the fallback chain -- nothing to substitute.
                    raise HTTPException(status_code=503, detail="Fast tier unavailable")
                # Degraded mode: cascade strategies need fast_confidence, which
                # does not exist when Fast is down. The correctness gate needs
                # no Fast call, so it is the natural degraded-mode router.
                # active_strategy drives the dispatch chain below; `strategy`
                # keeps the requested value for metrics/response honesty.
                degraded_routing = strategy
                active_strategy = "correctness_gate_router"
            else:
                fast_confidence = fast_result["confidence"]
                active_strategy = strategy

        if active_strategy == "ml_router":
            t0 = time.perf_counter()
            feature_row = build_feature_row(request_features, fast_confidence, load_stage)
            predicted_label = router_model.predict(feature_row)[0]
            chosen_tier = label_encoder.inverse_transform([predicted_label])[0]
            timings_ms["router_predict_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        elif active_strategy == "always_fast":
            chosen_tier = "fast"

        elif active_strategy == "always_heavy":
            chosen_tier = "heavy"

        elif active_strategy == "round_robin":
            chosen_tier = next_round_robin_tier()

        elif active_strategy == "rule_based":
            t0 = time.perf_counter()
            if fast_confidence < RULE_BASED_CONFIDENCE_THRESHOLD:
                chosen_tier = RULE_BASED_ESCALATE_TO
            else:
                chosen_tier = "fast"
            timings_ms["rule_decision_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        elif active_strategy == "single_shot_router":
            # Path A: predict tier directly from request features alone --
            # no Fast pre-call, exactly one network hop downstream (to
            # whichever tier gets picked). fast_result/fast_confidence stay
            # None here, same as always_heavy/round_robin.
            t0 = time.perf_counter()
            feature_row = build_singleshot_feature_row(request_features, load_stage)
            predicted_label = singleshot_router_model.predict(feature_row)[0]
            chosen_tier = singleshot_label_encoder.inverse_transform([predicted_label])[0]
            timings_ms["router_predict_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        elif active_strategy == "correctness_gate_router":
            # Path B single-shot: one CNN forward pass on the raw image (no
            # Fast pre-call, no hand-engineered feature row). Picks the tier
            # maximizing (P_correct_tier - LAMBDA*cost_tier).
            t0 = time.perf_counter()
            chosen_tier = route_correctness_gate(image_bytes)
            timings_ms["router_predict_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        elif active_strategy == "correctness_gate_v3_router":
            # Path B V3 single-shot: same pattern as V2b but using V3's
            # correctness_head only (confidence_head ignored per diagnostic).
            t0 = time.perf_counter()
            chosen_tier = route_correctness_gate_v3(image_bytes)
            timings_ms["router_predict_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        elif active_strategy == "correctness_gate_v3a_router":
            # Path B V3a single-shot: ablation winner (unfrozen + single head).
            # Same pattern as V3; model has no confidence_head so no unpack.
            t0 = time.perf_counter()
            chosen_tier = route_correctness_gate_v3a(image_bytes)
            timings_ms["router_predict_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        elif active_strategy == "hybrid_gate":
            # Hybrid: rule_based default, correctness gate overrides in
            # uncertain band [0.60, 0.80] around rule_based's 0.7 threshold.
            # Needs fast_confidence (already computed above via NEEDS_FAST_CONFIDENCE).
            t0 = time.perf_counter()
            chosen_tier = route_hybrid_gate(image_bytes, fast_confidence)
            timings_ms["hybrid_decision_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        elif active_strategy == "hybrid_dumb":
            # Scientific control: same trigger as hybrid_gate (band [0.60, 0.80]),
            # but deterministic "heavy" escalation instead of correctness gate.
            t0 = time.perf_counter()
            chosen_tier = route_hybrid_dumb(image_bytes, fast_confidence)
            timings_ms["hybrid_decision_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        ROUTING_DECISIONS.labels(tier=chosen_tier, strategy=strategy).inc()

        # Dispatch with fallback substitution: try the chosen tier, then walk
        # FALLBACK_ORDER (heavy -> balanced -> fast) on TierUnavailable. The
        # fast_result reuse only applies when fast was actually reached above.
        fallback_from = None
        if chosen_tier == "fast" and fast_result is not None:
            # Already called Fast for confidence above; reuse that result.
            final_result = fast_result
            timings_ms["escalated_call_ms"] = 0.0
        else:
            final_result = None
            last_reason = ""
            for attempt_tier in [chosen_tier] + fallback_chain(chosen_tier):
                t0 = time.perf_counter()
                try:
                    final_result, call_ms = await call_tier(
                        attempt_tier, image_bytes, file.filename, file.content_type)
                except TierUnavailable as exc:
                    last_reason = str(exc)
                    timings_ms[f"dispatch_{attempt_tier}_failed"] = str(exc)
                    continue
                timings_ms["escalated_call_ms"] = call_ms
                if attempt_tier != chosen_tier:
                    fallback_from = chosen_tier
                    chosen_tier = attempt_tier
                    FALLBACKS.labels(from_tier=fallback_from, to_tier=attempt_tier,
                                     strategy=strategy).inc()
                break
            if final_result is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"{chosen_tier} tier unavailable and no fallback left "
                           f"(last: {last_reason})")

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
            "fallback_from": fallback_from,
            "degraded_routing": degraded_routing,
        }

    finally:
        _bump_concurrency(-1)