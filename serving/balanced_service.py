"""
Balanced tier serving endpoint.

Same pattern as serving/fast_service.py, wrapping BalancedMobileNet instead.
Exposes Prometheus metrics at /metrics.

Run directly:
    uvicorn serving.balanced_service:app --host 0.0.0.0 --port 8002 --reload

Test:
    http://localhost:8002/docs
    curl http://localhost:8002/metrics
"""

import io
import time

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision.transforms import v2
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter

from training.models.balanced_mobilenet import BalancedMobileNet

# ---- Config ----
CHECKPOINT_PATH = "training/checkpoints/balanced_mobilenet_best.pt"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIER_NAME = "balanced"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = None  # populated at startup from the checkpoint
# ----------------

eval_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(IMG_SIZE),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

app = FastAPI(title="InferPilot - Balanced Tier Service")

# ---- Prometheus: auto-instrumentation ----
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ---- Prometheus: custom metrics ----
inference_latency = Histogram(
    "inference_latency_ms", "Model inference latency in milliseconds (excludes HTTP overhead)",
    ["tier"], buckets=[1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
)
prediction_confidence = Histogram(
    "prediction_confidence", "Model prediction confidence (softmax max)",
    ["tier"], buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)
predictions_total = Counter(
    "predictions_total", "Total predictions made, by tier and predicted class",
    ["tier", "predicted_class"]
)

model = None


@app.on_event("startup")
def load_model():
    global model, CLASS_NAMES
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    CLASS_NAMES = checkpoint["class_names"]
    val_acc = checkpoint.get("val_acc")
    epoch = checkpoint.get("epoch")

    model = BalancedMobileNet(num_classes=len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"[balanced_service] Loaded checkpoint from {CHECKPOINT_PATH} on {DEVICE}")
    print(f"[balanced_service] Checkpoint epoch={epoch}, val_acc={val_acc}")
    print(f"[balanced_service] Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")


@app.get("/health")
def health():
    return {"status": "ok", "tier": TIER_NAME, "device": str(DEVICE)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        raw_bytes = await file.read()
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    input_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
    latency_ms = (time.perf_counter() - start) * 1000

    predicted_class = CLASS_NAMES[pred_idx.item()]
    confidence_value = confidence.item()

    inference_latency.labels(tier=TIER_NAME).observe(latency_ms)
    prediction_confidence.labels(tier=TIER_NAME).observe(confidence_value)
    predictions_total.labels(tier=TIER_NAME, predicted_class=predicted_class).inc()

    return {
        "tier": TIER_NAME,
        "predicted_class": predicted_class,
        "confidence": round(confidence_value, 4),
        "latency_ms": round(latency_ms, 3),
    }