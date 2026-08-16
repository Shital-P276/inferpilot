"""
benchmark_correctness_gate.py

Benchmarks CorrectnessGateNet inference latency in isolation, using the EXACT
methodology of benchmark_models.py (the script that produced
router/model_benchmarks.json): NUM_WARMUP=10 discarded warm-up runs,
NUM_RUNS=100 timed runs, INPUT_SHAPE (1,3,224,224), CUDA with
torch.cuda.synchronize() per run, sorted-latency avg/p50/p95 stats.

Unlike the tier benchmark (which times a random dummy tensor), this uses REAL
test-set images (preprocessed ONCE outside the timed region, exactly like the
tier benchmark builds its dummy tensor before timing) in two modes:

  a) single_image : ONE fixed test image, run 100x -- isolates raw per-call
                    inference cost (no preprocessing in the timed path).
  b) random_image : 100 DIFFERENT test images, one inference each -- checks
                    whether latency is stable across varied real inputs.

Output:
  training/checkpoints/correctness_gate_latency_benchmark.json
  (list of dicts with the same schema as router/model_benchmarks.json entries,
  plus "std_latency_ms" and a "mode" discriminator)

USAGE:
    python benchmark_correctness_gate.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from PIL import Image
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

import psutil
import os

# ---- CONFIG (mirrors benchmark_models.py) ----
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "training" / "checkpoints" / "correctness_gate_best.pt"
OUTPUT_PATH = BASE_DIR / "training" / "checkpoints" / "correctness_gate_latency_benchmark.json"
TIER_BENCHMARKS_PATH = BASE_DIR / "router" / "model_benchmarks.json"
LABELS_PATH = BASE_DIR / "router" / "utility_labels.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WARMUP = 10       # runs discarded before timing starts (same as benchmark_models.py)
NUM_RUNS = 100        # runs actually measured (same as benchmark_models.py)
INPUT_SHAPE = (1, 3, 224, 224)

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Same 70/15/15 stratified split as training/train_correctness_gate.py
SEED = 42
# ------------------------------------------------------------------------


eval_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(IMG_SIZE),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def get_gpu_memory_mb():
    """Current GPU memory actually allocated by tensors, in MB. 0 if no CUDA."""
    if DEVICE.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated() / (1024 ** 2)


def load_model():
    from training.models.correctness_gate import CorrectnessGateNet
    model = CorrectnessGateNet(num_tiers=3)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def load_test_images():
    """Reproduce the gate training script's test split and preprocess its
    images to (1,3,224,224) tensors. Returns a list of tensors."""
    df = pd.read_csv(LABELS_PATH)
    _, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["best_tier"], random_state=SEED
    )
    _, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["best_tier"], random_state=SEED
    )
    print(f"Reproduced test split: {len(test_df)} test images")

    tensors = []
    for _, row in test_df.iterrows():
        img_path = BASE_DIR / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        t = eval_transform(image).unsqueeze(0)  # [1,3,224,224]
        if DEVICE.type == "cuda":
            t = t.cuda(non_blocking=True)
        tensors.append(t)
    return tensors


def time_inference(model, input_tensors):
    """
    Times NUM_RUNS forward passes. input_tensors is a list of preprocessed
    [1,3,224,224] tensors; run i uses input_tensors[i % len(input_tensors)].
    - single_image mode: list of length 1 -> same tensor all 100 runs
    - random_image mode: list of length NUM_RUNS -> one distinct tensor each
    """
    warmup_tensor = input_tensors[0]
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            _ = model(warmup_tensor)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    mem_before = get_process_memory_mb()
    latencies = []
    with torch.no_grad():
        for i in range(NUM_RUNS):
            x = input_tensors[i % len(input_tensors)]
            start = time.perf_counter()
            _ = model(x)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
    mem_after = get_process_memory_mb()

    gpu_mem_peak_inference_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2) if DEVICE.type == "cuda" else 0.0
    )

    latencies.sort()
    avg = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    std = float(np.std(latencies))

    return {
        "avg_latency_ms": round(avg, 3),
        "p50_latency_ms": round(p50, 3),
        "p95_latency_ms": round(p95, 3),
        "std_latency_ms": round(std, 3),
        "cpu_memory_delta_mb": round(mem_after - mem_before, 2),
        "gpu_peak_inference_mb": round(gpu_mem_peak_inference_mb, 2),
    }, latencies


def main():
    print(f"Device: {DEVICE}")
    if not CHECKPOINT_PATH.exists():
        print(f"FATAL: checkpoint not found at {CHECKPOINT_PATH}")
        return

    model = load_model()
    gpu_mem_model_mb = get_gpu_memory_mb()
    checkpoint_size_mb = CHECKPOINT_PATH.stat().st_size / (1024 ** 2)
    num_params = sum(p.numel() for p in model.parameters())

    test_tensors = load_test_images()
    if len(test_tensors) < NUM_RUNS:
        print(f"WARNING: test set has only {len(test_tensors)} images (< NUM_RUNS={NUM_RUNS}); "
              f"random_image mode will reuse images.")

    # ---- (a) Single-image latency: one fixed image, 100x ----
    print(f"\n=== CorrectnessGateNet: single-image latency (100x same image) ===")
    single_stats, single_lats = time_inference(model, [test_tensors[0]])
    print(json.dumps(single_stats, indent=2))

    # ---- (b) Random-image latency: 100 distinct images, one inference each ----
    print(f"\n=== CorrectnessGateNet: random-image latency (100 distinct images, 1x each) ===")
    random_tensors = test_tensors[:NUM_RUNS]
    if len(random_tensors) < NUM_RUNS:
        while len(random_tensors) < NUM_RUNS:
            random_tensors.append(test_tensors[len(random_tensors) % len(test_tensors)])
    random_stats, random_lats = time_inference(model, random_tensors)
    print(json.dumps(random_stats, indent=2))

    # ---- Summary table ----
    print(f"\n{'='*70}")
    print("SUMMARY: CorrectnessGateNet inference latency vs tier benchmarks")
    print(f"{'='*70}")
    print(f"{'model':<34} {'avg_ms':>8} {'p50_ms':>8} {'p95_ms':>8} {'std_ms':>8}")
    print(f"{'-'*70}")
    print(f"{'CorrectnessGateNet (single-image)':<34} {single_stats['avg_latency_ms']:>8.3f} "
          f"{single_stats['p50_latency_ms']:>8.3f} {single_stats['p95_latency_ms']:>8.3f} "
          f"{single_stats['std_latency_ms']:>8.3f}")
    print(f"{'CorrectnessGateNet (random-image)':<34} {random_stats['avg_latency_ms']:>8.3f} "
          f"{random_stats['p50_latency_ms']:>8.3f} {random_stats['p95_latency_ms']:>8.3f} "
          f"{random_stats['std_latency_ms']:>8.3f}")

    if TIER_BENCHMARKS_PATH.exists():
        with open(TIER_BENCHMARKS_PATH) as f:
            tier_bench = json.load(f)
        for entry in tier_bench:
            tier = entry.get("tier", "?")
            print(f"{f'Tier: {tier}':<34} {entry.get('avg_latency_ms', 0):>8.3f} "
                  f"{entry.get('p50_latency_ms', 0):>8.3f} {entry.get('p95_latency_ms', 0):>8.3f} "
                  f"{'':>8}")

    print(f"{'-'*70}")
    print("Cascade router checking-cost overhead: ~50-100ms "
          "(REPORTED, not measured here -- source: router/train_router_singleshot.py "
          "'~50-100ms network tax' for the Fast pre-call in the cascade path)")

    # ---- Save results ----
    output = [
        {
            "model": "correctness_gate",
            "mode": "single_image",
            "avg_latency_ms": single_stats["avg_latency_ms"],
            "p50_latency_ms": single_stats["p50_latency_ms"],
            "p95_latency_ms": single_stats["p95_latency_ms"],
            "std_latency_ms": single_stats["std_latency_ms"],
            "cpu_memory_delta_mb": single_stats["cpu_memory_delta_mb"],
            "gpu_model_footprint_mb": round(gpu_mem_model_mb, 2),
            "gpu_peak_inference_mb": single_stats["gpu_peak_inference_mb"],
            "checkpoint_size_mb": round(checkpoint_size_mb, 2),
            "num_params": num_params,
            "device": str(DEVICE),
        },
        {
            "model": "correctness_gate",
            "mode": "random_image",
            "avg_latency_ms": random_stats["avg_latency_ms"],
            "p50_latency_ms": random_stats["p50_latency_ms"],
            "p95_latency_ms": random_stats["p95_latency_ms"],
            "std_latency_ms": random_stats["std_latency_ms"],
            "cpu_memory_delta_mb": random_stats["cpu_memory_delta_mb"],
            "gpu_model_footprint_mb": round(gpu_mem_model_mb, 2),
            "gpu_peak_inference_mb": random_stats["gpu_peak_inference_mb"],
            "checkpoint_size_mb": round(checkpoint_size_mb, 2),
            "num_params": num_params,
            "device": str(DEVICE),
        },
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()