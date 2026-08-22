"""
Benchmark CorrectnessGateNetV2 inference latency (same methodology as
benchmark_correctness_gate.py): 10 warmup runs discarded, 100 timed runs,
CUDA with torch.cuda.synchronize(), single fixed test image, sorted-latency
avg/p50/p95 stats.

HARD GATE: if avg latency >= 2.0ms, the model is too slow for serving as a
gate (Balanced tier is 4.842ms, gate must be well under that).

Usage:
    python benchmark_correctness_gate_v2.py
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

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "training" / "checkpoints" / "correctness_gate_v2_best.pt"
OUTPUT_PATH = BASE_DIR / "training" / "checkpoints" / "correctness_gate_v2_latency_benchmark.json"
LABELS_PATH = BASE_DIR / "router" / "utility_labels.csv"
TIER_BENCHMARKS_PATH = BASE_DIR / "router" / "model_benchmarks.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WARMUP = 10
NUM_RUNS = 100
IMG_SIZE = 224
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
eval_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(IMG_SIZE),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

LATENCY_GATE_MS = 2.0


def load_model():
    from training.models.correctness_gate import CorrectnessGateNetV2
    model = CorrectnessGateNetV2(num_tiers=3)
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded V2 checkpoint from {CHECKPOINT_PATH}")
    else:
        print(f"No V2 checkpoint found at {CHECKPOINT_PATH}; benchmarking untrained V2 model")
    model.to(DEVICE)
    model.eval()
    return model


def load_one_test_image():
    df = pd.read_csv(LABELS_PATH)
    df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)
    _, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["best_tier"], random_state=SEED
    )
    _, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["best_tier"], random_state=SEED
    )
    img_path = BASE_DIR / test_df.iloc[0]["image_path"]
    image = Image.open(img_path).convert("RGB")
    t = eval_transform(image).unsqueeze(0)
    if DEVICE.type == "cuda":
        t = t.cuda(non_blocking=True)
    return t


def benchmark(model, input_tensor):
    # Warmup
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            _ = model(input_tensor)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(NUM_RUNS):
            start = time.perf_counter()
            _ = model(input_tensor)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    gpu_peak = torch.cuda.max_memory_allocated() / (1024 ** 2) if DEVICE.type == "cuda" else 0.0
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
        "gpu_peak_inference_mb": round(gpu_peak, 2),
    }


def main():
    print(f"Device: {DEVICE}")
    model = load_model()
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"V2 total params: {n_params:,}  trainable: {n_trainable:,}")

    input_tensor = load_one_test_image()

    stats = benchmark(model, input_tensor)

    print(f"\n=== CorrectnessGateNetV2 latency benchmark ({NUM_RUNS} runs) ===")
    print(f"  avg:  {stats['avg_latency_ms']:.3f} ms")
    print(f"  p50:  {stats['p50_latency_ms']:.3f} ms")
    print(f"  p95:  {stats['p95_latency_ms']:.3f} ms")
    print(f"  std:  {stats['std_latency_ms']:.3f} ms")
    print(f"  GPU peak: {stats['gpu_peak_inference_mb']:.2f} MB")

    # Hard gate
    passed = stats["avg_latency_ms"] < LATENCY_GATE_MS
    print(f"\n{'='*50}")
    print(f"HARD GATE: avg latency {stats['avg_latency_ms']:.3f} ms "
          f"{'<' if passed else '>='} {LATENCY_GATE_MS} ms -> "
          f"{'PASS - proceed to training' if passed else 'FAIL - STOP, too slow for serving'}")
    print(f"{'='*50}")

    if not passed:
        print("\nModel exceeds latency gate. Do NOT proceed to Step 4.")
        # Still save the benchmark for record
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump({"result": "FAIL", **stats}, f, indent=2)
        return stats, False

    # Compare with tier benchmarks
    print(f"\nLatency comparison:")
    print(f"  CorrectnessGateNetV2: {stats['avg_latency_ms']:.3f} ms")
    if TIER_BENCHMARKS_PATH.exists():
        with open(TIER_BENCHMARKS_PATH) as f:
            for entry in json.load(f):
                print(f"  Tier {entry['tier']:9s}: {entry['avg_latency_ms']:.3f} ms")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"result": "PASS", **stats}, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")
    return stats, True


if __name__ == "__main__":
    main()
