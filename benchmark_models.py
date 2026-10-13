"""
benchmark_models.py

Benchmarks latency and resource usage for the three InferPilot model tiers
(Fast / Balanced / Heavy). Run this AFTER training all three checkpoints.

BEFORE RUNNING:
  1. Install psutil if you don't have it:
         pip install psutil
  2. Edit the `load_model()` function below so each tier is constructed
     exactly the way your training scripts built it (same architecture,
     same num_classes). The imports/class names here are placeholders --
     swap them for your actual model definitions.
  3. Update CHECKPOINTS paths below if yours differ.
  4. Update YOUR_NUM_CLASSES to match your dataset.

USAGE:
    python benchmark_models.py

OUTPUT:
    Prints per-model latency/memory/size stats, and saves a combined
    router/model_benchmarks.json -- this file feeds directly into the
    router training step later, so don't lose it.
"""

import time
import os
import json
from pathlib import Path

import torch
import psutil

# ---- CONFIG ----------------------------------------------------------
YOUR_NUM_CLASSES = 182  # <-- change to your actual number of classes

CHECKPOINTS = {
    "balanced": "training/checkpoints/iwildcam_balanced_resnet18_best.pt",
    "heavy": "training/checkpoints/iwildcam_heavy_convnextbase_best.pt",
    "fast": "training/checkpoints/iwildcam_fast_mobilenetv3small_best.pt",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WARMUP = 10       # runs discarded before timing starts (JIT/cache warm-up)
NUM_RUNS = 100        # runs actually measured
INPUT_SHAPES = {
    "fast": (1, 3, 224, 224),
    "balanced": (1, 3, 256, 256),
    "heavy": (1, 3, 384, 384),
}
# ------------------------------------------------------------------------


def load_model(tier_name, checkpoint_path, device_override=None):
    """
    Builds the exact architecture used in training, per tier.
    """
    effective_device = device_override if device_override is not None else DEVICE
    # "fast_cpu" is a synthetic tier for CPU comparison — map to fast architecture
    effective_tier = "fast" if tier_name == "fast_cpu" else tier_name
    if effective_tier == "fast":
        from training.models.iwildcam_fast_mobilenetv3small import build_fast_mobilenetv3small

        model = build_fast_mobilenetv3small(num_classes=YOUR_NUM_CLASSES)
    elif effective_tier == "balanced":
        from training.models.iwildcam_balanced_resnet18 import build_balanced_resnet18

        model = build_balanced_resnet18(num_classes=YOUR_NUM_CLASSES)
    elif effective_tier == "heavy":
        from training.models.iwildcam_heavy_convnextbase import build_heavy_convnextbase

        model = build_heavy_convnextbase(num_classes=YOUR_NUM_CLASSES)

    else:
        raise ValueError(f"Unknown tier: {tier_name}")

    checkpoint = torch.load(checkpoint_path, map_location=effective_device)
    # Handle both "raw state_dict" and "dict wrapping state_dict" checkpoint styles
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(effective_device)
    model.eval()
    return model


def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def get_gpu_memory_mb(device_override=None):
    """Current GPU memory actually allocated by tensors, in MB. 0 if no CUDA."""
    dev = device_override if device_override is not None else DEVICE
    if dev.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated() / (1024 ** 2)


def benchmark_model(tier_name, checkpoint_path, device_override=None):
    print(f"\n=== Benchmarking {tier_name} ===")

    effective_device = device_override if device_override is not None else DEVICE
    effective_tier = "fast" if tier_name == "fast_cpu" else tier_name

    if effective_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = load_model(tier_name, checkpoint_path, device_override=device_override)

    # GPU memory taken up by the model weights alone (after load, before any inference)
    gpu_mem_model_mb = get_gpu_memory_mb(device_override=effective_device)

    dummy_input = torch.randn(INPUT_SHAPES[effective_tier]).to(effective_device)

    # Warm-up (not counted -- first few calls are always artificially slow)
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            _ = model(dummy_input)
    if effective_device.type == "cuda":
        torch.cuda.synchronize()
        # Reset peak tracker after warm-up so it only reflects the timed runs below
        torch.cuda.reset_peak_memory_stats()

    mem_before = get_process_memory_mb()
    latencies = []

    with torch.no_grad():
        for _ in range(NUM_RUNS):
            start = time.perf_counter()
            _ = model(dummy_input)
            if effective_device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # convert to ms

    mem_after = get_process_memory_mb()

    # Peak GPU memory actually touched during the timed inference runs
    # (activations + model weights + input tensor -- the real footprint at serve time)
    if effective_device.type == "cuda":
        gpu_mem_peak_inference_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        gpu_mem_peak_inference_mb = 0.0

    latencies.sort()
    avg_latency = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    file_size_mb = Path(checkpoint_path).stat().st_size / (1024 ** 2)
    num_params = sum(p.numel() for p in model.parameters())

    results = {
        "tier": tier_name,
        "avg_latency_ms": round(avg_latency, 3),
        "p50_latency_ms": round(p50, 3),
        "p95_latency_ms": round(p95, 3),
        "cpu_memory_delta_mb": round(mem_after - mem_before, 2),
        "gpu_model_footprint_mb": round(gpu_mem_model_mb, 2),
        "gpu_peak_inference_mb": round(gpu_mem_peak_inference_mb, 2),
        "checkpoint_size_mb": round(file_size_mb, 2),
        "num_params": num_params,
        "device": str(effective_device),
    }

    print(json.dumps(results, indent=2))

    # Free this model's GPU memory before the next tier loads
    del model
    if effective_device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def main():
    # One-time throwaway warm-up: primes CUDA/cuDNN kernel selection so the
    # first real-measured tier doesn't unfairly eat that one-time cost.
    try:
        from training.models.iwildcam_fast_mobilenetv3small import build_fast_mobilenetv3small

        _warmup_model = build_fast_mobilenetv3small(num_classes=YOUR_NUM_CLASSES)
        _warmup_model.to(DEVICE)
        _warmup_model.eval()
        _warmup_dummy = torch.randn((1, 3, 224, 224)).to(DEVICE)
        with torch.no_grad():
            for _ in range(5):
                _ = _warmup_model(_warmup_dummy)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        del _warmup_model, _warmup_dummy
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

    all_results = []
    for tier_name, checkpoint_path in CHECKPOINTS.items():
        if not Path(checkpoint_path).exists():
            print(f"WARNING: checkpoint not found for '{tier_name}' at {checkpoint_path}, skipping")
            continue
        result = benchmark_model(tier_name, checkpoint_path)
        all_results.append(result)

    output_path = Path("router/model_benchmarks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved combined benchmark results to {output_path}")
    print("This file is the input your router training step will use next.")

    print("\nWARNING: the following files contain hardcoded copies of the "
          "OLD benchmark numbers and will not update automatically:")
    print("  - router/compute_strategy_utility.py (TIER_GPU_MEMORY_MB dict)")
    print("  - serving/gateway_service.py (CORRECTNESS_GATE_COSTS dict)")
    print("Update both manually using the new values just written to "
          "router/model_benchmarks.json before trusting any downstream "
          "cost calculations.")

    # CPU comparison benchmark for fast tier — reuse same methodology but force CPU
    try:
        cpu_result = benchmark_model("fast_cpu", CHECKPOINTS["fast"], device_override=torch.device("cpu"))
        gpu_result = next((r for r in all_results if r["tier"] == "fast"), None)
        if gpu_result is not None:
            print(f"\nFast tier: GPU avg={gpu_result['avg_latency_ms']}ms vs CPU avg={cpu_result['avg_latency_ms']}ms")
        cpu_output_path = Path("router/fast_cpu_comparison.json")
        cpu_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cpu_output_path, "w") as f:
            json.dump(cpu_result, f, indent=2)
        print(f"Saved CPU comparison result to {cpu_output_path}")
    except Exception as e:
        print(f"WARNING: CPU comparison benchmark failed: {e}")


if __name__ == "__main__":
    main()
