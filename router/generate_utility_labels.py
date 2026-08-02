"""
Generates the router's training dataset, with a λ/μ sensitivity sweep
to pick a defensible utility-formula weighting before locking in final labels.

Also carves out a small HELD-OUT set (same domain, properly labeled, excluded
from all label generation) for later genuine end-to-end sanity checks --
avoids needing untrustworthy/unlabeled external images (e.g. random Google
images) for any future spot-checks.

Output:
  router/utility_sweep_summary.csv   -- label distribution per (lambda, mu) combo
  router/utility_labels.csv          -- final labels using CHOSEN_LAMBDA/CHOSEN_MU
  router/held_out_images.csv         -- image_path,true_label for later checks
"""
import json
import random
import csv
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from models.fast_cnn import FastCNN
from models.balanced_mobilenet import BalancedMobileNet
from models.heavy_efficientnet import HeavyEfficientNet

# ---------------- Config ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "test"
CHECKPOINT_DIR = BASE_DIR / "training" / "checkpoints"
BENCHMARKS_PATH = BASE_DIR / "router" / "model_benchmarks.json"
COMBINED_RESULTS_PATH = BASE_DIR / "monitoring" / "results" / "combined_results.json"

SWEEP_OUTPUT_PATH = BASE_DIR / "router" / "utility_sweep_summary.csv"
FINAL_OUTPUT_PATH = BASE_DIR / "router" / "utility_labels.csv"
HELD_OUT_OUTPUT_PATH = BASE_DIR / "router" / "held_out_images.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_SIZE = 4000          # target working-set size (label generation + sweep)
HOLD_OUT_SIZE = 250         # carved out BEFORE sampling, never touched by label gen
SEED = 42

LAMBDA_GRID = [0.3, 0.7, 1.2, 2.0]
MU_GRID = [0.1, 0.3, 0.5]

CHOSEN_LAMBDA = 0.7
CHOSEN_MU = 0.3

LATENCY_NORM_PERCENTILE = 95

LOAD_STAGE_WEIGHTS = {"light": 0.5, "heavy": 0.35, "burst": 0.15}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

eval_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(IMG_SIZE),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

random.seed(SEED)
np.random.seed(SEED)


def load_models():
    models = {}

    fast = FastCNN(num_classes=28).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_DIR / "fast_cnn_best.pt", map_location=DEVICE)
    fast.load_state_dict(ckpt["model_state_dict"])
    fast.eval()
    models["fast"] = (fast, ckpt["class_names"])

    balanced = BalancedMobileNet(num_classes=28, freeze_backbone=False).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_DIR / "balanced_mobilenet_best.pt", map_location=DEVICE)
    balanced.load_state_dict(ckpt["model_state_dict"])
    balanced.eval()
    models["balanced"] = (balanced, ckpt["class_names"])

    heavy = HeavyEfficientNet(num_classes=28, freeze_backbone=False).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_DIR / "heavy_efficientnet_best.pt", map_location=DEVICE)
    heavy.load_state_dict(ckpt["model_state_dict"])
    heavy.eval()
    models["heavy"] = (heavy, ckpt["class_names"])

    return models


def load_benchmarks():
    with open(BENCHMARKS_PATH) as f:
        data = json.load(f)
    benchmarks = {entry["tier"]: entry for entry in data}

    max_gpu_mem = max(e["gpu_peak_inference_mb"] for e in data)
    max_params = max(e["num_params"] for e in data)

    resource_norm = {}
    for tier, e in benchmarks.items():
        resource_norm[tier] = (
            0.7 * (e["gpu_peak_inference_mb"] / max_gpu_mem)
            + 0.3 * (e["num_params"] / max_params)
        )
    return resource_norm


def load_latency_distributions():
    distributions = {}
    with open(COMBINED_RESULTS_PATH) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "Point" or d.get("metric") != "http_req_duration":
                continue
            tags = d["data"].get("tags", {}) or {}
            tier = tags.get("tier")
            stage = tags.get("load_stage")
            if tier is None or stage is None:
                continue
            distributions.setdefault((tier, stage), []).append(d["data"]["value"])

    for key, vals in distributions.items():
        print(f"  loaded {len(vals):5d} latency samples for {key}")
    return distributions


def sample_latency(distributions, tier, stage):
    vals = distributions.get((tier, stage))
    if not vals:
        raise ValueError(f"No latency data for {(tier, stage)}")
    return random.choice(vals)


def extract_features(image_path, pil_image):
    img_cv = cv2.imread(str(image_path))
    if img_cv is None:
        img_cv = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    width, height = pil_image.size
    file_size_kb = image_path.stat().st_size / 1024
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    edges = cv2.Canny(gray, 100, 200)
    edge_density = (edges > 0).sum() / edges.size

    return {
        "width": width, "height": height,
        "file_size_kb": round(file_size_kb, 2),
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "edge_density": round(edge_density, 4),
    }


@torch.no_grad()
def predict(model, image_tensor, class_names, true_label):
    logits = model(image_tensor.unsqueeze(0).to(DEVICE))
    probs = F.softmax(logits, dim=1)
    pred_idx = probs.argmax(dim=1).item()
    confidence = probs[0, pred_idx].item()
    predicted_class = class_names[pred_idx]
    correct = int(predicted_class == true_label)
    return correct, confidence


def sample_all_image_paths(data_dir):
    """Returns full list of (path, class_name) for every image in data_dir."""
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    all_images = []
    for class_dir in class_dirs:
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        all_images.extend([(p, class_dir.name) for p in images])
    random.shuffle(all_images)
    return all_images


def split_holdout_and_working(all_images, hold_out_size, working_size):
    """
    Carves out a held-out set first (untouched by label generation), then
    builds the working set from the remainder. Small classes contribute
    everything they have; any leftover budget is redistributed across
    larger classes so the working set actually reaches `working_size`
    (subject to total availability).
    """
    hold_out = all_images[:hold_out_size]
    remainder = all_images[hold_out_size:]

    by_class = {}
    for path, label in remainder:
        by_class.setdefault(label, []).append(path)
    for label in by_class:
        random.shuffle(by_class[label])

    n_classes = len(by_class)
    base_per_class = working_size // n_classes

    working = []
    remaining_classes = dict(by_class)  # classes still with leftover images
    budget_left = working_size

    # Pass 1: take up to base_per_class from every class
    leftover_pool = []
    for label, paths in list(remaining_classes.items()):
        take = min(base_per_class, len(paths))
        working.extend([(p, label) for p in paths[:take]])
        leftover_pool.extend([(p, label) for p in paths[take:]])  # unused images, still available
        budget_left -= take

    # Pass 2: fill remaining budget from the leftover pool (larger classes' extra images)
    random.shuffle(leftover_pool)
    working.extend(leftover_pool[:budget_left])

    random.shuffle(working)
    return hold_out, working[:working_size]


def main():
    print(f"Using device: {DEVICE}")

    print("\nLoading models...")
    models = load_models()

    print("\nLoading resource benchmarks...")
    resource_norm = load_benchmarks()
    print(f"  resource_norm: {resource_norm}")

    print("\nLoading k6 latency distributions...")
    latency_dist = load_latency_distributions()

    all_latencies = [v for vals in latency_dist.values() for v in vals]
    max_latency = float(np.percentile(all_latencies, LATENCY_NORM_PERCENTILE))
    true_max = max(all_latencies)
    print(f"  true max latency: {true_max:.2f} ms")
    print(f"  p{LATENCY_NORM_PERCENTILE} latency (used for normalization): {max_latency:.2f} ms")

    print(f"\nGathering all test images and carving out held-out set...")
    all_images = sample_all_image_paths(DATA_DIR)
    hold_out, samples = split_holdout_and_working(all_images, HOLD_OUT_SIZE, SAMPLE_SIZE)
    print(f"  total test images: {len(all_images)}")
    print(f"  held out (never used for labels): {len(hold_out)}")
    print(f"  working set (label generation + sweep): {len(samples)}")

    # Write held-out set immediately -- untouched by anything below
    with open(HELD_OUT_OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label"])
        for path, label in hold_out:
            writer.writerow([str(path.relative_to(BASE_DIR)), label])
    print(f"  held-out set written to {HELD_OUT_OUTPUT_PATH}")

    tiers = ["fast", "balanced", "heavy"]

    # ---- Pass 1: run inference + feature extraction ONCE per image, cache results ----
    print("\nRunning inference (once per image, cached for sweep)...")
    cached_rows = []
    for image_path, true_label in tqdm(samples, desc="Inference"):
        pil_image = Image.open(image_path).convert("RGB")
        image_tensor = eval_transform(pil_image)
        features = extract_features(image_path, pil_image)

        stage = random.choices(
            list(LOAD_STAGE_WEIGHTS.keys()),
            weights=list(LOAD_STAGE_WEIGHTS.values()),
        )[0]

        per_tier = {}
        for tier in tiers:
            model, class_names = models[tier]
            correct, confidence = predict(model, image_tensor, class_names, true_label)
            latency_ms = sample_latency(latency_dist, tier, stage)
            latency_norm = min(latency_ms / max_latency, 1.0)
            per_tier[tier] = {
                "correct": correct,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "latency_norm": latency_norm,
                "resource_norm": resource_norm[tier],
            }

        cached_rows.append({
            "image_path": str(image_path.relative_to(BASE_DIR)),
            "true_label": true_label,
            "load_stage": stage,
            "features": features,
            "per_tier": per_tier,
        })

    # ---- Pass 2: sweep λ/μ over cached results ----
    print("\nRunning λ/μ sensitivity sweep...")
    sweep_results = []
    for lam in LAMBDA_GRID:
        for mu in MU_GRID:
            label_counts = Counter()
            for row in cached_rows:
                utilities = {}
                for tier in tiers:
                    t = row["per_tier"][tier]
                    utility = t["correct"] - lam * t["latency_norm"] - mu * t["resource_norm"]
                    utilities[tier] = utility
                best_tier = max(utilities, key=utilities.get)
                label_counts[best_tier] += 1

            total = len(cached_rows)
            sweep_results.append({
                "lambda": lam, "mu": mu,
                "fast_pct": round(100 * label_counts.get("fast", 0) / total, 1),
                "balanced_pct": round(100 * label_counts.get("balanced", 0) / total, 1),
                "heavy_pct": round(100 * label_counts.get("heavy", 0) / total, 1),
            })
            print(f"  λ={lam:.1f} μ={mu:.2f} -> "
                  f"fast={sweep_results[-1]['fast_pct']:5.1f}%  "
                  f"balanced={sweep_results[-1]['balanced_pct']:5.1f}%  "
                  f"heavy={sweep_results[-1]['heavy_pct']:5.1f}%")

    with open(SWEEP_OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sweep_results[0].keys())
        writer.writeheader()
        writer.writerows(sweep_results)
    print(f"\nSweep summary written to {SWEEP_OUTPUT_PATH}")

    # ---- Pass 3: generate FINAL labels using CHOSEN_LAMBDA / CHOSEN_MU ----
    print(f"\nGenerating final labels using λ={CHOSEN_LAMBDA}, μ={CHOSEN_MU}...")
    final_rows = []
    for row in cached_rows:
        out_row = {
            "image_path": row["image_path"],
            "true_label": row["true_label"],
            "load_stage": row["load_stage"],
            **row["features"],
        }
        utilities = {}
        for tier in tiers:
            t = row["per_tier"][tier]
            utility = t["correct"] - CHOSEN_LAMBDA * t["latency_norm"] - CHOSEN_MU * t["resource_norm"]
            out_row[f"{tier}_correct"] = t["correct"]
            out_row[f"{tier}_confidence"] = round(t["confidence"], 4)
            out_row[f"{tier}_latency_ms"] = round(t["latency_ms"], 2)
            out_row[f"{tier}_utility"] = round(utility, 4)
            utilities[tier] = utility

        out_row["best_tier"] = max(utilities, key=utilities.get)
        final_rows.append(out_row)

    with open(FINAL_OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_rows[0].keys())
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"Done. Wrote {len(final_rows)} labeled samples to {FINAL_OUTPUT_PATH}")

    label_counts = Counter(r["best_tier"] for r in final_rows)
    print("\nFinal label distribution:")
    for tier, count in label_counts.items():
        print(f"  {tier:10s}: {count:5d} ({100*count/len(final_rows):.1f}%)")


if __name__ == "__main__":
    main()