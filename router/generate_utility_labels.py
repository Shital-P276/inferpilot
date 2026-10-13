"""
Generates the router's training dataset, with a λ/μ sensitivity sweep
to pick a defensible utility-formula weighting before locking in final labels.

Also regenerates held_out_images.csv from the already-separate final_holdout
manifest (disjointness already established separately, not derived here).

Output:
  router/utility_sweep_summary.csv   -- label distribution per (lambda, mu) combo
  router/utility_labels.csv          -- final labels using CHOSEN_LAMBDA/CHOSEN_MU
  router/held_out_images.csv         -- image_path,true_label for later checks
"""
import argparse
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
from models.iwildcam_fast_mobilenetv3small import build_fast_mobilenetv3small
from models.iwildcam_balanced_resnet18 import build_balanced_resnet18
from models.iwildcam_heavy_convnextbase import build_heavy_convnextbase

# ---------------- Config ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
# New manifest-based source (no class-subfolder traversal)
UTILITY_LABEL_GEN_DIR = BASE_DIR / "data" / "iwildcam_utility_label_subset"
MANIFEST_PATH = UTILITY_LABEL_GEN_DIR / "manifest.csv"
# Final holdout already separate (do NOT derive here)
FINAL_HOLDOUT_DIR = BASE_DIR / "data" / "iwildcam_final_holdout_subset"
FINAL_HOLDOUT_MANIFEST = FINAL_HOLDOUT_DIR / "manifest.csv"

CHECKPOINT_DIR = BASE_DIR / "training" / "checkpoints"
BENCHMARKS_PATH = BASE_DIR / "router" / "model_benchmarks.json"
COMBINED_RESULTS_PATH = BASE_DIR / "monitoring" / "results" / "combined_results.json"

SWEEP_OUTPUT_PATH = BASE_DIR / "router" / "utility_sweep_summary.csv"
FINAL_OUTPUT_PATH = BASE_DIR / "router" / "utility_labels.csv"
HELD_OUT_OUTPUT_PATH = BASE_DIR / "router" / "held_out_images.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAMBDA_GRID = [0.3, 0.7, 1.2, 2.0]
MU_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]

CHOSEN_LAMBDA = 0.7
CHOSEN_MU = 0.02

LATENCY_NORM_PERCENTILE = 95

LOAD_STAGE_WEIGHTS = {"light": 0.5, "heavy": 0.35, "burst": 0.15}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Per-tier transforms matching each tier's training resolution
# Ratio Resize = IMG_SIZE * 256/224 preserved from training Colab scripts
fast_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),          # 224 * 256/224
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
balanced_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(293),          # 256 * 256/224 ≈293
    v2.CenterCrop(256),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
heavy_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(439),          # 384 * 256/224 ≈439
    v2.CenterCrop(384),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Map tier -> transform for easy lookup
TIER_TRANSFORMS = {
    "fast": fast_transform,
    "balanced": balanced_transform,
    "heavy": heavy_transform,
}

random.seed(42)
np.random.seed(42)


def load_models():
    models = {}

    # Fast
    ckpt_fast = torch.load(CHECKPOINT_DIR / "iwildcam_fast_mobilenetv3small_best.pt", map_location=DEVICE)
    n_classes_fast = ckpt_fast.get("num_classes", 182)
    class_names_fast = ckpt_fast.get("class_names", [f"class_{i}" for i in range(n_classes_fast)])
    if "class_names" not in ckpt_fast:
        print(f"[WARNING] fast checkpoint has no class_names -- using placeholder names. Real class names should be patched separately.")
    fast = build_fast_mobilenetv3small(num_classes=len(class_names_fast))
    fast.to(DEVICE)
    fast.load_state_dict(ckpt_fast["model_state_dict"])
    fast.eval()
    models["fast"] = (fast, class_names_fast)

    # Balanced
    ckpt_bal = torch.load(CHECKPOINT_DIR / "iwildcam_balanced_resnet18_best.pt", map_location=DEVICE)
    n_classes_bal = ckpt_bal.get("num_classes", 182)
    class_names_bal = ckpt_bal.get("class_names", [f"class_{i}" for i in range(n_classes_bal)])
    if "class_names" not in ckpt_bal:
        print(f"[WARNING] balanced checkpoint has no class_names -- using placeholder names. Real class names should be patched separately.")
    balanced = build_balanced_resnet18(num_classes=len(class_names_bal))
    balanced.to(DEVICE)
    balanced.load_state_dict(ckpt_bal["model_state_dict"])
    balanced.eval()
    models["balanced"] = (balanced, class_names_bal)

    # Heavy
    ckpt_heavy = torch.load(CHECKPOINT_DIR / "iwildcam_heavy_convnextbase_best.pt", map_location=DEVICE)
    n_classes_heavy = ckpt_heavy.get("num_classes", 182)
    class_names_heavy = ckpt_heavy.get("class_names", [f"class_{i}" for i in range(n_classes_heavy)])
    if "class_names" not in ckpt_heavy:
        print(f"[WARNING] heavy checkpoint has no class_names -- using placeholder names. Real class names should be patched separately.")
    heavy = build_heavy_convnextbase(num_classes=len(class_names_heavy))
    heavy.to(DEVICE)
    heavy.load_state_dict(ckpt_heavy["model_state_dict"])
    heavy.eval()
    models["heavy"] = (heavy, class_names_heavy)

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
    # true_label is now a placeholder string like "class_42" (converted from int
    # manifest label) so string comparison is consistent. This convention matches
    # CLASS_NAMES placeholder f"class_{i}" used when checkpoints lack real names.
    # Once real class names are patched into both checkpoints AND this formatting
    # logic together, comparison will automatically resolve correctly.
    correct = int(predicted_class == true_label)
    return correct, confidence


def load_class_names():
    """Load the canonical 182-entry class_names list from a checkpoint.
    All three tier checkpoints share the identical list, so any one works."""
    ckpt_path = CHECKPOINT_DIR / "iwildcam_fast_mobilenetv3small_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt["class_names"]


def load_manifest_samples(class_names):
    """
    Read UTILITY_LABEL_GEN_DIR/manifest.csv (columns: test_pool_index,
    image_path, label). Resolve image_path relative to UTILITY_LABEL_GEN_DIR.
    This manifest already IS the full working set -- use every row directly.
    """
    samples = []
    with open(MANIFEST_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # label is raw integer 0-181
            raw_label = int(row["label"])
            # Fixed -- was comparing placeholder strings (class_N) against real predicted class names,
            # causing every prediction to be scored incorrect regardless of actual correctness.
            # class_names now resolved consistently on both sides.
            true_label = class_names[raw_label]
            img_path = (UTILITY_LABEL_GEN_DIR / row["image_path"]).resolve()
            samples.append((img_path, true_label))
    return samples


def regenerate_held_out_from_final_manifest(class_names):
    """
    Regenerate held_out_images.csv from the already-separate final_holdout
    manifest (FINAL_HOLDOUT_DIR/manifest.csv). Simply copy/reformat
    (image_path, label) -> (image_path, true_label) where true_label is
    class_names[label] (real species name), resolving image_path relative to
    FINAL_HOLDOUT_DIR when writing so downstream scripts can find files.
    This disjointness was already established separately (SEPARATE
    final_holdout manifest/zip downloaded independently, NOT read elsewhere).
    """
    holdout_rows = []
    with open(FINAL_HOLDOUT_MANIFEST, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_label = int(row["label"])
            # Fixed -- was comparing placeholder strings (class_N) against real predicted class names,
            # causing every prediction to be scored incorrect regardless of actual correctness.
            # class_names now resolved consistently on both sides.
            true_label = class_names[raw_label]
            # Keep path relative to repo root for downstream: FINAL_HOLDOUT_DIR / image_path
            # But spec says resolving relative to FINAL_HOLDOUT_DIR when writing
            # so we write path relative to BASE_DIR so it resolves: data/iwildcam_final_holdout_subset/<image_path>
            rel_path = (FINAL_HOLDOUT_DIR / row["image_path"]).relative_to(BASE_DIR)
            holdout_rows.append((str(rel_path), true_label))

    with open(HELD_OUT_OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label"])
        writer.writerows(holdout_rows)
    print(f"  held-out set (from final_holdout manifest) written to {HELD_OUT_OUTPUT_PATH} ({len(holdout_rows)} rows)")


def main():
    print(f"Using device: {DEVICE}")

    # ---- Staleness / existence check for benchmarks and k6 latency ----
    benchmarks_exists = BENCHMARKS_PATH.exists()
    latency_exists = COMBINED_RESULTS_PATH.exists()
    # Compare mtimes if both exist and new checkpoints exist
    new_ckpts = [
        CHECKPOINT_DIR / "iwildcam_fast_mobilenetv3small_best.pt",
        CHECKPOINT_DIR / "iwildcam_balanced_resnet18_best.pt",
        CHECKPOINT_DIR / "iwildcam_heavy_convnextbase_best.pt",
    ]
    newest_ckpt_mtime = max((p.stat().st_mtime for p in new_ckpts if p.exists()), default=0)
    stale_warning = False
    if not benchmarks_exists or not latency_exists:
        stale_warning = True
        print("\n[WARNING] model_benchmarks.json / combined_results.json not found or may be stale "
              "(computed against the OLD produce-tier architectures). This script's output will be "
              "WRONG until these are regenerated against the new iWildCam-tier services. "
              "See project handover notes.")
        print(f"  BENCHMARKS_PATH exists: {benchmarks_exists} ({BENCHMARKS_PATH})")
        print(f"  COMBINED_RESULTS_PATH exists: {latency_exists} ({COMBINED_RESULTS_PATH})")
    elif newest_ckpt_mtime:
        bench_mtime = BENCHMARKS_PATH.stat().st_mtime if benchmarks_exists else 0
        latency_mtime = COMBINED_RESULTS_PATH.stat().st_mtime if latency_exists else 0
        if bench_mtime < newest_ckpt_mtime or latency_mtime < newest_ckpt_mtime:
            stale_warning = True
            print("\n[WARNING] model_benchmarks.json / combined_results.json may be stale "
                  "(computed against the OLD produce-tier architectures). This script's output will be "
                  "WRONG until these are regenerated against the new iWildCam-tier services. "
                  "See project handover notes.")
            print(f"  newest iWildCam checkpoint mtime: {newest_ckpt_mtime}")
            print(f"  benchmarks mtime: {bench_mtime}, latency mtime: {latency_mtime}")
    if stale_warning:
        print("  Proceeding anyway (structural correctness only) -- do not trust numeric output until regeneration.\n")
    else:
        print("  Benchmarks/latency files exist and appear fresh.\n")

    print("\nLoading models...")
    models = load_models()

    print("\nLoading resource benchmarks...")
    resource_norm = load_benchmarks()
    print(f"  resource_norm: {resource_norm}")

    print("\nLoading k6 latency distributions...")
    latency_dist = load_latency_distributions()

    all_latencies = [v for vals in latency_dist.values() for v in vals]
    max_latency = float(np.percentile(all_latencies, 95))
    true_max = max(all_latencies)
    print(f"  true max latency: {true_max:.2f} ms")
    print(f"  p95 latency (used for normalization): {max_latency:.2f} ms")

    print(f"\nLoading manifest samples from {MANIFEST_PATH}...")
    canonical_class_names = load_class_names()
    samples = load_manifest_samples(canonical_class_names)
    print(f"  working set (from manifest, every row used): {len(samples)}")

    print(f"\nRegenerating held-out set from final_holdout manifest...")
    regenerate_held_out_from_final_manifest(canonical_class_names)

    tiers = ["fast", "balanced", "heavy"]

    # ---- Pass 1: run inference + feature extraction ONCE per image, cache results ----
    print("\nRunning inference (once per image, cached for sweep)...")
    cached_rows = []
    for image_path, true_label in tqdm(samples, desc="Inference"):
        pil_image = Image.open(image_path).convert("RGB")
        # Per-tier transforms: produce three differently-sized tensors (resolution-independent features need only one image)
        image_tensors = {
            tier: TIER_TRANSFORMS[tier](pil_image)
            for tier in tiers
        }
        features = extract_features(image_path, pil_image)

        stage = random.choices(
            list(LOAD_STAGE_WEIGHTS.keys()),
            weights=list(LOAD_STAGE_WEIGHTS.values()),
        )[0]

        per_tier = {}
        for tier in tiers:
            model, class_names = models[tier]
            image_tensor = image_tensors[tier]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--regen-holdout-only", action="store_true",
        help="Only regenerate router/held_out_images.csv from the "
             "final_holdout manifest, skip everything else.")
    args = parser.parse_args()
    if args.regen_holdout_only:
        canonical_class_names = load_class_names()
        regenerate_held_out_from_final_manifest(canonical_class_names)
    else:
        main()
