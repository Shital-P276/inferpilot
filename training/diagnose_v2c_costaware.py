"""
training/diagnose_v2c_costaware.py

Read-only diagnostic for CorrectnessGateNetV2c_costaware:
- Checks whether induced routing decisions collapsed onto a single tier
- Computes balanced (macro-averaged) routing accuracy (apples-to-apples vs V2b)
- Random-baseline comparison to separate skill from base-rate noise

Run: python training/diagnose_v2c_costaware.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models.correctness_gate import CorrectnessGateNetV2c_costaware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "router"))
from router_eval_utils import random_baseline_routing_acc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_correctness_gate import (
    CorrectnessGateDataset, load_resource_norm,
    BASE_DIR, LABELS_PATH, CHECKPOINT_DIR,
    TARGET_COLUMNS, TIER_NAMES, eval_transform,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 32
V2B_SWEEP_CSV = CHECKPOINT_DIR / "correctness_gate_v2b_lambda_sweep_finegrid.csv"


def main():
    print("=" * 60)
    print("V2c-costaware Diagnostic: Collapse Check + Balanced Accuracy")
    print("=" * 60)

    # ---- 1. Load checkpoint ----
    ckpt_path = CHECKPOINT_DIR / "correctness_gate_v2c_costaware_best.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    lambda_train = float(ckpt["lambda_train"])
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"  lambda_train = {lambda_train:.4f}")
    print(f"  saved val_induced_accuracy = {ckpt['val_induced_accuracy']:.4f}")
    print(f"  saved val_induced_expected_cost = {ckpt['val_induced_expected_cost']:.4f}")
    print(f"  epoch = {ckpt['epoch']}")

    # ---- 2. Rebuild same test split as training ----
    df = pd.read_csv(LABELS_PATH)
    df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)
    missing = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing target columns: {missing}")

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["best_tier"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["best_tier"], random_state=SEED
    )
    print(f"\nSplit: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    test_loader = DataLoader(
        CorrectnessGateDataset(test_df, eval_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
    )

    # ---- 3. Load model, run inference ----
    model = CorrectnessGateNetV2c_costaware(num_tiers=3).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_raw, all_targets = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(DEVICE)
            raw_scores = model(images)
            all_raw.append(raw_scores.cpu().numpy())
            all_targets.append(targets.numpy())

    raw_all = np.concatenate(all_raw, axis=0)       # [N, 3]
    targets_all = np.concatenate(all_targets, axis=0) # [N, 3]
    n_test = len(raw_all)
    print(f"\nTest images: {n_test}")

    # ---- 4. Load costs ----
    resource_norm = load_resource_norm()
    costs_all = np.array([resource_norm[t] for t in TIER_NAMES], dtype=np.float32)
    print(f"Costs: fast={costs_all[0]:.3f} balanced={costs_all[1]:.3f} heavy={costs_all[2]:.3f}")

    # ---- 5. Induced decisions ----
    utilities = raw_all - lambda_train * costs_all
    chosen = np.argmax(utilities, axis=1)

    # ---- 6. Tier distribution ----
    pcts = [float((chosen == t).mean()) * 100 for t in range(3)]
    print(f"\nInduced tier distribution:")
    for i, tier in enumerate(TIER_NAMES):
        print(f"  {tier:9s}: {pcts[i]:5.1f}%")

    # ---- 7a. Plain accuracy ----
    plain_acc = float(np.mean([targets_all[i, c] for i, c in enumerate(chosen)]))
    print(f"\nPlain induced_accuracy: {plain_acc:.4f}")

    # ---- 7b. Balanced routing accuracy (macro-averaged) ----
    per_tier_acc = []
    per_tier_count = []
    for t in range(3):
        mask = chosen == t
        count = int(mask.sum())
        per_tier_count.append(count)
        if count > 0:
            acc = float(targets_all[mask, t].mean())
            per_tier_acc.append(acc)
            print(f"  {TIER_NAMES[t]:9s}: acc={acc:.4f} (n={count})")
        else:
            print(f"  {TIER_NAMES[t]:9s}: n=0 (skipped)")
    bal_acc = float(np.nanmean(per_tier_acc)) if per_tier_acc else float("nan")
    print(f"  Balanced routing accuracy: {bal_acc:.4f}")

    # ---- 8. Random baseline comparison ----
    rnd_mean, rnd_std = random_baseline_routing_acc(targets_all, pcts[0], pcts[1], pcts[2])
    lift = bal_acc - rnd_mean
    sigma = lift / rnd_std if rnd_std > 0 else (float("inf") if lift > 0 else 0.0)
    print(f"\nRandom baseline (matched proportions): {rnd_mean:.4f} +/- {rnd_std:.4f}")
    print(f"Model lift over random: +{lift:.4f} ({sigma:.1f}sigma)")
    if sigma < 2:
        print("WARNING: model's advantage over random routing is not clearly "
              "distinguishable from noise.")

    # ---- 9. Collapse verdict ----
    print(f"\n{'='*60}")
    collapsed_tier = None
    for i, tier in enumerate(TIER_NAMES):
        if pcts[i] >= 95.0:
            collapsed_tier = tier
            break
    if collapsed_tier:
        print(f"COLLAPSED: model is routing near-uniformly to {collapsed_tier} "
              f"regardless of image content.")
    else:
        print("NOT COLLAPSED: model shows per-image discrimination across tiers.")
    print(f"{'='*60}")

    # ---- 10. V2b comparison table ----
    print(f"\n{'='*60}")
    print("APPLES-TO-APPLES COMPARISON AT SAME LAMBDA")
    print(f"{'='*60}")

    if V2B_SWEEP_CSV.exists():
        sweep_df = pd.read_csv(V2B_SWEEP_CSV)
        sweep_df["lambda_diff"] = (sweep_df["lambda"] - lambda_train).abs()
        closest_idx = int(sweep_df["lambda_diff"].idxmin())
        closest_row = sweep_df.loc[closest_idx]
        closest_lam = float(closest_row["lambda"])

        print(f"  V2c-costaware induced_balanced_accuracy : {bal_acc:.4f}")
        print(f"  V2b post-hoc balanced_routing_acc       : {closest_row['balanced_routing_acc']:.4f}")
        print(f"  (V2b at lambda={closest_lam:.4f}, "
              f"pct_fast/bal/heavy={closest_row['pct_fast']:.1f}%/"
              f"{closest_row['pct_balanced']:.1f}%/{closest_row['pct_heavy']:.1f}%)")

        diff = bal_acc - closest_row["balanced_routing_acc"]
        print(f"\n  Difference (V2c - V2b): {diff:+.4f}")
        if diff > 0:
            print(f"  V2c-costaware WINS by {diff:.4f}")
        else:
            print(f"  V2b post-hoc WINS by {-diff:.4f}")
    else:
        print(f"WARNING: V2b sweep CSV not found at {V2B_SWEEP_CSV}")
        print(f"  V2c-costaware induced_balanced_accuracy: {bal_acc:.4f}")

    print(f"\nNOTE: V2c-costaware outputs are NOT calibrated P(correct) and are NOT "
          f"AUROC-comparable to V1/V2/V2b. Comparison metric is induced routing "
          f"decision quality only.")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
