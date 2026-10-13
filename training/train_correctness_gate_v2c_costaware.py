"""
training/train_correctness_gate_v2c_costaware.py

Trains CorrectnessGateNetV2c_costaware: same frozen MobileNetV3-Small
features[:5] backbone as V2b, same head architecture but WITHOUT Sigmoid
(raw logits output). Trained with a softmax-based cost-aware utility loss
(not BCE), so outputs are NOT calibrated P(correct) -- they are raw scores
used only for induced routing decisions (argmax of utility).

LAMBDA_TRAIN is selected dynamically from the V2b lambda sweep CSV
(correctness_gate_v2b_lambda_sweep_finegrid.csv) -- the non-degenerate
lambda with the highest balanced_routing_acc.

Comparison metric: induced routing decision quality (induced_accuracy,
induced_expected_cost), NOT AUROC. AUROC does not apply to raw logits.

Run: python training/train_correctness_gate_v2c_costaware.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
import mlflow
import matplotlib.pyplot as plt

from models.correctness_gate import CorrectnessGateNetV2c_costaware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "router"))
from router_eval_utils import random_baseline_routing_acc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_correctness_gate import (
    CorrectnessGateDataset, load_resource_norm,
    BASE_DIR, LABELS_PATH, CHECKPOINT_DIR,
    TARGET_COLUMNS, TIER_NAMES, IMAGENET_MEAN, IMAGENET_STD,
    train_transform, eval_transform,
)

IMG_SIZE = 224
EPOCHS = 25
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MLFLOW_EXPERIMENT = "inferpilot-model-tiers"

TARGET_COLUMNS = ["fast_correct", "balanced_correct", "heavy_correct"]
TIER_NAMES = ["fast", "balanced", "heavy"]

# Cost-aware loss constants
T_TARGET = 0.3
T_PRED = 1.0

# Lambda sweep CSVs -- prefer V2b fine-grid if available, fall back to V1
V2B_FINESWEEP_CSV = CHECKPOINT_DIR / "correctness_gate_v2b_lambda_sweep_finegrid.csv"
V1_SWEEP_CSV = CHECKPOINT_DIR / "correctness_gate_lambda_sweep.csv"
DEGENERATE_PCT = 95.0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
eval_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(IMG_SIZE),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def cost_aware_utility_loss(raw_scores, targets, costs_tensor, lam,
                            t_target=T_TARGET, t_pred=T_PRED):
    """
    raw_scores: [batch, 3] model output logits.
    targets:    [batch, 3] ground-truth per-tier correctness (0/1).
    costs_tensor: [3] normalized per-tier cost, broadcast across the batch.
    """
    target_utility = targets - lam * costs_tensor.unsqueeze(0)
    q = torch.softmax(target_utility / t_target, dim=1)
    pred_utility = raw_scores - lam * costs_tensor.unsqueeze(0)
    log_p = torch.log_softmax(pred_utility / t_pred, dim=1)
    loss = -(q * log_p).sum(dim=1).mean()
    return loss


@torch.no_grad()
def evaluate_induced_decision(model, loader, costs_tensor, lam):
    """
    Returns val_loss, induced_accuracy, induced_expected_cost, chosen, targets_all.
    induced decision per image: chosen = argmax(raw_scores - lam*costs, dim=1)
    induced_accuracy = mean(targets[i, chosen[i]] for all i)
    induced_expected_cost = mean(costs[chosen[i]] for all i)
    """
    model.eval()
    total_loss, total = 0.0, 0
    all_raw, all_targets = [], []
    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        raw_scores = model(images)
        loss = cost_aware_utility_loss(raw_scores, targets, costs_tensor, lam)
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        all_raw.append(raw_scores.cpu())
        all_targets.append(targets.cpu())

    val_loss = total_loss / total
    raw_all = torch.cat(all_raw, dim=0).numpy()
    targets_all = torch.cat(all_targets, dim=0).numpy()

    costs_np = np.asarray(costs_tensor.cpu())
    utilities = raw_all - lam * costs_np
    chosen = np.argmax(utilities, axis=1)

    induced_acc = float(np.mean([targets_all[i, c] for i, c in enumerate(chosen)]))
    induced_cost = float(np.mean([costs_np[c] for c in chosen]))

    return val_loss, induced_acc, induced_cost, chosen, targets_all


def compute_balanced_routing_acc(chosen, targets_all):
    """
    Macro-averaged balanced routing accuracy: for each tier, mean(correct)
    among images actually routed to that tier, then nanmean across tiers
    with >=1 routed image. Same formula as sweep_routing() in
    train_correctness_gate.py.
    """
    per_tier_acc = []
    for t in range(3):
        mask = chosen == t
        if mask.sum() > 0:
            per_tier_acc.append(float(targets_all[mask, t].mean()))
    return float(np.nanmean(per_tier_acc)) if per_tier_acc else float("nan")


def compute_tier_pcts(chosen):
    """Tier distribution as percentages."""
    return [float((chosen == t).mean()) * 100 for t in range(3)]


def select_lambda_train():
    """
    Load the V2b fine-grid lambda sweep CSV, filter degenerate rows,
    pick the lambda with the highest balanced_routing_acc.
    """
    sweep_path = V2B_FINESWEEP_CSV if V2B_FINESWEEP_CSV.exists() else V1_SWEEP_CSV

    if not sweep_path.exists():
        raise FileNotFoundError(
            f"Lambda sweep CSV not found: {sweep_path}\n"
            f"Please run the appropriate training script first:\n"
            f"  - For V2b fine-grid: training/train_correctness_gate_v2b.py\n"
            f"  - For V1: training/train_correctness_gate.py"
        )

    sweep_df = pd.read_csv(sweep_path)
    print(f"Loaded lambda sweep from: {sweep_path}")
    print(f"  Total lambdas: {len(sweep_df)}")

    pct_cols = ["pct_fast", "pct_balanced", "pct_heavy"]
    max_pct = sweep_df[pct_cols].max(axis=1)
    non_degenerate = sweep_df[max_pct <= DEGENERATE_PCT].copy()

    print(f"  Non-degenerate lambdas (no tier >{DEGENERATE_PCT:.0f}%): {len(non_degenerate)}")

    if len(non_degenerate) == 0:
        raise RuntimeError(
            f"Every swept lambda is degenerate (routes >{DEGENERATE_PCT:.0f}% of images "
            f"to a single tier). Cannot select LAMBDA_TRAIN from sweep data.\n"
            f"Sweep file: {sweep_path}"
        )

    best_idx = int(non_degenerate["balanced_routing_acc"].idxmax())
    best_row = non_degenerate.loc[best_idx]
    lam = float(best_row["lambda"])
    acc = float(best_row["balanced_routing_acc"])

    print(f"\nLAMBDA_TRAIN selected from sweep: {lam:.4f}")
    print(f"  balanced_routing_acc = {acc:.4f}")
    print(f"  pct_fast/balanced/heavy = {best_row['pct_fast']:.1f}%/"
          f"{best_row['pct_balanced']:.1f}%/{best_row['pct_heavy']:.1f}%")

    return lam


def _print_diagnostic_report(chosen, targets_all, pcts, bal_acc, lam_train, costs_all):
    """Print full diagnostic: tier distribution, balanced acc, random baseline, verdict, V2b comparison."""
    n = len(chosen)

    # Tier distribution
    print(f"\nInduced tier distribution:")
    for i, tier in enumerate(TIER_NAMES):
        print(f"  {tier:9s}: {pcts[i]:5.1f}%")

    # Per-tier accuracy among routed images
    print(f"\nPer-tier accuracy (among images routed to each tier):")
    for t in range(3):
        mask = chosen == t
        if mask.sum() > 0:
            acc = float(targets_all[mask, t].mean())
            print(f"  {TIER_NAMES[t]:9s}: acc={acc:.4f} (n={int(mask.sum())})")
        else:
            print(f"  {TIER_NAMES[t]:9s}: n=0 (skipped)")

    print(f"  Balanced routing accuracy: {bal_acc:.4f}")

    # Random baseline comparison
    rnd_mean, rnd_std = random_baseline_routing_acc(targets_all, pcts[0], pcts[1], pcts[2])
    lift = bal_acc - rnd_mean
    sigma = lift / rnd_std if rnd_std > 0 else (float("inf") if lift > 0 else 0.0)
    print(f"\nRandom baseline (matched proportions): {rnd_mean:.4f} +/- {rnd_std:.4f}")
    print(f"Model lift over random: +{lift:.4f} ({sigma:.1f}sigma)")
    if sigma < 2:
        print("WARNING: model's advantage over random routing is not clearly "
              "distinguishable from noise.")

    # Collapse verdict
    print(f"\n{'='*60}")
    collapsed_tier = None
    for i, tier in enumerate(TIER_NAMES):
        if pcts[i] >= 90.0:
            collapsed_tier = tier
            break
    if collapsed_tier:
        print(f"STILL LARGELY COLLAPSED: model is routing near-uniformly to "
              f"{collapsed_tier} regardless of image content.")
    else:
        print("NOT COLLAPSED: model shows per-image discrimination across tiers.")
    print(f"{'='*60}")

    # V2b comparison
    sweep_path = V2B_FINESWEEP_CSV if V2B_FINESWEEP_CSV.exists() else V1_SWEEP_CSV
    print(f"\n{'='*60}\nAPPLES-TO-APPLES COMPARISON AT SAME LAMBDA\n{'='*60}")
    print("NOTE: V2c-costaware outputs are NOT calibrated P(correct) and are NOT "
          "AUROC-comparable to V1/V2/V2b. Comparison metric is induced routing "
          "decision quality only.")

    if sweep_path.exists():
        sweep_df = pd.read_csv(sweep_path)
        sweep_df["lambda_diff"] = (sweep_df["lambda"] - lam_train).abs()
        closest_idx = int(sweep_df["lambda_diff"].idxmin())
        closest_row = sweep_df.loc[closest_idx]
        closest_lam = float(closest_row["lambda"])

        print(f"\n  V2c-costaware induced_balanced_accuracy : {bal_acc:.4f}")
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
        print(f"\nWARNING: V2b sweep CSV not found at {sweep_path}")
        print(f"  V2c-costaware induced_balanced_accuracy: {bal_acc:.4f}")


def main():
    print(f"Using device: {DEVICE}")

    LAMBDA_TRAIN = select_lambda_train()

    df = pd.read_csv(LABELS_PATH)
    df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)

    missing = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing required target column(s) in {LABELS_PATH}: {missing}. "
            f"Actual columns found: {list(df.columns)}"
        )
    print(f"\nLoaded {len(df)} labeled samples")
    print(f"Per-tier correctness rates:")
    for c in TARGET_COLUMNS:
        print(f"  {c}: {df[c].mean():.3f}")

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["best_tier"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["best_tier"], random_state=SEED
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    train_loader = DataLoader(
        CorrectnessGateDataset(train_df, train_transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        CorrectnessGateDataset(val_df, eval_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        CorrectnessGateDataset(test_df, eval_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
    )

    model = CorrectnessGateNetV2c_costaware(num_tiers=3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nCorrectnessGateNetV2c_costaware params: {n_params:,}")

    resource_norm = load_resource_norm()
    costs_all = np.array([resource_norm[t] for t in TIER_NAMES], dtype=np.float32)
    costs_tensor = torch.tensor(costs_all, dtype=torch.float32).to(DEVICE)
    print(f"Costs: fast={costs_all[0]:.3f} balanced={costs_all[1]:.3f} heavy={costs_all[2]:.3f}")
    print(f"LAMBDA_TRAIN = {LAMBDA_TRAIN:.4f}")

    best_ckpt_path = CHECKPOINT_DIR / "correctness_gate_v2c_costaware_best.pt"

    if best_ckpt_path.exists():
        print(f"\nCheckpoint already exists: {best_ckpt_path}")
        print(f"Skipping training -- loading checkpoint and running test-set evaluation only.\n")
        ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        test_loss, test_acc, test_cost, test_chosen, test_targets = evaluate_induced_decision(
            model, test_loader, costs_tensor, LAMBDA_TRAIN
        )
        test_pcts = compute_tier_pcts(test_chosen)
        test_bal_acc = compute_balanced_routing_acc(test_chosen, test_targets)
        print(f"\nTest induced_accuracy: {test_acc:.4f}")
        print(f"Test balanced_routing_accuracy: {test_bal_acc:.4f}")
        print(f"Test induced_expected_cost: {test_cost:.4f}")
        _print_diagnostic_report(test_chosen, test_targets, test_pcts, test_bal_acc,
                                 LAMBDA_TRAIN, costs_all)
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    history = {"train_loss": [], "val_loss": [],
               "val_bal_acc": [], "val_induced_acc": [], "val_induced_cost": []}
    best_val_bal_acc = -1.0
    best_val_idx = -1

    with mlflow.start_run(run_name="correctness_gate_v2c_costaware"):
        mlflow.log_params({
            "architecture": "CorrectnessGateNetV2c_costaware",
            "num_params": n_params,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "targets": TARGET_COLUMNS,
            "lambda_train": LAMBDA_TRAIN,
            "t_target": T_TARGET,
            "t_pred": T_PRED,
            "loss": "cost_aware_utility_loss",
        })

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            model.train()
            running_loss, total = 0.0, 0
            for images, targets in train_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)
                optimizer.zero_grad()
                raw_scores = model(images)
                loss = cost_aware_utility_loss(raw_scores, targets, costs_tensor, LAMBDA_TRAIN)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                total += images.size(0)
            train_loss = running_loss / total

            val_loss, val_acc, val_cost, val_chosen, val_targets = evaluate_induced_decision(
                model, val_loader, costs_tensor, LAMBDA_TRAIN
            )
            val_bal_acc = compute_balanced_routing_acc(val_chosen, val_targets)
            scheduler.step()
            epoch_time = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_bal_acc"].append(val_bal_acc)
            history["val_induced_acc"].append(val_acc)
            history["val_induced_cost"].append(val_cost)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_balanced_routing_acc": val_bal_acc,
                "val_induced_accuracy": val_acc,
                "val_induced_expected_cost": val_cost,
                "epoch_time_sec": epoch_time,
            }, step=epoch)

            print(f"Epoch {epoch:2d}/{EPOCHS} | "
                  f"train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | "
                  f"val_bal_acc {val_bal_acc:.4f} | "
                  f"val_induced_acc {val_acc:.4f} | "
                  f"val_induced_cost {val_cost:.4f} | "
                  f"{epoch_time:.1f}s")

            # Checkpoint on best balanced_routing_acc, tiebreak by lowest cost
            is_better = val_bal_acc > best_val_bal_acc or (
                val_bal_acc == best_val_bal_acc and best_val_idx >= 0
                and val_cost < history["val_induced_cost"][best_val_idx]
            )
            if is_better:
                best_val_bal_acc = val_bal_acc
                best_val_idx = len(history["val_bal_acc"]) - 1
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_names": TARGET_COLUMNS,
                    "val_balanced_routing_acc": val_bal_acc,
                    "val_induced_accuracy": val_acc,
                    "val_induced_expected_cost": val_cost,
                    "epoch": epoch,
                    "lambda_train": LAMBDA_TRAIN,
                }, best_ckpt_path)
                print(f"  -> new best val_bal_acc {val_bal_acc:.4f}, checkpoint saved")

        # ---- Final test evaluation ----
        best_ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_loss, test_acc, test_cost, test_chosen, test_targets = evaluate_induced_decision(
            model, test_loader, costs_tensor, LAMBDA_TRAIN
        )
        test_pcts = compute_tier_pcts(test_chosen)
        test_bal_acc = compute_balanced_routing_acc(test_chosen, test_targets)

        mlflow.log_metrics({
            "test_balanced_routing_acc": test_bal_acc,
            "test_induced_accuracy": test_acc,
            "test_induced_expected_cost": test_cost,
            "test_loss": test_loss,
            "test_pct_fast": test_pcts[0],
            "test_pct_balanced": test_pcts[1],
            "test_pct_heavy": test_pcts[2],
        })

        print(f"\n{'='*60}\nFINAL TEST RESULTS\n{'='*60}")
        print(f"  lambda_train:              {LAMBDA_TRAIN:.4f}")
        print(f"  test induced_acc:          {test_acc:.4f}")
        print(f"  test balanced_routing_acc: {test_bal_acc:.4f}")
        print(f"  test induced_cost:         {test_cost:.4f}")

        _print_diagnostic_report(test_chosen, test_targets, test_pcts, test_bal_acc,
                                 LAMBDA_TRAIN, costs_all)

        # ---- Plot loss/metric curves ----
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs_range = range(1, EPOCHS + 1)
        axes[0].plot(epochs_range, history["train_loss"], label="Train")
        axes[0].plot(epochs_range, history["val_loss"], label="Val")
        axes[0].set_title("Cost-Aware Utility Loss")
        axes[0].legend()
        axes[1].plot(epochs_range, history["val_bal_acc"], label="Val Balanced Routing Acc")
        axes[1].plot(epochs_range, history["val_induced_acc"], label="Val Induced Acc (plain)")
        ax2 = axes[1].twinx()
        ax2.plot(epochs_range, history["val_induced_cost"], label="Val Induced Cost",
                 color="red", linestyle="--")
        axes[1].set_title("Routing Decision Quality")
        axes[1].set_ylabel("Accuracy")
        ax2.set_ylabel("Induced Expected Cost")
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, loc="center right")
        fig.tight_layout()
        curve_path = CHECKPOINT_DIR / "correctness_gate_v2c_costaware_curves.png"
        fig.savefig(curve_path)
        plt.close(fig)
        mlflow.log_artifact(str(curve_path))
        mlflow.log_artifact(str(best_ckpt_path))

    print(f"\nDone. Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
