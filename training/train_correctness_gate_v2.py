"""
Training script for CorrectnessGateNetV2 (frozen MobileNetV3-Small backbone +
trainable head). Mirrors train_correctness_gate.py exactly: same seed=42,
same stratified 70/15/15 split, same BCELoss, same AdamW/CosineAnnealing,
same checkpointing-by-avg-AUROC. Only the model class and the optimizer's
parameter list differ (head params only).

Saves to: training/checkpoints/correctness_gate_v2_best.pt
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import mlflow
import matplotlib.pyplot as plt

from models.correctness_gate import CorrectnessGateNetV2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "router"))
from router_eval_utils import random_baseline_routing_acc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_correctness_gate import (
    CorrectnessGateDataset, per_tier_metrics, load_resource_norm,
    BASE_DIR, LABELS_PATH, BENCHMARKS_PATH, CHECKPOINT_DIR,
    TARGET_COLUMNS, TIER_NAMES, IMAGENET_MEAN, IMAGENET_STD,
    train_transform, eval_transform, LAMBDA_RANGE, DEGENERATE_PCT,
)

IMG_SIZE = 224
EPOCHS = 25
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MLFLOW_EXPERIMENT = "inferpilot-model-tiers"
ROUTE_LAMBDA = 0.7


def sweep_routing(test_preds, test_true, costs):
    rows = []
    for lam in LAMBDA_RANGE:
        chosen = np.argmax(test_preds - lam * costs, axis=1)
        pcts = [float((chosen == t).mean()) * 100 for t in range(3)]
        per_tier_acc = []
        for t in range(3):
            mask = chosen == t
            if mask.sum() > 0:
                per_tier_acc.append(float(test_true[mask, t].mean()))
        bal_acc = float(np.nanmean(per_tier_acc)) if per_tier_acc else float("nan")
        rnd_mean, rnd_std = random_baseline_routing_acc(test_true, pcts[0], pcts[1], pcts[2])
        rows.append({
            "lambda": lam, "pct_fast": round(pcts[0], 2),
            "pct_balanced": round(pcts[1], 2), "pct_heavy": round(pcts[2], 2),
            "balanced_routing_acc": round(bal_acc, 4),
            "random_baseline_mean": round(rnd_mean, 4),
            "random_baseline_std": round(rnd_std, 4),
        })
    return pd.DataFrame(rows)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, total = 0.0, 0
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        total += images.size(0)
    return running_loss / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, total = 0.0, 0
    all_preds, all_targets = [], []
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        outputs = model(images)
        running_loss += criterion(outputs, targets).item() * images.size(0)
        total += images.size(0)
        all_preds.append(outputs.cpu())
        all_targets.append(targets.cpu())
    val_loss = running_loss / total
    y_pred = torch.cat(all_preds, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy()
    accs, aurocs = per_tier_metrics(y_true, y_pred)
    avg_auroc = float(np.nanmean(aurocs))
    return val_loss, accs, aurocs, avg_auroc, y_pred, y_true


def _report(test_accs, test_aurocs, test_avg_auroc, test_preds, test_true):
    resource_norm = load_resource_norm()
    costs = np.array([resource_norm[t] for t in TIER_NAMES])

    v1_aurocs = [0.6764, 0.7692, 0.6748]
    print(f"\n--- V2 vs V1 AUROC comparison ---")
    for i, tier in enumerate(TIER_NAMES):
        delta = test_aurocs[i] - v1_aurocs[i]
        sign = "+" if delta >= 0 else ""
        print(f"  {tier:9s}: V1={v1_aurocs[i]:.4f}  V2={test_aurocs[i]:.4f}  delta={sign}{delta:.4f}")
    v1_avg = float(np.nanmean(v1_aurocs))
    delta_avg = test_avg_auroc - v1_avg
    sign = "+" if delta_avg >= 0 else ""
    print(f"  {'avg':9s}: V1={v1_avg:.4f}  V2={test_avg_auroc:.4f}  delta={sign}{delta_avg:.4f}")

    sweep_df = sweep_routing(test_preds, test_true, costs)
    best_idx = int(sweep_df["balanced_routing_acc"].idxmax())
    best = sweep_df.loc[best_idx]
    print(f"\nV2 best balanced_routing_acc: lambda={best['lambda']:.2f} -> {best['balanced_routing_acc']:.4f}")
    print(f"  random baseline: {best['random_baseline_mean']:.4f} +/- {best['random_baseline_std']:.4f}")


def main():
    print(f"Using device: {DEVICE}")
    df = pd.read_csv(LABELS_PATH)
    df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)

    missing = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing target columns: {missing}")
    print(f"Loaded {len(df)} labeled samples")

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["best_tier"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["best_tier"], random_state=SEED
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    train_loader = DataLoader(CorrectnessGateDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(CorrectnessGateDataset(val_df, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(CorrectnessGateDataset(test_df, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = CorrectnessGateNetV2(num_tiers=3).to(DEVICE)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CorrectnessGateNetV2 total params: {n_total:,}  trainable: {n_trainable:,}")

    criterion = nn.BCELoss()
    best_ckpt_path = CHECKPOINT_DIR / "correctness_gate_v2_best.pt"

    if best_ckpt_path.exists():
        print(f"\nCheckpoint exists: {best_ckpt_path} -- loading and evaluating only.\n")
        ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        _, test_accs, test_aurocs, test_avg_auroc, test_preds, test_true = evaluate(model, test_loader, criterion)
        _report(test_accs, test_aurocs, test_avg_auroc, test_preds, test_true)
        return

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Optimizer: {sum(p.numel() for p in trainable_params):,} trainable params")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    history = {"train_loss": [], "val_loss": [], "val_auroc": [[] for _ in range(3)]}
    best_avg_auroc = 0.0

    with mlflow.start_run(run_name="correctness_gate_v2"):
        mlflow.log_params({
            "architecture": "CorrectnessGateNetV2",
            "backbone": "MobileNetV3-Small features[:5] (frozen)",
            "total_params": n_total, "trainable_params": n_trainable,
            "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE, "targets": TARGET_COLUMNS,
        })

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, accs, aurocs, avg_auroc, _, _ = evaluate(model, val_loader, criterion)
            scheduler.step()
            epoch_time = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            for t in range(3):
                history["val_auroc"][t].append(aurocs[t])

            mlflow.log_metrics({
                "train_loss": train_loss, "val_loss": val_loss,
                "val_fast_auroc": aurocs[0], "val_balanced_auroc": aurocs[1],
                "val_heavy_auroc": aurocs[2], "val_avg_auroc": avg_auroc,
                "epoch_time_sec": epoch_time,
            }, step=epoch)

            print(f"Epoch {epoch:2d}/{EPOCHS} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} avg_auroc {avg_auroc:.4f} | "
                  f"auroc(f/b/h) {aurocs[0]:.3f}/{aurocs[1]:.3f}/{aurocs[2]:.3f} | {epoch_time:.1f}s")

            if avg_auroc > best_avg_auroc:
                best_avg_auroc = avg_auroc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_names": TARGET_COLUMNS,
                    "val_auroc": aurocs, "val_avg_auroc": avg_auroc, "epoch": epoch,
                }, best_ckpt_path)
                print(f"  -> new best val_avg_auroc {avg_auroc:.4f}, checkpoint saved")

        best_ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        model.load_state_dict(best_ckpt["model_state_dict"])
        _, test_accs, test_aurocs, test_avg_auroc, test_preds, test_true = evaluate(model, test_loader, criterion)

        mlflow.log_metrics({
            "test_avg_auroc": test_avg_auroc,
            "test_fast_auroc": test_aurocs[0], "test_balanced_auroc": test_aurocs[1],
            "test_heavy_auroc": test_aurocs[2],
        })

        print(f"\n{'='*60}\nFINAL TEST RESULTS (V2)\n{'='*60}")
        for i, tier in enumerate(TIER_NAMES):
            print(f"  {tier:9s}: acc {test_accs[i]:.4f}  auroc {test_aurocs[i]:.4f}")

        _report(test_accs, test_aurocs, test_avg_auroc, test_preds, test_true)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs_range = range(1, EPOCHS + 1)
        axes[0].plot(epochs_range, history["train_loss"], label="Train")
        axes[0].plot(epochs_range, history["val_loss"], label="Val")
        axes[0].set_title("Loss (V2)"); axes[0].legend()
        for t, tier in enumerate(TIER_NAMES):
            axes[1].plot(epochs_range, history["val_auroc"][t], label=f"Val {tier} AUROC")
        axes[1].set_title("Per-Tier AUROC (V2)"); axes[1].legend()
        curve_path = CHECKPOINT_DIR / "correctness_gate_v2_curves.png"
        fig.savefig(curve_path); plt.close(fig)
        mlflow.log_artifact(str(curve_path))
        mlflow.log_artifact(str(best_ckpt_path))

    print(f"\nDone. Best V2 checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
