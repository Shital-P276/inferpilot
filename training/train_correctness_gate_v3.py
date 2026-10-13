"""
Training script for CorrectnessGateNetV3 (frozen MobileNetV3-Small features[:5]
with last block unfrozen + dual heads: correctness + confidence).

Same setup as V2b: seed=42, 70/15/15 split, AdamW/CosineAnnealing,
checkpointing-by-avg-AUROC. Saves to: training/checkpoints/correctness_gate_v3_best.pt

Changes vs V2b:
- V3 has unfrozen last backbone block (vs V2b fully frozen)
- Two heads sharing shared trunk: correctness (BCE) + confidence (MSE, weight 0.5)
- Per-tier weighted BCE from label imbalance, normalized sum to 3
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import mlflow
import matplotlib.pyplot as plt

from models.correctness_gate import CorrectnessGateNetV3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "router"))
from router_eval_utils import random_baseline_routing_acc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_correctness_gate import (
    CorrectnessGateDataset, per_tier_metrics, load_resource_norm,
    BASE_DIR, LABELS_PATH, CHECKPOINT_DIR,
    TARGET_COLUMNS, TIER_NAMES, IMAGENET_MEAN, IMAGENET_STD,
    train_transform, eval_transform, LAMBDA_RANGE,
)

EPOCHS = 25
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIDENCE_COLUMNS = ["fast_confidence", "balanced_confidence", "heavy_confidence"]
CONFIDENCE_LOSS_WEIGHT = 0.5
BCE_EPSILON = 1e-6

# V2b baseline (same utility_labels.csv, honest post-fix) for comparison
V2B_BASELINE = {"fast": 0.7589, "balanced": 0.7440, "heavy": 0.7376, "avg": 0.7469}


class CorrectnessGateV3Dataset(Dataset):
    """Returns image + correctness [3] + confidence [3] targets."""
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = BASE_DIR / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        correctness = torch.tensor([row[c] for c in TARGET_COLUMNS], dtype=torch.float32)
        confidence = torch.tensor([row[c] for c in CONFIDENCE_COLUMNS], dtype=torch.float32)
        return image, correctness, confidence


def compute_bce_weights(df):
    """Per-tier BCE weights from positive rates: 1/(pos_rate+eps), normalized sum to 3."""
    pos_rates = np.array([df[c].mean() for c in TARGET_COLUMNS], dtype=float)
    raw_weights = 1.0 / (pos_rates + BCE_EPSILON)
    # Normalize to sum to 3 (keeps overall magnitude comparable to unweighted)
    norm_weights = raw_weights * (3.0 / raw_weights.sum())
    weights_tensor = torch.tensor(norm_weights, dtype=torch.float32)
    print(f"Per-tier positive rates: fast={pos_rates[0]:.4f} balanced={pos_rates[1]:.4f} heavy={pos_rates[2]:.4f}")
    print(f"Raw BCE weights (1/(pos+eps)): fast={raw_weights[0]:.4f} balanced={raw_weights[1]:.4f} heavy={raw_weights[2]:.4f}")
    print(f"Normalized BCE weights (sum to 3): fast={norm_weights[0]:.4f} balanced={norm_weights[1]:.4f} heavy={norm_weights[2]:.4f}")
    return weights_tensor


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


def train_one_epoch(model, loader, bce_weights, mse_criterion, optimizer):
    model.train()
    running_loss, total = 0.0, 0
    bce_none = nn.BCELoss(reduction='none')
    for images, corr_targets, conf_targets in loader:
        images = images.to(DEVICE)
        corr_targets = corr_targets.to(DEVICE)
        conf_targets = conf_targets.to(DEVICE)
        optimizer.zero_grad()
        corr_pred, conf_pred = model(images)
        # Weighted BCE per tier
        bce_per_elem = bce_none(corr_pred, corr_targets)  # [B,3]
        # Apply per-tier weights
        w = bce_weights.to(DEVICE)  # [3]
        weighted_bce = (bce_per_elem * w).mean()
        confidence_loss = mse_criterion(conf_pred, conf_targets)
        total_loss = weighted_bce + CONFIDENCE_LOSS_WEIGHT * confidence_loss
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item() * images.size(0)
        total += images.size(0)
    return running_loss / total


@torch.no_grad()
def evaluate(model, loader, bce_weights, mse_criterion):
    model.eval()
    running_loss, total = 0.0, 0
    all_corr_preds, all_corr_targets = [], []
    all_conf_preds, all_conf_targets = [], []
    bce_none = nn.BCELoss(reduction='none')
    for images, corr_targets, conf_targets in loader:
        images = images.to(DEVICE)
        corr_targets = corr_targets.to(DEVICE)
        conf_targets = conf_targets.to(DEVICE)
        corr_pred, conf_pred = model(images)
        bce_per_elem = bce_none(corr_pred, corr_targets)
        w = bce_weights.to(DEVICE)
        weighted_bce = (bce_per_elem * w).mean()
        confidence_loss = mse_criterion(conf_pred, conf_targets)
        total_loss = weighted_bce + CONFIDENCE_LOSS_WEIGHT * confidence_loss
        running_loss += total_loss.item() * images.size(0)
        total += images.size(0)
        all_corr_preds.append(corr_pred.cpu())
        all_corr_targets.append(corr_targets.cpu())
        all_conf_preds.append(conf_pred.cpu())
        all_conf_targets.append(conf_targets.cpu())
    val_loss = running_loss / total
    y_corr_pred = torch.cat(all_corr_preds, dim=0).numpy()
    y_corr_true = torch.cat(all_corr_targets, dim=0).numpy()
    y_conf_pred = torch.cat(all_conf_preds, dim=0).numpy()
    y_conf_true = torch.cat(all_conf_targets, dim=0).numpy()
    accs, aurocs = per_tier_metrics(y_corr_true, y_corr_pred)
    avg_auroc = float(np.nanmean(aurocs))
    # Confidence MSE per tier and overall
    conf_mse_per_tier = ((y_conf_pred - y_conf_true) ** 2).mean(axis=0)
    conf_mse_overall = float(((y_conf_pred - y_conf_true) ** 2).mean())
    return val_loss, accs, aurocs, avg_auroc, y_corr_pred, y_corr_true, conf_mse_per_tier, conf_mse_overall


def _report(test_accs, test_aurocs, test_avg_auroc):
    resource_norm = load_resource_norm()
    costs = np.array([resource_norm[t] for t in TIER_NAMES])

    print(f"\n{'='*60}")
    print(f"iWildCam-tier CorrectnessGateNetV3 -- Test AUROC")
    print(f"{'='*60}")
    for i, tier in enumerate(TIER_NAMES):
        print(f"  {tier:9s}: {test_aurocs[i]:.4f}")
    print(f"  avg: {test_avg_auroc:.4f}")

    print(f"\nV2b baseline, same utility_labels.csv:")
    for tier in TIER_NAMES:
        print(f"  {tier:9s}: {V2B_BASELINE[tier]:.4f}")
    print(f"  avg: {V2B_BASELINE['avg']:.4f}")

    print(f"\nDelta V3 - V2b:")
    for tier in TIER_NAMES:
        delta = test_aurocs[TIER_NAMES.index(tier)] - V2B_BASELINE[tier]
        print(f"  {tier:9s}: {delta:+.4f} {'(improved)' if delta > 0 else '(regressed)' if delta < 0 else ''}")
    delta_avg = test_avg_auroc - V2B_BASELINE["avg"]
    print(f"  avg: {delta_avg:+.4f} {'(improved)' if delta_avg > 0 else '(regressed)' if delta_avg < 0 else ''}")
    # Highlight Heavy specifically
    heavy_delta = test_aurocs[TIER_NAMES.index("heavy")] - V2B_BASELINE["heavy"]
    print(f"\nHeavy specifically: V3 {test_aurocs[2]:.4f} vs V2b {V2B_BASELINE['heavy']:.4f} delta {heavy_delta:+.4f}")

    print(f"\nLatency: V1=~0.5ms(from-scratch)  V2=1.620ms  V2b=1.868ms  V3=~1.9ms (shared trunk + last block unfrozen)")
    # Params will be printed in main


def main():
    print(f"Using device: {DEVICE}")
    df = pd.read_csv(LABELS_PATH)
    df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)

    missing = [c for c in TARGET_COLUMNS + CONFIDENCE_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing target columns: {missing}")
    print(f"Loaded {len(df)} labeled samples")

    # Compute BCE weights from full df positive rates before split (or train split - using full for stability)
    # Spec says from actual label imbalance - using whole df before split
    bce_weights = compute_bce_weights(df)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["best_tier"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["best_tier"], random_state=SEED
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    train_loader = DataLoader(CorrectnessGateV3Dataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(CorrectnessGateV3Dataset(val_df, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(CorrectnessGateV3Dataset(test_df, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = CorrectnessGateNetV3(num_tiers=3).to(DEVICE)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CorrectnessGateNetV3 total params: {n_total:,}  trainable: {n_trainable:,}")
    # Breakout: frozen backbone portion vs unfrozen last block vs heads
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  trainable: {name} shape {tuple(param.shape)}")

    mse_criterion = nn.MSELoss()
    best_ckpt_path = CHECKPOINT_DIR / "correctness_gate_v3_best.pt"

    if best_ckpt_path.exists():
        print(f"\nCheckpoint exists: {best_ckpt_path} -- loading and evaluating only.\n")
        ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        _, test_accs, test_aurocs, test_avg_auroc, _, _, _, _ = evaluate(model, test_loader, bce_weights, mse_criterion)
        _report(test_accs, test_aurocs, test_avg_auroc)
        return

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Optimizer: {sum(p.numel() for p in trainable_params):,} trainable params")
    print(f"CONFIDENCE_LOSS_WEIGHT = {CONFIDENCE_LOSS_WEIGHT}")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    mlflow.set_experiment("inferpilot-model-tiers")
    history = {"train_loss": [], "val_loss": [], "val_auroc": [[] for _ in range(3)], "val_conf_mse": []}
    best_avg_auroc = 0.0

    with mlflow.start_run(run_name="correctness_gate_v3"):
        mlflow.log_params({
            "architecture": "CorrectnessGateNetV3",
            "backbone": "MobileNetV3-Small features[:5] last block unfrozen, shared 40->180->90 + dual heads 90->3 sigmoid",
            "total_params": n_total, "trainable_params": n_trainable,
            "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
            "confidence_loss_weight": CONFIDENCE_LOSS_WEIGHT,
            "bce_weights": bce_weights.tolist(),
        })

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, bce_weights, mse_criterion, optimizer)
            val_loss, accs, aurocs, avg_auroc, _, _, conf_mse_per_tier, conf_mse_overall = evaluate(model, val_loader, bce_weights, mse_criterion)
            scheduler.step()
            epoch_time = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            for t in range(3):
                history["val_auroc"][t].append(aurocs[t])
            history["val_conf_mse"].append(conf_mse_overall)

            mlflow.log_metrics({
                "train_loss": train_loss, "val_loss": val_loss,
                "val_fast_auroc": aurocs[0], "val_balanced_auroc": aurocs[1],
                "val_heavy_auroc": aurocs[2], "val_avg_auroc": avg_auroc,
                "val_conf_mse": conf_mse_overall,
                "val_conf_mse_fast": float(conf_mse_per_tier[0]),
                "val_conf_mse_balanced": float(conf_mse_per_tier[1]),
                "val_conf_mse_heavy": float(conf_mse_per_tier[2]),
                "epoch_time_sec": epoch_time,
            }, step=epoch)

            print(f"Epoch {epoch:2d}/{EPOCHS} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} avg_auroc {avg_auroc:.4f} | "
                  f"auroc(f/b/h) {aurocs[0]:.3f}/{aurocs[1]:.3f}/{aurocs[2]:.3f} | "
                  f"conf_mse {conf_mse_overall:.4f} | {epoch_time:.1f}s")

            if avg_auroc > best_avg_auroc:
                best_avg_auroc = avg_auroc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_names": TARGET_COLUMNS,
                    "val_auroc": aurocs, "val_avg_auroc": avg_auroc, "epoch": epoch,
                    "val_conf_mse": conf_mse_overall,
                    "bce_weights": bce_weights.tolist(),
                }, best_ckpt_path)
                print(f"  -> new best val_avg_auroc {avg_auroc:.4f}, checkpoint saved")

        best_ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        model.load_state_dict(best_ckpt["model_state_dict"])
        _, test_accs, test_aurocs, test_avg_auroc, _, _, _, _ = evaluate(model, test_loader, bce_weights, mse_criterion)

        mlflow.log_metrics({
            "test_avg_auroc": test_avg_auroc,
            "test_fast_auroc": test_aurocs[0], "test_balanced_auroc": test_aurocs[1],
            "test_heavy_auroc": test_aurocs[2],
        })

        print(f"\n{'='*60}\nFINAL TEST RESULTS (V3)\n{'='*60}")
        for i, tier in enumerate(TIER_NAMES):
            print(f"  {tier:9s}: acc {test_accs[i]:.4f}  auroc {test_aurocs[i]:.4f}")

        _report(test_accs, test_aurocs, test_avg_auroc)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs_range = range(1, EPOCHS + 1)
        axes[0].plot(epochs_range, history["train_loss"], label="Train")
        axes[0].plot(epochs_range, history["val_loss"], label="Val")
        axes[0].set_title("Loss (V3)"); axes[0].legend()
        for t, tier in enumerate(TIER_NAMES):
            axes[1].plot(epochs_range, history["val_auroc"][t], label=f"Val {tier} AUROC")
        axes[1].set_title("Per-Tier AUROC (V3)"); axes[1].legend()
        curve_path = CHECKPOINT_DIR / "correctness_gate_v3_curves.png"
        fig.savefig(curve_path); plt.close(fig)
        mlflow.log_artifact(str(curve_path))
        mlflow.log_artifact(str(best_ckpt_path))

    print(f"\nDone. Best V3 checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
