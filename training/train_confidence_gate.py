"""
training/train_confidence_gate.py

STAGE 1: trains a small CNN to predict Fast's confidence directly from
image pixels, without calling Fast at inference time.

Reuses router/utility_labels.csv -- same images, but regresses toward the
continuous fast_confidence column instead of classifying the discrete
best_tier label (which 3 prior attempts showed is too entangled with
Fast's idiosyncratic per-image correctness to learn from pixels alone).

Confidence is heavily saturated (median 0.916, 90th pct 0.996) -- uses
inverse-density sample weighting (same philosophy as the project's
class-weighted sklearn router and the earlier weighted-CrossEntropy
GateNet attempt) so the model doesn't just learn to predict "~0.9" for
everything.

Evaluation reports raw weighted/unweighted MSE/MAE, AND (the metric that
actually matters) decision-consistency balanced accuracy: applying the
SAME rule_based thresholds (low=0.50, high=0.66) to PREDICTED confidence
instead of real fast_confidence, comparing the resulting routed tier
against the true best_tier label. This produces a number directly
comparable to the three prior attempts (Path A ~0.40, GateNet ~0.395,
current sklearn router ~0.56).

Run: python training/train_confidence_gate.py
"""
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

from models.confidence_gate import ConfidenceGateNet

# ---------------- Config ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
LABELS_PATH = BASE_DIR / "router" / "utility_labels.csv"
CHECKPOINT_DIR = BASE_DIR / "training" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MLFLOW_EXPERIMENT = "inferpilot-model-tiers"

# Rule-based thresholds, reused for decision-consistency evaluation
RULE_LOW_THRESH = 0.50
RULE_HIGH_THRESH = 0.66

# Weighting: bin confidence into N_BINS, weight inversely to bin frequency,
# capped to avoid instability from near-empty bins near the sparse tail
N_BINS = 20
MAX_WEIGHT = 15.0

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


def compute_sample_weights(confidences: np.ndarray, n_bins=N_BINS, max_weight=MAX_WEIGHT):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidences, bins) - 1, 0, n_bins - 1)
    bin_counts = np.bincount(bin_idx, minlength=n_bins)

    weights = np.zeros_like(confidences, dtype=np.float32)
    for i in range(n_bins):
        mask = bin_idx == i
        if bin_counts[i] > 0:
            weights[mask] = 1.0 / bin_counts[i]

    weights = weights / weights.mean()  # normalize so avg weight = 1
    weights = np.clip(weights, None, max_weight)
    return weights


class ConfidenceDataset(Dataset):
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
        target = float(row["fast_confidence"])
        weight = float(row["sample_weight"])
        best_tier = row["best_tier"]
        return image, target, weight, best_tier


def rule_based_tier(confidence: float) -> str:
    if confidence >= RULE_HIGH_THRESH:
        return "fast"
    elif confidence >= RULE_LOW_THRESH:
        return "balanced"
    else:
        return "heavy"


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_weighted_loss, total_weight = 0.0, 0.0
    for images, targets, weights, _ in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE).float()
        weights = weights.to(DEVICE).float()

        optimizer.zero_grad()
        preds = model(images)
        per_sample_loss = (preds - targets) ** 2
        loss = (per_sample_loss * weights).sum() / weights.sum()
        loss.backward()
        optimizer.step()

        total_weighted_loss += loss.item() * weights.sum().item()
        total_weight += weights.sum().item()

    return total_weighted_loss / total_weight


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_targets, all_weights, all_true_tiers = [], [], [], []

    for images, targets, weights, best_tiers in loader:
        images = images.to(DEVICE)
        preds = model(images).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())
        all_weights.extend(weights.tolist())
        all_true_tiers.extend(best_tiers)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_weights = np.array(all_weights)

    mse = np.mean((all_preds - all_targets) ** 2)
    weighted_mse = np.sum(all_weights * (all_preds - all_targets) ** 2) / all_weights.sum()
    mae = np.mean(np.abs(all_preds - all_targets))

    # ---- Decision-consistency balanced accuracy ----
    predicted_tiers = [rule_based_tier(p) for p in all_preds]
    tiers = ["fast", "balanced", "heavy"]
    per_class_correct = {t: 0 for t in tiers}
    per_class_total = {t: 0 for t in tiers}
    for pred_t, true_t in zip(predicted_tiers, all_true_tiers):
        per_class_total[true_t] += 1
        if pred_t == true_t:
            per_class_correct[true_t] += 1

    balanced_acc = np.mean([
        per_class_correct[t] / per_class_total[t] if per_class_total[t] > 0 else 0.0
        for t in tiers
    ])
    overall_acc = sum(per_class_correct.values()) / sum(per_class_total.values())

    return {
        "mse": mse, "weighted_mse": weighted_mse, "mae": mae,
        "decision_balanced_acc": balanced_acc, "decision_overall_acc": overall_acc,
        "per_class_total": per_class_total, "per_class_correct": per_class_correct,
    }


def main():
    print(f"Using device: {DEVICE}")
    df = pd.read_csv(LABELS_PATH)
    print(f"Loaded {len(df)} labeled samples")
    print(f"fast_confidence stats:\n{df['fast_confidence'].describe()}\n")

    df["sample_weight"] = compute_sample_weights(df["fast_confidence"].values)
    print(f"Sample weight range: [{df['sample_weight'].min():.3f}, {df['sample_weight'].max():.3f}]")

    # Stratify split by confidence BUCKET (not best_tier) -- ensures the
    # sparse low-confidence tail is represented in all 3 splits
    df["conf_bucket"] = pd.cut(df["fast_confidence"], bins=10, labels=False)
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["conf_bucket"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["conf_bucket"], random_state=SEED)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    train_loader = DataLoader(ConfidenceDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ConfidenceDataset(val_df, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(ConfidenceDataset(test_df, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = ConfidenceGateNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ConfidenceGateNet params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    history = {"train_loss": [], "val_mse": [], "val_decision_balanced_acc": []}
    best_decision_balanced_acc = 0.0
    best_ckpt_path = CHECKPOINT_DIR / "confidence_gate_best.pt"

    with mlflow.start_run(run_name="confidence_gate"):
        mlflow.log_params({
            "architecture": "ConfidenceGateNet", "num_params": n_params, "epochs": EPOCHS,
            "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "n_bins": N_BINS, "max_weight": MAX_WEIGHT,
            "rule_low_thresh": RULE_LOW_THRESH, "rule_high_thresh": RULE_HIGH_THRESH,
        })

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer)
            val_metrics = evaluate(model, val_loader)
            scheduler.step()
            epoch_time = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_mse"].append(val_metrics["mse"])
            history["val_decision_balanced_acc"].append(val_metrics["decision_balanced_acc"])

            mlflow.log_metrics({
                "train_weighted_loss": train_loss,
                "val_mse": val_metrics["mse"],
                "val_weighted_mse": val_metrics["weighted_mse"],
                "val_mae": val_metrics["mae"],
                "val_decision_balanced_acc": val_metrics["decision_balanced_acc"],
                "val_decision_overall_acc": val_metrics["decision_overall_acc"],
                "epoch_time_sec": epoch_time,
            }, step=epoch)

            print(f"Epoch {epoch:2d}/{EPOCHS} | train_wloss {train_loss:.4f} | "
                  f"val_mse {val_metrics['mse']:.4f} val_mae {val_metrics['mae']:.4f} | "
                  f"val_decision_bal_acc {val_metrics['decision_balanced_acc']:.4f} | {epoch_time:.1f}s")

            if val_metrics["decision_balanced_acc"] > best_decision_balanced_acc:
                best_decision_balanced_acc = val_metrics["decision_balanced_acc"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "val_decision_balanced_acc": best_decision_balanced_acc,
                    "val_mse": val_metrics["mse"],
                    "epoch": epoch,
                }, best_ckpt_path)
                print(f"  -> new best val_decision_balanced_acc {best_decision_balanced_acc:.4f}, checkpoint saved")

        # ---- Final test evaluation ----
        best_ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_metrics = evaluate(model, test_loader)

        mlflow.log_metrics({
            "test_mse": test_metrics["mse"],
            "test_mae": test_metrics["mae"],
            "test_decision_balanced_acc": test_metrics["decision_balanced_acc"],
            "test_decision_overall_acc": test_metrics["decision_overall_acc"],
        })

        print(f"\n{'='*60}\nFINAL TEST RESULTS\n{'='*60}")
        print(f"MSE: {test_metrics['mse']:.4f}  MAE: {test_metrics['mae']:.4f}")
        print(f"Decision-consistency overall accuracy:  {test_metrics['decision_overall_acc']:.4f}")
        print(f"Decision-consistency BALANCED accuracy: {test_metrics['decision_balanced_acc']:.4f}")
        print(f"\nPer-class breakdown (true best_tier -> correctly predicted):")
        for t in ["fast", "balanced", "heavy"]:
            c = test_metrics["per_class_correct"][t]
            n = test_metrics["per_class_total"][t]
            print(f"  {t:10s}: {c}/{n} ({100*c/n if n else 0:.1f}%)")

        print(f"\nComparison to prior attempts:")
        print(f"  Path A (hand-crafted features, no fast_confidence): ~0.40 balanced acc")
        print(f"  GateNet (raw pixels, discrete classification):      ~0.395 balanced acc")
        print(f"  Current sklearn router (WITH real fast_confidence): ~0.56 balanced acc")
        print(f"  ConfidenceGateNet (THIS, predicted confidence):     {test_metrics['decision_balanced_acc']:.4f} balanced acc")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs_range = range(1, EPOCHS + 1)
        axes[0].plot(epochs_range, history["train_loss"], label="Train (weighted)")
        axes[0].plot(epochs_range, history["val_mse"], label="Val MSE")
        axes[0].set_title("Loss"); axes[0].legend()
        axes[1].plot(epochs_range, history["val_decision_balanced_acc"], label="Val Decision Balanced Acc")
        axes[1].axhline(0.40, color="gray", linestyle="--", label="Prior attempts ceiling (~0.40)")
        axes[1].set_title("Decision-Consistency Balanced Accuracy"); axes[1].legend()
        curve_path = CHECKPOINT_DIR / "confidence_gate_curves.png"
        fig.savefig(curve_path)
        plt.close(fig)
        mlflow.log_artifact(str(curve_path))
        mlflow.log_artifact(str(best_ckpt_path))

    print(f"\nDone. Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()