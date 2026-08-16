"""
training/train_correctness_gate.py

Trains a small CNN that predicts P(correct) independently for each tier
(fast, balanced, heavy) directly from image pixels, WITHOUT calling any
tier model at inference time.

Unlike GateNet (discrete 3-way classification over a single best_tier
label) or ConfidenceGateNet (regression toward Fast's confidence scalar),
this uses each tier's own {tier}_correct ground-truth column from
generate_utility_labels.py's output (router/utility_labels.csv), so the
model learns per-tier correctness directly. Three independent sigmoid
outputs -- NOT a softmax; an image can plausibly be correct for all three
tiers, or none, so the values must not be forced to sum to 1.

Structure mirrors training/train_gate.py: same config constants, same
stratified 70/15/15 split (seed=42), same transforms, same MLflow
experiment. Changes: BCELoss over all 3 heads combined, per-tier
val accuracy + AUROC every epoch, best checkpoint by average AUROC, and a
final routing-utility lambda sweep (0.05-0.95) with balanced routing accuracy.

Run: python training/train_correctness_gate.py
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
from sklearn.metrics import roc_auc_score
import mlflow
import matplotlib.pyplot as plt

from models.correctness_gate import CorrectnessGateNet

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "router"))
from router_eval_utils import random_baseline_routing_acc

# ---------------- Config ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
LABELS_PATH = BASE_DIR / "router" / "utility_labels.csv"
BENCHMARKS_PATH = BASE_DIR / "router" / "model_benchmarks.json"
CHECKPOINT_DIR = BASE_DIR / "training" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
EPOCHS = 25
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MLFLOW_EXPERIMENT = "inferpilot-model-tiers"

# Ground-truth target columns (fixed order matches the model head's outputs)
TARGET_COLUMNS = ["fast_correct", "balanced_correct", "heavy_correct"]
TIER_NAMES = ["fast", "balanced", "heavy"]

# Routing-utility lambda at test time, matching generate_utility_labels.py
ROUTE_LAMBDA = 0.7

# Lambda sweep range (same style as router/utility_lambda_sweep.py): 0.05, ..., 0.95
LAMBDA_RANGE = [round(x * 0.05, 2) for x in range(1, 20)]
DEGENERATE_PCT = 95.0
LAMBDA_SWEEP_CSV = CHECKPOINT_DIR / "correctness_gate_lambda_sweep.csv"

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


class CorrectnessGateDataset(Dataset):
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
        targets = torch.tensor([row[c] for c in TARGET_COLUMNS], dtype=torch.float32)
        return image, targets


def route_by_correctness_utility(pred_probs, costs, lam):
    """
    Pick the tier maximizing predicted utility = P(correct) - lam * cost.

    pred_probs: [3] predicted P(correct) for (fast, balanced, heavy).
    costs:      [3] normalized per-tier costs (resource_norm from
                router/model_benchmarks.json).
    lam:        cost weight. This is a SERVING-TIME parameter, NOT baked into a
                training label -- unlike best_tier in the other two gating
                scripts, lam can be tuned at deployment without retraining.
    """
    return int(np.argmax(np.asarray(pred_probs) - lam * np.asarray(costs)))


def load_resource_norm():
    with open(BENCHMARKS_PATH) as f:
        data = json.load(f)
    benchmarks = {entry["tier"]: entry for entry in data}
    max_gpu_mem = max(e["gpu_peak_inference_mb"] for e in data)
    max_params = max(e["num_params"] for e in data)
    return {
        tier: (
            0.7 * (e["gpu_peak_inference_mb"] / max_gpu_mem)
            + 0.3 * (e["num_params"] / max_params)
        )
        for tier, e in benchmarks.items()
    }


def sweep_routing(test_preds: np.ndarray, test_true: np.ndarray, costs: np.ndarray):
    """
    Sweep the routing-utility lambda over cached test predictions, recording
    the routed tier distribution and balanced routing accuracy per lambda.

    Predictions are computed ONCE; only the routing rule varies with lambda.
    Balanced routing accuracy = macro-average over the 3 chosen-tier classes:
    for each tier, mean(correct) among images actually routed to that tier,
    then averaged across tiers (empty tiers excluded). This is NOT overall
    accuracy, which is misleading when routing skews toward one tier.

    For each lambda the real balanced_routing_acc is paired with a random
    routing baseline matched to that lambda's tier distribution, so a high
    number driven purely by base-rate imbalance is identifiable.
    """
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
        rnd_mean, rnd_std = random_baseline_routing_acc(
            test_true, pcts[0], pcts[1], pcts[2]
        )
        rows.append({
            "lambda": lam,
            "pct_fast": round(pcts[0], 2),
            "pct_balanced": round(pcts[1], 2),
            "pct_heavy": round(pcts[2], 2),
            "balanced_routing_acc": round(bal_acc, 4),
            "random_baseline_mean": round(rnd_mean, 4),
            "random_baseline_std": round(rnd_std, 4),
        })
    return pd.DataFrame(rows)


def report_final(test_accs, test_aurocs, test_avg_auroc, test_preds, test_true, best_ckpt_path):
    resource_norm = load_resource_norm()
    costs = np.array([resource_norm[t] for t in TIER_NAMES])

    print(f"\nPer-tier test AUROC (this model; no prior attempt reported this metric):")
    for i, tier in enumerate(TIER_NAMES):
        print(f"  {tier:9s}: {test_aurocs[i]:.4f}")

    sweep_df = sweep_routing(test_preds, test_true, costs)
    print(f"\nRouting-utility lambda sweep (resource_norm costs: "
          f"fast={costs[0]:.3f} balanced={costs[1]:.3f} heavy={costs[2]:.3f}):")
    print(sweep_df.to_string(index=False))
    sweep_df.to_csv(LAMBDA_SWEEP_CSV, index=False)
    print(f"Saved sweep table to {LAMBDA_SWEEP_CSV}")

    pct_cols = ["pct_fast", "pct_balanced", "pct_heavy"]
    max_pct = sweep_df[pct_cols].max(axis=1)
    non_degenerate = sweep_df[max_pct <= DEGENERATE_PCT]

    print(f"\nNon-degenerate lambdas (no single tier receives >{DEGENERATE_PCT:.0f}% of routed images):")
    if len(non_degenerate) == 0:
        print(f"  NONE -- every swept lambda routes >{DEGENERATE_PCT:.0f}% of test images to a single tier.")
    else:
        for _, row in non_degenerate.iterrows():
            print(f"  lambda={row['lambda']:.2f}  balanced_routing_acc={row['balanced_routing_acc']:.4f}")

    best_idx = int(sweep_df["balanced_routing_acc"].idxmax())
    best_row = sweep_df.loc[best_idx]
    best_lambda = best_row["lambda"]
    best_acc = best_row["balanced_routing_acc"]
    best_degenerate = max_pct.loc[best_idx] > DEGENERATE_PCT

    print(f"\nBest balanced routing accuracy across full lambda sweep (lambda={best_lambda:.2f}): {best_acc:.4f}")

    rnd_mean = best_row["random_baseline_mean"]
    rnd_std = best_row["random_baseline_std"]
    lift = best_acc - rnd_mean
    print(f"Model balanced_routing_acc: {best_acc:.4f}  |  "
          f"Random baseline (matched proportions): {rnd_mean:.4f} +/- {rnd_std:.4f}  |  "
          f"Lift over random: +{lift:.4f}")
    if lift < 2 * rnd_std:
        print(f"WARNING: model's advantage over random routing at matched proportions is not clearly "
              f"distinguishable from noise -- do not report this as a genuine routing-skill result "
              f"without further investigation.")

    print(f"Comparison to prior attempts (all balanced/macro-averaged accuracy):")
    print(f"  GateNet (discrete pixel classification):        ~0.395")
    print(f"  ConfidenceGateNet (predicted Fast confidence):   ~0.40")
    print(f"  Cascade router (real fast_confidence, sklearn):  0.5588")
    print(f"  CorrectnessGateNet (best point on lambda sweep): {best_acc:.4f}")

    if best_degenerate:
        print(f"\nWARNING: the best balanced-routing-accuracy lambda ({best_lambda:.2f}) is DEGENERATE "
              f"({max_pct.loc[best_idx]:.1f}% of images routed to a single tier). This is effectively a "
              f"single-tier policy, so the 'best' number should NOT be reported as a meaningful routing "
              f"win without that caveat.")


def per_tier_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    accs, aurocs = [], []
    for t in range(3):
        t_true = y_true[:, t]
        t_pred = y_pred[:, t]
        accs.append(float(((t_pred > 0.5) == t_true).mean()))
        if len(np.unique(t_true)) < 2:
            aurocs.append(float("nan"))
        else:
            aurocs.append(roc_auc_score(t_true, t_pred))
    return accs, aurocs


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, total = 0.0, 0
    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
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
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
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


def main():
    print(f"Using device: {DEVICE}")
    df = pd.read_csv(LABELS_PATH)

    missing = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing required target column(s) in {LABELS_PATH}: {missing}. "
            f"Actual columns found: {list(df.columns)}"
        )

    print(f"Loaded {len(df)} labeled samples")
    print(f"Per-tier correctness (positive) rates:")
    for c in TARGET_COLUMNS:
        print(f"  {c}: {df[c].mean():.3f}")

    # Stratified 70/15/15 split, same discipline as train_gate.py
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

    model = CorrectnessGateNet(num_tiers=3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CorrectnessGateNet params: {n_params:,}")

    # BCELoss over the full [batch, 3] output vs [batch, 3] targets -- an
    # element-wise average across all three heads combined, no manual loop
    criterion = nn.BCELoss()
    best_ckpt_path = CHECKPOINT_DIR / "correctness_gate_best.pt"

    if best_ckpt_path.exists():
        print(f"\nCheckpoint already exists: {best_ckpt_path}")
        print(f"Skipping training -- loading checkpoint and running test-set evaluation only.\n")
        ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        _, test_accs, test_aurocs, test_avg_auroc, test_preds, test_true = evaluate(model, test_loader, criterion)
        report_final(test_accs, test_aurocs, test_avg_auroc, test_preds, test_true, best_ckpt_path)
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    history = {"train_loss": [], "val_loss": [], "val_auroc": [[] for _ in range(3)]}
    best_avg_auroc = 0.0

    with mlflow.start_run(run_name="correctness_gate"):
        mlflow.log_params({
            "architecture": "CorrectnessGateNet", "num_params": n_params, "epochs": EPOCHS,
            "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "targets": TARGET_COLUMNS, "route_lambda": ROUTE_LAMBDA,
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
                "val_fast_acc": accs[0], "val_balanced_acc": accs[1], "val_heavy_acc": accs[2],
                "val_fast_auroc": aurocs[0], "val_balanced_auroc": aurocs[1], "val_heavy_auroc": aurocs[2],
                "val_avg_auroc": avg_auroc,
                "epoch_time_sec": epoch_time,
            }, step=epoch)

            print(f"Epoch {epoch:2d}/{EPOCHS} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} avg_auroc {avg_auroc:.4f} | "
                  f"acc(f/b/h) {accs[0]:.3f}/{accs[1]:.3f}/{accs[2]:.3f} "
                  f"auroc(f/b/h) {aurocs[0]:.3f}/{aurocs[1]:.3f}/{aurocs[2]:.3f} | {epoch_time:.1f}s")

            if avg_auroc > best_avg_auroc:
                best_avg_auroc = avg_auroc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_names": TARGET_COLUMNS,
                    "val_auroc": aurocs,
                    "val_avg_auroc": avg_auroc,
                    "epoch": epoch,
                }, best_ckpt_path)
                print(f"  -> new best val_avg_auroc {avg_auroc:.4f}, checkpoint saved")

        # ---- Final test evaluation ----
        best_ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_loss, test_accs, test_aurocs, test_avg_auroc, test_preds, test_true = evaluate(model, test_loader, criterion)

        mlflow.log_metrics({
            "test_avg_auroc": test_avg_auroc,
            "test_fast_acc": test_accs[0], "test_balanced_acc": test_accs[1], "test_heavy_acc": test_accs[2],
            "test_fast_auroc": test_aurocs[0], "test_balanced_auroc": test_aurocs[1], "test_heavy_auroc": test_aurocs[2],
        })

        print(f"\n{'='*60}\nFINAL TEST RESULTS\n{'='*60}")
        for i, tier in enumerate(TIER_NAMES):
            print(f"  {tier:9s}: acc {test_accs[i]:.4f}  auroc {test_aurocs[i]:.4f}")

        report_final(test_accs, test_aurocs, test_avg_auroc, test_preds, test_true, best_ckpt_path)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs_range = range(1, EPOCHS + 1)
        axes[0].plot(epochs_range, history["train_loss"], label="Train")
        axes[0].plot(epochs_range, history["val_loss"], label="Val")
        axes[0].set_title("Loss"); axes[0].legend()
        for t, tier in enumerate(TIER_NAMES):
            axes[1].plot(epochs_range, history["val_auroc"][t], label=f"Val {tier} AUROC")
        axes[1].set_title("Per-Tier AUROC"); axes[1].legend()
        curve_path = CHECKPOINT_DIR / "correctness_gate_curves.png"
        fig.savefig(curve_path)
        plt.close(fig)
        mlflow.log_artifact(str(curve_path))
        mlflow.log_artifact(str(best_ckpt_path))

    print(f"\nDone. Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
