"""
training/train_correctness_gate_v2d_css.py

Trains CorrectnessGateNetV2d_css: same frozen MobileNetV3-Small
features[:5] backbone as V2b/V2c, same head architecture but with
num_tiers+1 outputs (one per tier PLUS one "defer" logit), trained with
a Cost-Sensitive Softmax Cross-Entropy (CSS) loss
(Mozannar & Sontag 2020 / Narasimhan et al. 2022 formulation), a
Bayes-consistent alternative to the ad hoc softmax-utility loss used in
V2c-costaware, which was found to underperform V2b's simpler post-hoc
approach (0.9599 vs 0.9686 balanced routing accuracy).

Run: python training/train_correctness_gate_v2d_css.py
"""
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
import mlflow
import matplotlib.pyplot as plt

from models.correctness_gate import CorrectnessGateNetV2d_css

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


def css_loss(raw_scores, targets, costs_tensor, lam):
    """
    raw_scores: [batch, num_tiers+1] logits (last column = "defer to heavy"
                or "defer to most expensive tier" action).
    targets:    [batch, num_tiers] ground-truth per-tier correctness (0/1).
    costs_tensor: [num_tiers] normalized per-tier cost.

    Cost-sensitive label for each sample: the TRUE best action is whichever
    tier has correct=1 at lowest cost, OR "defer" if no cheap tier (fast) is
    correct but a more expensive tier is -- concretely:
      - if targets[fast]==1: true_label = fast (cheapest correct option)
      - elif targets[balanced]==1: true_label = balanced
      - elif targets[heavy]==1: true_label = heavy
      - else: true_label = defer (none of the tiers were correct -- this
        sample is genuinely hard, treat as a distinct case, do not force it
        into one of the three tiers)
    Loss = standard cross-entropy over the num_tiers+1 raw_scores against
    true_label, i.e. nn.CrossEntropyLoss()(raw_scores, true_label_indices).
    This is the CSS reduction from Mozannar & Sontag: cost-sensitivity comes
    from how true_label is CONSTRUCTED (cheapest correct tier wins), not
    from reweighting the loss function itself -- keep the loss itself as
    plain CrossEntropyLoss, do not add manual cost multipliers on top of it,
    that would double-count cost and is a different (worse-understood)
    formulation.
    """
    # Vectorized construction of true_label per the priority logic above,
    # fast=0, balanced=1, heavy=2, defer=3
    # Start with defer (3) for all, then overwrite in reverse priority order
    # so cheapest correct wins.
    batch_size = raw_scores.shape[0]
    true_label = torch.full((batch_size,), 3, dtype=torch.long, device=raw_scores.device)
    # targets is [batch, 3] with 0/1, columns fast, balanced, heavy
    # Use boolean masks; overwrite heavy first, then balanced, then fast
    heavy_correct = targets[:, 2] == 1
    balanced_correct = targets[:, 1] == 1
    fast_correct = targets[:, 0] == 1

    # heavy where heavy correct (will be overwritten by balanced/fast if they are also correct)
    true_label[heavy_correct] = 2
    true_label[balanced_correct] = 1
    true_label[fast_correct] = 0

    return nn.functional.cross_entropy(raw_scores, true_label), true_label


@torch.no_grad()
def evaluate_induced_decision(model, loader, costs_tensor, lam):
    """
    Returns val_loss, induced_accuracy, induced_expected_cost, chosen, targets_all.
    For V2d-css, raw_scores are [batch, 4] (fast, balanced, heavy, defer).
    Induced decision: chosen_raw = argmax(raw_scores, dim=1)  (0..3)
    For routing, defer (3) is mapped to heavy (2), the safest fallback, since
    "no cheap tier was correct" implies escalation is warranted. This mapping
    is explicit and commented why (see below).
    induced_accuracy = mean(targets[i, chosen_routed[i]] for all i)
                       but for deferred-then-mapped cases, we use heavy correctness
                       as the fallback target (since defer was mapped to heavy).
    induced_expected_cost = mean(costs[chosen_routed[i]] for all i)
    """
    model.eval()
    total_loss, total = 0.0, 0
    all_raw, all_targets = [], []
    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        raw_scores = model(images)
        loss, _ = css_loss(raw_scores, targets, costs_tensor, lam)
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        all_raw.append(raw_scores.cpu())
        all_targets.append(targets.cpu())

    val_loss = total_loss / total
    raw_all = torch.cat(all_raw, dim=0).numpy()  # [N, 4]
    targets_all = torch.cat(all_targets, dim=0).numpy()  # [N, 3]

    # Induced decision: argmax over 4 logits, then map defer (3) -> heavy (2)
    chosen_raw = np.argmax(raw_all, axis=1)  # 0..3
    # Map defer -> heavy for routing purposes (safest fallback, since "no cheap tier
    # was correct" implies escalation is warranted; heavy is the most accurate
    # tier and the natural deferral target, not a learned routing to defer itself)
    chosen = np.where(chosen_raw == 3, 2, chosen_raw)

    costs_np = np.asarray(costs_tensor.cpu())
    induced_acc = float(np.mean([targets_all[i, c] for i, c in enumerate(chosen)]))
    induced_cost = float(np.mean([costs_np[c] for c in chosen]))

    return val_loss, induced_acc, induced_cost, chosen, targets_all


def compute_balanced_routing_acc(chosen, targets_all):
    """
    Macro-averaged balanced routing accuracy: for each tier, mean(correct)
    among images actually routed to that tier, then nanmean across tiers
    with >=1 routed image. Same formula as sweep_routing() in
    train_correctness_gate.py. Note: chosen here is already mapped  defer->heavy,
    so only tiers 0..2 appear (no defer tier counted).
    """
    per_tier_acc = []
    for t in range(3):
        mask = chosen == t
        if mask.sum() > 0:
            per_tier_acc.append(float(targets_all[mask, t].mean()))
    return float(np.nanmean(per_tier_acc)) if per_tier_acc else float("nan")


def compute_tier_pcts(chosen):
    """Tier distribution as percentages (after defer->heavy mapping, so 3 tiers)."""
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
    """Print full diagnostic: tier distribution, balanced acc, random baseline, verdict, three-way comparison."""
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

    # Three-way comparison: V2b post-hoc AND V2c-costaware at same LAMBDA_TRAIN
    sweep_path = V2B_FINESWEEP_CSV if V2B_FINESWEEP_CSV.exists() else V1_SWEEP_CSV
    print(f"\n{'='*60}\nTHREE-WAY COMPARISON AT SAME LAMBDA (V2b / V2c / V2d-css)\n{'='*60}")
    print("NOTE: V2d-css and V2c-costaware outputs are NOT AUROC-comparable to")
    print("V1/V2/V2b BCE models. Comparison metric is induced routing decision")
    print("quality only. CSS is Bayes-consistent via cheapest-correct label construction.")
    print(f"LAMBDA_TRAIN (from V2b sweep) = {lam_train:.4f}")

    # Load V2b sweep closest row
    v2b_row = None
    if sweep_path.exists():
        sweep_df = pd.read_csv(sweep_path)
        sweep_df["lambda_diff"] = (sweep_df["lambda"] - lam_train).abs()
        closest_idx = int(sweep_df["lambda_diff"].idxmin())
        v2b_row = sweep_df.loc[closest_idx]
        print(f"\n  V2d-css  induced_balanced_accuracy : {bal_acc:.4f}")
        print(f"  V2b      post-hoc balanced_routing_acc : {v2b_row['balanced_routing_acc']:.4f}")
        print(f"    (V2b at lambda={float(v2b_row['lambda']):.4f}, "
              f"pct_fast/bal/heavy={v2b_row['pct_fast']:.1f}%/"
              f"{v2b_row['pct_balanced']:.1f}%/{v2b_row['pct_heavy']:.1f}%)")

    # Load V2c checkpoint for comparison if exists
    v2c_ckpt_path = CHECKPOINT_DIR / "correctness_gate_v2c_costaware_best.pt"
    v2c_bal_acc = None
    if v2c_ckpt_path.exists():
        try:
            # Try to read MLflow or checkpoint's stored val metric; fall back to re-evaluating
            # For quick comparison, we read the checkpoint's val_balanced_routing_acc if stored,
            # and also try to load sweep comparison from V2c's own evaluation.
            # The V2c checkpoint stores val_balanced_routing_acc.
            import torch as _torch
            _ckpt = _torch.load(v2c_ckpt_path, map_location="cpu")
            # V2c stores val_balanced_routing_acc
            if "val_balanced_routing_acc" in _ckpt:
                v2c_bal_acc = float(_ckpt["val_balanced_routing_acc"])
                print(f"  V2c-costaware val_balanced_routing_acc   : {v2c_bal_acc:.4f} (from checkpoint {v2c_ckpt_path.name})")
            # Also try to check if we have a test balanced acc in checkpoint
            if "val_induced_accuracy" in _ckpt:
                print(f"    (V2c val_induced_accuracy {float(_ckpt['val_induced_accuracy']):.4f}, cost {float(_ckpt.get('val_induced_expected_cost',0)):.4f})")
        except Exception as e:
            print(f"  V2c-costaware checkpoint found but could not load for comparison: {e}")
    else:
        print(f"  V2c-costaware checkpoint NOT found at {v2c_ckpt_path} — skipping V2c comparison.")

    # Print deltas
    if v2b_row is not None:
        v2b_acc = float(v2b_row["balanced_routing_acc"])
        diff_v2b = bal_acc - v2b_acc
        print(f"\n  Difference (V2d-css - V2b): {diff_v2b:+.4f}")
        if diff_v2b > 0:
            print(f"    V2d-css WINS over V2b by {diff_v2b:.4f}")
        else:
            print(f"    V2b WINS over V2d-css by {-diff_v2b:.4f}")
    if v2c_bal_acc is not None:
        diff_v2c = bal_acc - v2c_bal_acc
        print(f"  Difference (V2d-css - V2c val): {diff_v2c:+.4f}")
        if diff_v2c > 0:
            print(f"    V2d-css WINS over V2c by {diff_v2c:.4f}")
        else:
            print(f"    V2c WINS over V2d-css by {-diff_v2c:.4f}")

    # Full three-way table
    print(f"\n  Full three-way (balanced_routing_acc):")
    v2b_s = f"{float(v2b_row['balanced_routing_acc']):.4f}" if v2b_row is not None else "N/A"
    v2c_s = f"{v2c_bal_acc:.4f}" if v2c_bal_acc is not None else "N/A"
    print(f"    V2b post-hoc: {v2b_s}  |  V2c-costaware val: {v2c_s}  |  V2d-css (this): {bal_acc:.4f}")
    # Also test-level if available for V2c vs V2d: we will print test after evaluation, this is val-level prelude


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

    model = CorrectnessGateNetV2d_css(num_tiers=3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nCorrectnessGateNetV2d_css params: total {n_params:,}  trainable {n_trainable:,}")
    print(f"  (num_tiers+1={4} logits, no Sigmoid — CSS defer head)")

    resource_norm = load_resource_norm()
    costs_all = np.array([resource_norm[t] for t in TIER_NAMES], dtype=np.float32)
    costs_tensor = torch.tensor(costs_all, dtype=torch.float32).to(DEVICE)
    print(f"Costs: fast={costs_all[0]:.3f} balanced={costs_all[1]:.3f} heavy={costs_all[2]:.3f}")
    print(f"LAMBDA_TRAIN = {LAMBDA_TRAIN:.4f} (from V2b sweep, used only for diagnostic lam display; CSS loss cost comes from label construction)")

    best_ckpt_path = CHECKPOINT_DIR / "correctness_gate_v2d_css_best.pt"

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
        # Also print test-level comparison to V2b and V2c (re-evaluate V2c quickly if needed is not done; we show val-level already)
        return

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    history = {"train_loss": [], "val_loss": [],
               "val_bal_acc": [], "val_induced_acc": [], "val_induced_cost": []}
    best_val_bal_acc = -1.0
    best_val_idx = -1

    with mlflow.start_run(run_name="correctness_gate_v2d_css"):
        mlflow.log_params({
            "architecture": "CorrectnessGateNetV2d_css",
            "num_params": n_params,
            "trainable_params": n_trainable,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "targets": TARGET_COLUMNS,
            "lambda_train": LAMBDA_TRAIN,
            "loss": "css_loss (CrossEntropy over 4 logits, cheapest-correct label)",
            "num_outputs": 4,
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
                loss, _ = css_loss(raw_scores, targets, costs_tensor, LAMBDA_TRAIN)
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
        axes[0].set_title("CSS Loss (V2d-css)")
        axes[0].legend()
        axes[1].plot(epochs_range, history["val_bal_acc"], label="Val Balanced Routing Acc")
        axes[1].plot(epochs_range, history["val_induced_acc"], label="Val Induced Acc (plain)")
        ax2 = axes[1].twinx()
        ax2.plot(epochs_range, history["val_induced_cost"], label="Val Induced Cost",
                 color="red", linestyle="--")
        axes[1].set_title("Routing Decision Quality (V2d-css)")
        axes[1].set_ylabel("Accuracy")
        ax2.set_ylabel("Induced Expected Cost")
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, loc="center right")
        fig.tight_layout()
        curve_path = CHECKPOINT_DIR / "correctness_gate_v2d_css_curves.png"
        fig.savefig(curve_path)
        plt.close(fig)
        mlflow.log_artifact(str(curve_path))
        mlflow.log_artifact(str(best_ckpt_path))

    print(f"\nDone. Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
