"""
Read-only diagnostic: evaluates ONLY the confidence_head of
CorrectnessGateNetV3 (training/checkpoints/correctness_gate_v3_best.pt)
on the identical held-out test set (n=632) used in train_correctness_gate_v3.py.

No training, no checkpoint modification.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr, f_oneway

from models.correctness_gate import CorrectnessGateNetV3

# Reuse canonical constants / transforms from train_correctness_gate
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_correctness_gate import (
    BASE_DIR, LABELS_PATH, CHECKPOINT_DIR,
    TARGET_COLUMNS, TIER_NAMES, IMAGENET_MEAN, IMAGENET_STD,
    eval_transform,
)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

CONFIDENCE_COLUMNS = ["fast_confidence", "balanced_confidence", "heavy_confidence"]

# True heavy_confidence baseline means per best_tier group (from prior session finding)
TRUE_HEAVY_MEANS = {"fast": 0.766, "balanced": 0.800, "heavy": 0.868}
TRUE_DELTA = TRUE_HEAVY_MEANS["heavy"] - TRUE_HEAVY_MEANS["fast"]  # 0.102


class CorrectnessGateV3Dataset(Dataset):
    """Returns image + correctness [3] + confidence [3]. Mirrors train_correctness_gate_v3.py:56"""
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


def main():
    print(f"Using device: {DEVICE}")
    print(f"LABELS_PATH: {LABELS_PATH}")
    print(f"CHECKPOINT: {CHECKPOINT_DIR / 'correctness_gate_v3_best.pt'}")

    df = pd.read_csv(LABELS_PATH)
    df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)
    print(f"Loaded {len(df)} labeled samples")
    print(f"Label distribution:\n{df['best_tier'].value_counts()}\n")

    missing = [c for c in TARGET_COLUMNS + CONFIDENCE_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    # Identical split to train_correctness_gate_v3.py:214-219
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["best_tier"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["best_tier"], random_state=SEED
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print(f"Test best_tier counts:\n{test_df['best_tier'].value_counts()}\n")

    test_loader = DataLoader(
        CorrectnessGateV3Dataset(test_df, eval_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    model = CorrectnessGateNetV3(num_tiers=3).to(DEVICE)
    ckpt_path = CHECKPOINT_DIR / "correctness_gate_v3_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded V3 checkpoint epoch {ckpt.get('epoch')} val_avg_auroc {ckpt.get('val_avg_auroc'):.4f}")

    # Inference
    all_pred_conf = []
    all_true_conf = []
    all_best_tier = []

    # Need mapping from test_df index order to best_tier labels in loader order.
    # Dataset is built on test_df reset_index, so loader order matches test_df order.
    best_tier_list = test_df["best_tier"].tolist()

    with torch.no_grad():
        start_idx = 0
        for images, _, conf_targets in test_loader:
            images = images.to(DEVICE)
            _, conf_pred = model(images)  # each [B,3] Sigmoid
            batch_size = images.size(0)
            all_pred_conf.append(conf_pred.cpu().numpy())
            all_true_conf.append(conf_targets.numpy())
            # Slice corresponding best_tier labels
            all_best_tier.extend(best_tier_list[start_idx:start_idx + batch_size])
            start_idx += batch_size

    pred_conf = np.concatenate(all_pred_conf, axis=0)  # [N,3]
    true_conf = np.concatenate(all_true_conf, axis=0)  # [N,3]
    best_tier_arr = np.array(all_best_tier)
    N = len(pred_conf)
    print(f"\nCollected test predictions: N={N} pred_conf shape {pred_conf.shape} true_conf shape {true_conf.shape}")

    # §3 Per-tier table
    print("\n" + "="*70)
    print("Per-tier confidence_head evaluation (predicted vs true confidence)")
    print("="*70)
    header = f"{'Tier':<10} {'MSE':>8} {'Pearson r':>10} {'p':>10} {'Spearman rho':>13} {'p':>10} {'pred mean':>10} {'pred std':>10}"
    print(header)
    print("-"*70)

    # Store for verdict
    pearson_rs = {}
    spearman_rhos = {}
    pred_stds = {}
    anova_needed = True

    for i, tier in enumerate(TIER_NAMES):
        pred = pred_conf[:, i]
        true = true_conf[:, i]
        mse = float(np.mean((pred - true) ** 2))
        # Pearson
        try:
            pr, pp = pearsonr(pred, true)
        except Exception as e:
            pr, pp = float("nan"), float("nan")
        # Spearman
        try:
            sr, sp = spearmanr(pred, true)
        except Exception as e:
            sr, sp = float("nan"), float("nan")
        pmean = float(pred.mean())
        pstd = float(pred.std())
        pearson_rs[tier] = pr
        spearman_rhos[tier] = sr
        pred_stds[tier] = pstd

        collapsed_flag = "  WARNING: collapsed" if pstd < 0.02 else ""
        print(f"{tier:<10} {mse:8.4f} {pr:10.4f} {pp:10.2e} {sr:13.4f} {sp:10.2e} {pmean:10.4f} {pstd:10.4f}{collapsed_flag}")
        if pstd < 0.02:
            print(f"  -> WARNING: predicted confidence may have collapsed toward the mean {pmean:.3f}")

    # §4 Separation check on PREDICTED heavy_confidence
    print("\n" + "="*70)
    print("Separation check: PREDICTED heavy_confidence grouped by ground-truth optimal tier")
    print("="*70)
    idx_heavy = TIER_NAMES.index("heavy")
    pred_heavy = pred_conf[:, idx_heavy]

    groups = {}
    for tier in TIER_NAMES:
        mask = best_tier_arr == tier
        groups[tier] = pred_heavy[mask]

    for tier in TIER_NAMES:
        g = groups[tier]
        print(f"  {tier:9s} (n={len(g):3d}): mean predicted heavy_conf = {g.mean():.4f}  std {g.std():.4f}  median {np.median(g):.4f}")

    # True baseline for comparison
    print(f"\n  True heavy_confidence baseline (prior finding):")
    for tier in TIER_NAMES:
        print(f"    {tier:9s}: {TRUE_HEAVY_MEANS[tier]:.3f}")
    print(f"  True delta heavy - fast = {TRUE_DELTA:.3f}")

    # ANOVA on predicted
    try:
        F, p_anova = f_oneway(groups["fast"], groups["balanced"], groups["heavy"])
    except Exception as e:
        F, p_anova = float("nan"), float("nan")
    print(f"\n  ANOVA on predicted heavy_confidence across 3 groups:")
    print(f"    F = {F:.4f}, p = {p_anova:.4e}")

    # Monotonic check (addition per approval)
    mean_fast = groups["fast"].mean()
    mean_bal = groups["balanced"].mean()
    mean_heavy = groups["heavy"].mean()
    monotonic = (mean_fast < mean_bal < mean_heavy)
    # Provide explicit verdict line with values if not monotonic
    if monotonic:
        print(f"  Monotonic (fast < balanced < heavy): YES")
    else:
        # Find which inequality fails for explicit message
        if not (mean_fast < mean_bal):
            print(f"  Monotonic (fast < balanced < heavy): NO (e.g. fast {mean_fast:.4f} >= balanced {mean_bal:.4f})")
        elif not (mean_bal < mean_heavy):
            print(f"  Monotonic (fast < balanced < heavy): NO (e.g. balanced {mean_bal:.4f} >= heavy {mean_heavy:.4f})")
        else:
            print(f"  Monotonic (fast < balanced < heavy): NO")

    # Comparison verdict line for separation
    pred_delta = mean_heavy - mean_fast
    if TRUE_DELTA != 0:
        preserved_pct = (pred_delta / TRUE_DELTA) * 100
    else:
        preserved_pct = float("nan")
    if p_anova < 0.05 and monotonic:
        sep_verdict = f"predicted heavy_confidence preserves {preserved_pct:.1f}% of true separation (Δpred {pred_delta:.4f} vs Δtrue {TRUE_DELTA:.3f}), significant (p={p_anova:.2e})"
    elif p_anova < 0.05 and not monotonic:
        sep_verdict = f"predicted heavy_confidence shows significant ANOVA (p={p_anova:.2e}) but NOT monotonic (means {mean_fast:.3f}/{mean_bal:.3f}/{mean_heavy:.3f}), unlike true monotonic increase"
    elif p_anova >= 0.05:
        sep_verdict = f"predicted heavy_confidence shows NO significant separation (p={p_anova:.2e}), unlike true signal (which showed real separation 0.766/0.800/0.868, p<0.001)"
    else:
        sep_verdict = f"ANOVA inconclusive (p={p_anova:.2e})"
    print(f"  -> {sep_verdict}")

    # §5 VERDICT
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    # Criteria for (a) usable: |r|>=0.4 and rho>=0.4 on >=2 tiers AND ANOVA p<0.05 AND monotonic
    usable_tiers = sum(1 for t in TIER_NAMES if abs(pearson_rs[t]) >= 0.4 and abs(spearman_rhos[t]) >= 0.4)
    collapsed_any = any(v < 0.02 for v in pred_stds.values())
    anova_sig = p_anova < 0.05

    if usable_tiers >= 2 and anova_sig and monotonic and not collapsed_any:
        print("(a) Confidence head predictions are well-calibrated/correlated enough to be a usable standalone difficulty signal.")
        print(f"    Evidence: {usable_tiers}/3 tiers have |r|>=0.4 & rho>=0.4, ANOVA p={p_anova:.2e} significant with monotonic means, no collapse (stds >0.02).")
    else:
        reasons = []
        if usable_tiers < 2:
            reasons.append(f"only {usable_tiers}/3 tiers meet |r|>=0.4 & rho>=0.4")
        if not anova_sig:
            reasons.append(f"ANOVA p={p_anova:.2e} not significant")
        if not monotonic:
            reasons.append("means not monotonically increasing fast<balanced<heavy")
        if collapsed_any:
            reasons.append("predicted std <0.02 indicates collapse toward mean on at least one tier")
        print("(b) Confidence head predictions are too noisy/collapsed to be useful beyond auxiliary training-time regularization.")
        print(f"    Evidence: {'; '.join(reasons)}.")
        print(f"    While V3's correctness AUROC improved (+0.02 heavy), confidence predictions themselves show limited correlation/monotonic separation.")


if __name__ == "__main__":
    main()
