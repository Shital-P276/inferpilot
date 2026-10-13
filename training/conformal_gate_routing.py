"""
training/conformal_gate_routing.py

Workstream A: V3 calibration + conformal selective routing (read-only analysis).

Frozen CorrectnessGateNetV3 checkpoint; modifies nothing (no checkpoint, gateway,
or training-script changes). CPU only.

Parts:
  1. Calibration refresh (extends V1-only calibrate_correctness_gate.py to V3):
     test AUROC sanity, ECE (10 bins) raw vs Platt-scaled per tier head,
     reliability-diagram PNGs. Platt fit on VAL only, evaluated on TEST only.
  2. Split-conformal selective routing: nonconformity s = 1 - P(correct of the
     gate-chosen tier at LAMBDA=0.20); thresholds from VAL quantiles at nominal
     coverage [0.80, 0.90, 0.95]; TEST covered set routed by gate, abstained set
     escalated to Heavy. Compared against fixed-lambda gate on same partition.

Split: exact 70/15/15 stratified reproduction from train_correctness_gate_v3.py
(seed=42, stratify on best_tier).

Run: source ~/venv_inferpilot/bin/activate && python training/conformal_gate_routing.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "training"))
sys.path.insert(0, str(REPO_ROOT))

from training.models.correctness_gate import CorrectnessGateNetV3  # noqa: E402
from training.train_correctness_gate import (  # noqa: E402
    route_by_correctness_utility,
)

# ---------------- Config ----------------
LABELS_PATH = REPO_ROOT / "router" / "utility_labels.csv"
V3_CKPT = REPO_ROOT / "training" / "checkpoints" / "correctness_gate_v3_best.pt"
OUT_CSV = REPO_ROOT / "training" / "checkpoints" / "conformal_gate_routing.csv"
RELIAB_PREFIX = REPO_ROOT / "training" / "checkpoints" / "v3_reliability"

COSTS = {"fast": 0.0367, "balanced": 0.1515, "heavy": 1.0000}
TIERS = ["fast", "balanced", "heavy"]
COST_ARR = np.array([COSTS[t] for t in TIERS])
FIXED_LAMBDA = 0.20
COVERAGE_LEVELS = [0.80, 0.90, 0.95]
N_BINS = 10
SEED = 42
DEVICE = torch.device("cpu")
BATCH_SIZE = 64
TARGET_COLUMNS = ["fast_correct", "balanced_correct", "heavy_correct"]

# V3 test AUROC reported by train_correctness_gate_v3.py (sanity gate tolerance)
V3_EXPECTED_AVG_AUROC = 0.7724
AUROC_TOL = 0.05
PLATT_AUROC_EQ_TOL = 0.005

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
eval_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class ImageDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = REPO_ROOT / self.df.iloc[idx]["image_path"]
        return self.transform(Image.open(img_path).convert("RGB"))


def ece_score(y_true, y_prob, n_bins=N_BINS):
    """Equal-width binning ECE."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if b > 0 else (y_prob >= lo) & (y_prob <= hi)
        frac = mask.mean()
        if frac > 0:
            ece += frac * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def main():
    print("=" * 70)
    print("Workstream A: V3 calibration + conformal selective routing (read-only)")
    print("=" * 70)

    df = pd.read_csv(LABELS_PATH)
    df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["best_tier"], random_state=SEED)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["best_tier"], random_state=SEED)
    print(f"Split (seed=42, stratify best_tier): train={len(train_df)} "
          f"val={len(val_df)} test={len(test_df)}")

    model = CorrectnessGateNetV3(num_tiers=3)
    ckpt = torch.load(V3_CKPT, map_location=DEVICE)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(sd)
    model.to(DEVICE)
    model.eval()
    print("Loaded CorrectnessGateNetV3 (correctness_head only; confidence_head ignored)")

    @torch.no_grad()
    def infer(dframe):
        loader = DataLoader(ImageDataset(dframe, eval_transform),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        outs = []
        for batch in loader:
            corr, _ = model(batch.to(DEVICE))
            outs.append(corr.cpu().numpy())
        return np.concatenate(outs, axis=0)

    print("Running gate inference on val+test (CPU)...")
    val_pred = infer(val_df)
    test_pred = infer(test_df)
    val_true = val_df[TARGET_COLUMNS].to_numpy(dtype=float)
    test_true = test_df[TARGET_COLUMNS].to_numpy(dtype=float)

    # ---- 1. AUROC sanity + ECE raw ----
    print("\n--- Part 1: calibration refresh ---")
    raw_aurocs = [float(roc_auc_score(test_true[:, t], test_pred[:, t])) for t in range(3)]
    avg_auroc = float(np.mean(raw_aurocs))
    for t, tier in enumerate(TIERS):
        print(f"  raw test AUROC {tier:9s}: {raw_aurocs[t]:.4f}")
    print(f"  raw test avg AUROC: {avg_auroc:.4f} (expected ~{V3_EXPECTED_AVG_AUROC})")
    if abs(avg_auroc - V3_EXPECTED_AVG_AUROC) > AUROC_TOL:
        print(f"HALT: avg AUROC deviates > {AUROC_TOL} from reported V3 value -- "
              f"split or checkpoint mismatch. Fix before trusting anything below.")
        sys.exit(1)

    raw_ece = [ece_score(test_true[:, t], test_pred[:, t]) for t in range(3)]
    for t, tier in enumerate(TIERS):
        print(f"  raw test ECE    {tier:9s}: {raw_ece[t]:.4f}")

    # ---- 2. Platt scaling (fit VAL, apply TEST) ----
    platt_models = []
    test_cal = np.zeros_like(test_pred)
    for t in range(3):
        clf = LogisticRegression()
        clf.fit(val_pred[:, t].reshape(-1, 1), val_true[:, t])
        platt_models.append(clf)
        test_cal[:, t] = clf.predict_proba(test_pred[:, t].reshape(-1, 1))[:, 1]

    cal_aurocs = [float(roc_auc_score(test_true[:, t], test_cal[:, t])) for t in range(3)]
    cal_ece = [ece_score(test_true[:, t], test_cal[:, t]) for t in range(3)]
    for t, tier in enumerate(TIERS):
        drift = abs(cal_aurocs[t] - raw_aurocs[t])
        print(f"  {tier:9s}: cal AUROC={cal_aurocs[t]:.4f} (drift {drift:.5f}), "
              f"ECE {raw_ece[t]:.4f} -> {cal_ece[t]:.4f}")
        if drift > PLATT_AUROC_EQ_TOL:
            print(f"HALT: Platt drift > {PLATT_AUROC_EQ_TOL} on {tier} -- smooth monotonic "
                  f"transform cannot legitimately do this; real bug. Stopping.")
            sys.exit(1)

    # Reliability diagrams (raw vs calibrated, test)
    for t, tier in enumerate(TIERS):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", label="perfect")
        for probs, label in [(test_pred[:, t], "raw"), (test_cal[:, t], "platt")]:
            edges = np.linspace(0, 1, N_BINS + 1)
            cx, cy = [], []
            for b in range(N_BINS):
                lo, hi = edges[b], edges[b + 1]
                m = (probs > lo) & (probs <= hi) if b > 0 else (probs >= lo) & (probs <= hi)
                if m.sum() > 0:
                    cx.append(probs[m].mean())
                    cy.append(test_true[m, t].mean())
            ax.plot(cx, cy, "o-", label=label)
        ax.set_xlabel("mean predicted P(correct)")
        ax.set_ylabel("empirical accuracy")
        ax.set_title(f"V3 reliability: {tier} (test, ECE raw={raw_ece[t]:.3f} cal={cal_ece[t]:.3f})")
        ax.legend()
        out = Path(str(RELIAB_PREFIX) + f"_{tier}.png")
        fig.savefig(out)
        plt.close(fig)
        print(f"  saved {out}")

    # ---- 3. Split-conformal selective routing (RAW scores; calibrated compared too) ----
    print("\n--- Part 2: split-conformal selective routing ---")
    print(f"  Gate lambda={FIXED_LAMBDA}, costs={COSTS}; abstention -> heavy.")
    rows = []
    # Baseline: fixed-lambda gate on test (raw scores)
    base_chosen = np.array([route_by_correctness_utility(p, COST_ARR, FIXED_LAMBDA)
                            for p in test_pred])
    base_acc = float(np.mean([test_true[i, base_chosen[i]] for i in range(len(test_df))]))
    base_heavy = float((base_chosen == 2).mean() * 100)
    print(f"  BASELINE fixed-lambda gate (test): accuracy={base_acc:.4f}, heavy%={base_heavy:.1f}")
    rows.append({"probs": "raw", "coverage_nominal": 1.00, "threshold": float("nan"),
                 "emp_coverage": 1.0,
                 "selective_acc": round(base_acc, 4), "overall_acc": round(base_acc, 4),
                 "heavy_pct": round(base_heavy, 1), "n_covered": len(test_df),
                 "n_total": len(test_df)})

    val_cal = np.zeros_like(val_pred)
    for t in range(3):
        val_cal[:, t] = platt_models[t].predict_proba(
            val_pred[:, t].reshape(-1, 1))[:, 1]

    for prob_label, v_use, t_use in [("raw", val_pred, test_pred),
                                     ("calibrated", val_cal, test_cal)]:
        v_chosen = np.array([route_by_correctness_utility(p, COST_ARR, FIXED_LAMBDA)
                             for p in v_use])
        v_scores = np.array([1.0 - v_use[i, v_chosen[i]] for i in range(len(val_df))])
        t_chosen = np.array([route_by_correctness_utility(p, COST_ARR, FIXED_LAMBDA)
                             for p in t_use])
        for cov in COVERAGE_LEVELS:
            alpha = 1.0 - cov
            # Finite-sample-corrected split-conformal quantile
            k = int(np.ceil((len(v_scores) + 1) * (1.0 - alpha))) - 1
            k = min(max(k, 0), len(v_scores) - 1)
            thr = float(np.sort(v_scores)[k])
            t_scores = np.array([1.0 - t_use[i, t_chosen[i]] for i in range(len(test_df))])
            covered = t_scores <= thr
            n_cov = int(covered.sum())
            emp_cov = float(n_cov / len(test_df))
            if n_cov > 0:
                sel_acc = float(np.mean([test_true[i, t_chosen[i]]
                                         for i in np.nonzero(covered)[0]]))
                final = np.where(covered, t_chosen, 2)  # abstain -> heavy
            else:
                sel_acc = float("nan")
                final = np.full(len(test_df), 2)
            overall = float(np.mean([test_true[i, final[i]] for i in range(len(test_df))]))
            heavy_pct = float((final == 2).mean() * 100)
            flag = "" if abs(emp_cov - cov) <= 0.02 else "  <-- OUTSIDE +-2pts, flag"
            print(f"  [{prob_label}] nominal={cov:.2f} thr={thr:.4f} emp_cov={emp_cov:.3f} "
                  f"n_cov={n_cov} sel_acc={sel_acc:.4f} overall={overall:.4f} "
                  f"heavy%={heavy_pct:.1f}{flag}")
            rows.append({"probs": prob_label, "coverage_nominal": cov, "threshold": round(thr, 4),
                         "emp_coverage": round(emp_cov, 4),
                         "selective_acc": round(sel_acc, 4), "overall_acc": round(overall, 4),
                         "heavy_pct": round(heavy_pct, 1), "n_covered": n_cov,
                         "n_total": len(test_df)})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    print("\n" + "=" * 70 + "\nVERDICT (fill from numbers above; both outcomes acceptable):\n"
          "Compare each conformal row's overall_acc + heavy_pct against the BASELINE\n"
          "row. If a conformal setting beats fixed-lambda at matched heavy-rate\n"
          "(or matches accuracy at lower heavy-rate), selective routing adds value\n"
          "with guarantees. Otherwise report: coverage works but adds no routing\n"
          "value over fixed lambda -- do not force a narrative.\n" + "=" * 70)


if __name__ == "__main__":
    main()
