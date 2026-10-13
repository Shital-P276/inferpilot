"""
training/sweep_lambda_per_load_stage.py

Per-load-stage CORRECTNESS_GATE_LAMBDA sweep (read-only analysis -- modifies
nothing in the gateway or any checkpoint).

Question: CORRECTNESS_GATE_LAMBDA=0.20 in serving/gateway_service.py is a single
fixed value used regardless of system load. Would load-adaptive lambda
(separate values for light/heavy/burst, looked up via estimate_load_stage())
produce better accuracy/cost tradeoffs?

Method: for each load stage independently, sweep LAMBDA over a grid using the
gate's scoring rule score(tier) = P(correct|tier) - LAMBDA*cost(tier) with the
post-migration cost landscape {"fast": 0.0367, "balanced": 0.1515, "heavy": 1.0},
and evaluate each lambda by the resulting routing distribution, resulting
accuracy (from each image's ACTUAL per-tier correctness), and a combined
utility consistent with the project's utility convention.

DATA SOURCE DECISION (checked before writing this script):
  - router/concurrent_experiment_raw_results.csv (12,025 rows) was the preferred
    candidate because it carries real light/heavy/burst phase labels. REJECTED:
    it has NO per-request phase column (only strategy/image_path/true_label/
    predicted_class/correct/routed_tier/gateway_latency_ms) and NO per-tier
    correctness ground truth (only the correctness of whichever tier was
    actually routed to). A lambda sweep needs, per image, all three tiers'
    correctness plus a stage label -- this file has neither.
  - router/utility_labels.csv (4,207 rows) USED: despite the task prompt's
    assumption, it DOES have a load_stage column (light=2141/heavy=1441/
    burst=625, assigned via LOAD_STAGE_WEIGHTS at generation time) AND per-tier
    correctness ground truth (fast_correct/balanced_correct/heavy_correct).
    Caveat (flagged, not hidden): its load_stage was randomly assigned at
    label-generation time with per-(tier,stage) k6 latency sampling -- it is
    NOT live-concurrency load. The sweep therefore answers "what would the
    optimal lambda be for each stage's latency/correctness mix", which is the
    right precursor question before wiring adaptive lambda into the gateway,
    but the values must be re-validated live afterwards.

P(correct) SOURCE: gate-predicted (CorrectnessGateNetV3 correctness_head;
fallback V2b if V3 checkpoint missing), reusing the existing architecture via
import -- never reimplemented here. The routing rule itself is imported from
training/train_correctness_gate.py::route_by_correctness_utility. If gate
inference fails, falls back to oracle P (= actual 0/1 correctness) and flags
it explicitly.

Run:  source ~/venv_inferpilot/bin/activate && python training/sweep_lambda_per_load_stage.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "training"))
sys.path.insert(0, str(REPO_ROOT))

from training.models.correctness_gate import (  # noqa: E402 -- reused, not reimplemented
    CorrectnessGateNetV3,
    CorrectnessGateNetV2b,
)
from training.train_correctness_gate import (  # noqa: E402 -- existing scoring logic
    route_by_correctness_utility,
)

# ---------------- Config ----------------
LABELS_PATH = REPO_ROOT / "router" / "utility_labels.csv"
V3_CKPT = REPO_ROOT / "training" / "checkpoints" / "correctness_gate_v3_best.pt"
V2B_CKPT = REPO_ROOT / "training" / "checkpoints" / "correctness_gate_v2b_best.pt"
OUT_CSV = REPO_ROOT / "training" / "checkpoints" / "lambda_per_load_stage_sweep.csv"

# Post-migration cost landscape (serving/gateway_service.py::CORRECTNESS_GATE_COSTS)
COSTS = {"fast": 0.0367, "balanced": 0.1515, "heavy": 1.0000}
TIERS = ["fast", "balanced", "heavy"]
COST_ARR = np.array([COSTS[t] for t in TIERS])

CURRENT_FIXED_LAMBDA = 0.20
DEFAULT_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70]
# Floor-extension grid: locates the true optimum below the original 0.05 floor
# (all stages optimized to the floor in the first run). Pass --grid floor.
FLOOR_GRID = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
STAGES = ["light", "heavy", "burst"]

DEVICE = torch.device("cpu")  # gate checkpoint is tiny; CPU keeps this read-only and fast
BATCH_SIZE = 64

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Same eval-time pipeline as train_correctness_gate.py / gateway_service.py
eval_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class SweepImageDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = REPO_ROOT / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)


def load_gate():
    """Load V3 (preferred) else V2b. Returns (model, name, uses_confidence_head_ignored)."""
    if V3_CKPT.exists():
        model = CorrectnessGateNetV3(num_tiers=3)
        ckpt = torch.load(V3_CKPT, map_location=DEVICE)
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(sd)
        model.to(DEVICE)
        model.eval()
        return model, "CorrectnessGateNetV3 (correctness_head only; confidence_head ignored)"
    if V2B_CKPT.exists():
        model = CorrectnessGateNetV2b(num_tiers=3)
        ckpt = torch.load(V2B_CKPT, map_location=DEVICE)
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(sd)
        model.to(DEVICE)
        model.eval()
        return model, "CorrectnessGateNetV2b (fallback; V3 checkpoint not found)"
    return None, "NONE (no checkpoint found -- oracle fallback)"


@torch.no_grad()
def predict_gate(model, df):
    loader = DataLoader(SweepImageDataset(df, eval_transform), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)
    preds = []
    is_v3 = isinstance(model, CorrectnessGateNetV3)
    for batch in loader:
        out = model(batch.to(DEVICE))
        corr = out[0] if is_v3 else out  # V3 returns (correctness, confidence); ignore confidence
        preds.append(corr.cpu().numpy())
    return np.concatenate(preds, axis=0)


def sweep_stage(preds, true_correct, lam_grid):
    """Sweep lambda on one stage slice. Returns list of row dicts."""
    rows = []
    for lam in lam_grid:
        costs = COST_ARR
        chosen = np.array([
            route_by_correctness_utility(p, costs, lam) for p in preds
        ])
        pcts = [float((chosen == t).mean()) * 100 for t in range(3)]
        # Resulting accuracy: actual correctness of the tier each image was routed to
        acc = float(np.mean([true_correct[i, chosen[i]] for i in range(len(chosen))]))
        mean_cost = float(np.mean([COST_ARR[c] for c in chosen]))
        # Combined utility, same convention as the routing objective:
        # utility = accuracy - lambda * expected_cost
        utility = acc - lam * mean_cost
        rows.append({
            "lambda": lam,
            "pct_fast": round(pcts[0], 1),
            "pct_balanced": round(pcts[1], 1),
            "pct_heavy": round(pcts[2], 1),
            "accuracy": round(acc, 4),
            "mean_cost": round(mean_cost, 4),
            "utility": round(utility, 4),
        })
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["default", "floor"], default="default",
                    help="default: original 0.05-0.70 grid; floor: 0.00-0.10 extension")
    args = ap.parse_args()
    lam_grid = FLOOR_GRID if args.grid == "floor" else DEFAULT_GRID
    print("=" * 70)
    print("Per-load-stage CORRECTNESS_GATE_LAMBDA sweep (read-only analysis)")
    print("=" * 70)
    print(f"Costs (post-migration): {COSTS}")
    print(f"Current fixed lambda: {CURRENT_FIXED_LAMBDA}")
    print(f"Grid ({args.grid}): {lam_grid}")

    # --- Data source (decision documented in module docstring) ---
    print("\nDATA SOURCE: router/utility_labels.csv")
    print("  Why: concurrent_experiment_raw_results.csv has no per-request phase")
    print("  column and no per-tier correctness ground truth, so it cannot serve")
    print("  as a sweep source. utility_labels.csv HAS both load_stage")
    print("  (light/heavy/burst) and fast/balanced/heavy_correct per image.")
    print("  Caveat: its load_stage was randomly assigned at generation time")
    print("  (LOAD_STAGE_WEIGHTS), not observed under live concurrency -- treat")
    print("  these as per-stage optima to re-validate live, not final values.")

    df = pd.read_csv(LABELS_PATH)
    df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)
    print(f"\nLoaded {len(df)} labeled samples: "
          + ", ".join(f"{s}={int((df['load_stage'] == s).sum())}" for s in STAGES))

    # --- P(correct) source: gate-predicted, oracle fallback ---
    print("\nLoading gate for P(correct|tier) predictions...")
    model, model_name = load_gate()
    print(f"  Model: {model_name}")
    if model is not None:
        print("  Running gate inference (CPU) over 4,207 images...")
        preds_all = predict_gate(model, df)
        p_source = "gate-predicted P(correct)"
    else:
        preds_all = df[[f"{t}_correct" for t in TIERS]].to_numpy(dtype=float)
        p_source = "ORACLE fallback (actual 0/1 correctness as P -- upper bound, not deployable)"
        print(f"  WARNING: {p_source}")
    print(f"  P(correct) source: {p_source}")

    true_correct = df[[f"{t}_correct" for t in TIERS]].to_numpy(dtype=float)

    all_rows = []
    optima = {}
    for stage in STAGES:
        mask = (df["load_stage"] == stage).to_numpy()
        n = int(mask.sum())
        preds = preds_all[mask]
        true_c = true_correct[mask]
        print(f"\n{'-' * 70}\nStage: {stage} (n={n})")
        if n < 200:
            print(f"  NOTE: n={n} is small -- per-lambda estimates here are noisy;")
            print("  treat this stage's optimum as low-confidence (see sensitivity below).")
        rows = sweep_stage(preds, true_c, lam_grid)
        for r in rows:
            r["stage"] = stage
            r["n"] = n
            all_rows.append(r)
        s_df = pd.DataFrame(rows)
        print(s_df.to_string(index=False))
        best = s_df.loc[s_df["utility"].idxmax()]
        optima[stage] = (float(best["lambda"]), float(best["utility"]),
                         float(best["accuracy"]), n)
        acc_range = float(s_df["accuracy"].max() - s_df["accuracy"].min())
        print(f"  -> optimal lambda (max utility): {best['lambda']:.2f} "
              f"(utility={best['utility']:.4f}, accuracy={best['accuracy']:.4f}, "
              f"routing fast/bal/heavy={best['pct_fast']:.1f}/{best['pct_balanced']:.1f}/{best['pct_heavy']:.1f})")
        print(f"  Sensitivity: accuracy range across grid = {acc_range:.4f} "
              f"({'FLAT -- lambda barely matters here' if acc_range < 0.02 else 'SENSITIVE -- lambda choice matters'})")

    sweep_df = pd.DataFrame(all_rows)
    sweep_df = sweep_df.loc[:, ["stage", "n", "lambda", "pct_fast", "pct_balanced",
                                "pct_heavy", "accuracy", "mean_cost", "utility"]]
    out_csv = (str(OUT_CSV).replace(".csv", "_floor.csv") if args.grid == "floor"
               else OUT_CSV)
    sweep_df.to_csv(out_csv, index=False)
    print(f"\nSaved full sweep table to {out_csv}")

    # --- Comparison table ---
    print("\n" + "=" * 70)
    print("COMPARISON: per-stage optimal lambda vs current fixed value")
    print("=" * 70)
    print(f"  Current fixed CORRECTNESS_GATE_LAMBDA = {CURRENT_FIXED_LAMBDA:.2f} (all stages)")
    for stage in STAGES:
        lam, util, acc, n = optima[stage]
        print(f"  {stage:6s} (n={n:4d}): optimal lambda = {lam:.2f} "
              f"(utility={util:.4f}, accuracy={acc:.4f})")
    opt_vals = [optima[s][0] for s in STAGES]
    spread = max(opt_vals) - min(opt_vals)
    off_fixed = [abs(v - CURRENT_FIXED_LAMBDA) for v in opt_vals]
    print(f"\n  Spread across stages (max-min optimal lambda): {spread:.2f}")
    if spread < 0.06:
        print("  VERDICT: per-stage optima are all close together -- NO meaningful")
        print("  evidence for load-adaptive lambda from this data. Keep the fixed value")
        print("  (after re-validating its level via the global optimum) and do NOT wire")
        print("  adaptive lambda into the gateway on this evidence.")
    elif max(off_fixed) < 0.06:
        print("  VERDICT: optima differ slightly across stages but all sit near the")
        print("  current fixed 0.20 -- adaptive lambda would add complexity for")
        print("  negligible gain. Keep fixed lambda (re-validated level).")
    else:
        print("  VERDICT: per-stage optima differ meaningfully from each other and/or")
        print("  from 0.20 -- adaptive lambda is worth wiring into the gateway as a")
        print("  follow-up task, with live re-validation per stage.")
    print("  (Gateway + checkpoints untouched by this script -- analysis only.)")


if __name__ == "__main__":
    main()
