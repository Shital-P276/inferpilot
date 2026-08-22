"""
Analyze-only: re-run CorrectnessGateNetV2b's lambda sweep with latency-based
costs. Coarse grid first (0.05-0.95, step 0.05), then fine-grid around the
significance cliff (where sigma drops sharply).

Saves to new files; does not overwrite V1 sweep CSVs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_correctness_gate import (
    BASE_DIR, LABELS_PATH, BENCHMARKS_PATH, SEED, BATCH_SIZE, DEVICE,
    TARGET_COLUMNS, TIER_NAMES, CorrectnessGateDataset, eval_transform,
    evaluate,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "router"))
from router_eval_utils import random_baseline_routing_acc
from models.correctness_gate import CorrectnessGateNetV2b

COARSE_CSV = BASE_DIR / "training" / "checkpoints" / "correctness_gate_v2b_lambda_sweep_latencycost.csv"
FINE_CSV = BASE_DIR / "training" / "checkpoints" / "correctness_gate_v2b_lambda_sweep_finegrid.csv"


def load_latency_costs():
    with open(BENCHMARKS_PATH) as f:
        benchmarks = json.load(f)
    bench_dict = {b["tier"]: b for b in benchmarks}
    max_lat = max(b["avg_latency_ms"] for b in benchmarks)
    return {
        tier: bench_dict[tier]["avg_latency_ms"] / max_lat
        for tier in TIER_NAMES
    }, bench_dict


def sweep_at_lambdas(test_preds, test_true, costs, lambdas):
    rows = []
    for lam in lambdas:
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
        lift = bal_acc - rnd_mean
        sigma = lift / rnd_std if rnd_std > 0 else 0.0
        rows.append({
            "lambda": round(lam, 4),
            "pct_fast": round(pcts[0], 2),
            "pct_balanced": round(pcts[1], 2),
            "pct_heavy": round(pcts[2], 2),
            "pct_expensive": round(pcts[1] + pcts[2], 2),
            "balanced_routing_acc": round(bal_acc, 4),
            "random_baseline_mean": round(rnd_mean, 4),
            "random_baseline_std": round(rnd_std, 4),
            "sigma": round(sigma, 4),
        })
    return pd.DataFrame(rows)


def find_significance_cliff(sweep_df):
    above = sweep_df[sweep_df["sigma"] >= 3.0]
    below = sweep_df[sweep_df["sigma"] < 3.0]
    if len(above) == 0 or len(below) == 0:
        return None
    last_above = above["lambda"].max()
    first_below = below[below["lambda"] > last_above]["lambda"].min()
    return last_above, first_below


def main():
    latency_costs, bench_dict = load_latency_costs()
    print("Per-tier avg_latency_ms:")
    for tier in TIER_NAMES:
        print(f"  {tier:9s}: {bench_dict[tier]['avg_latency_ms']:.3f} ms")
    costs_arr = np.array([latency_costs[t] for t in TIER_NAMES])
    print(f"\nLATENCY_COSTS: fast={costs_arr[0]:.4f} balanced={costs_arr[1]:.4f} heavy={costs_arr[2]:.4f}")

    df = pd.read_csv(LABELS_PATH)
    df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["best_tier"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["best_tier"], random_state=SEED
    )
    print(f"Split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    test_loader = DataLoader(
        CorrectnessGateDataset(test_df, eval_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
    )

    ckpt_path = BASE_DIR / "training" / "checkpoints" / "correctness_gate_v2b_best.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = CorrectnessGateNetV2b(num_tiers=3).to(DEVICE)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    print(f"Loaded V2b checkpoint (epoch={ckpt.get('epoch', '?')})")

    criterion = nn.BCELoss()
    _, _, _, _, test_preds, test_true = evaluate(model, test_loader, criterion)
    print(f"Test predictions: {test_preds.shape}")

    # ---- Coarse sweep ----
    coarse_lambdas = [round(x * 0.05, 2) for x in range(1, 20)]
    coarse_df = sweep_at_lambdas(test_preds, test_true, costs_arr, coarse_lambdas)
    coarse_df.to_csv(COARSE_CSV, index=False)

    print(f"\n--- V2b Coarse sweep (lambda 0.05-0.95, step 0.05):")
    print(coarse_df[[
        "lambda", "pct_fast", "pct_balanced", "pct_heavy", "pct_expensive",
        "balanced_routing_acc", "random_baseline_mean", "random_baseline_std", "sigma",
    ]].to_string(index=False))
    print(f"Saved to {COARSE_CSV}")

    # ---- Find the significance cliff ----
    cliff = find_significance_cliff(coarse_df)
    if cliff is not None:
        lo, hi = cliff
        print(f"\nSignificance cliff: sigma >= 3.0 up to lambda={lo:.2f}, drops below at lambda={hi:.2f}")
        fine_start = max(0.01, lo - 0.02)
        fine_end = hi + 0.02
    else:
        print("\nNo clear cliff. Scanning around max-sigma lambda.")
        best_lam = coarse_df.loc[coarse_df["sigma"].idxmax(), "lambda"]
        fine_start = max(0.01, best_lam - 0.05)
        fine_end = min(0.99, best_lam + 0.05)

    # ---- Fine-grid sweep ----
    fine_lambdas = [round(fine_start + i * 0.005, 4)
                    for i in range(int((fine_end - fine_start) / 0.005) + 1)]
    fine_df = sweep_at_lambdas(test_preds, test_true, costs_arr, fine_lambdas)
    fine_df.to_csv(FINE_CSV, index=False)

    print(f"\n--- V2b Fine-grid sweep (lambda {fine_start:.3f}-{fine_end:.3f}, step 0.005):")
    print(fine_df[[
        "lambda", "pct_fast", "pct_balanced", "pct_heavy", "pct_expensive",
        "balanced_routing_acc", "random_baseline_mean", "random_baseline_std", "sigma",
    ]].to_string(index=False))
    print(f"Saved to {FINE_CSV}")

    # Best point: lowest pct_expensive among sigma >= 3.0
    sig3 = fine_df[fine_df["sigma"] >= 3.0]
    if len(sig3) > 0:
        best = sig3.loc[sig3["pct_expensive"].idxmin()]
        print(f"\n*** V2b BEST COST-EFFICIENT POINT (sigma >= 3.0, lowest expensive-tier usage):")
        print(f"    lambda={best['lambda']:.4f}  "
              f"pct_expensive={best['pct_expensive']:.1f}% "
              f"(balanced={best['pct_balanced']:.1f}% + heavy={best['pct_heavy']:.1f}%)  "
              f"acc={best['balanced_routing_acc']:.4f}  "
              f"sigma={best['sigma']:.1f}")
    else:
        print(f"\n*** No fine-grid point has sigma >= 3.0. Showing highest-sigma:")
        best = fine_df.loc[fine_df["sigma"].idxmax()]
        print(f"    lambda={best['lambda']:.4f}  "
              f"pct_expensive={best['pct_expensive']:.1f}%  "
              f"acc={best['balanced_routing_acc']:.4f}  "
              f"sigma={best['sigma']:.1f}")

    # Also show coarse best
    best_c = coarse_df.loc[coarse_df["balanced_routing_acc"].idxmax()]
    print(f"\n--- Coarse-sweep best balanced_routing_acc:")
    print(f"    lambda={best_c['lambda']:.2f}  "
          f"pct_expensive={best_c['pct_expensive']:.1f}%  "
          f"acc={best_c['balanced_routing_acc']:.4f}  "
          f"sigma={best_c['sigma']:.1f}")


if __name__ == "__main__":
    main()
