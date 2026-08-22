"""
Analyze-only: re-run CorrectnessGateNet's lambda sweep with latency-based
costs (avg_latency_ms ratio-to-max) instead of memory/param-based costs.
Saves to a NEW file so the existing memory-cost sweep is preserved.

Does NOT modify serving/gateway_service.py, the checkpoint, or any training
logic. Pure post-hoc analysis.
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
    evaluate, sweep_routing, LAMBDA_RANGE,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "router"))
from router_eval_utils import random_baseline_routing_acc
from models.correctness_gate import CorrectnessGateNet

OUT_CSV = BASE_DIR / "training" / "checkpoints" / "correctness_gate_lambda_sweep_latencycost.csv"
FINE_GRID_CSV = BASE_DIR / "training" / "checkpoints" / "correctness_gate_lambda_sweep_finegrid.csv"


def load_latency_costs():
    with open(BENCHMARKS_PATH) as f:
        benchmarks = json.load(f)
    bench_dict = {b["tier"]: b for b in benchmarks}
    max_lat = max(b["avg_latency_ms"] for b in benchmarks)
    return {
        tier: bench_dict[tier]["avg_latency_ms"] / max_lat
        for tier in TIER_NAMES
    }, bench_dict


def main():
    latency_costs, bench_dict = load_latency_costs()
    print("Per-tier avg_latency_ms:")
    for tier in TIER_NAMES:
        print(f"  {tier:9s}: {bench_dict[tier]['avg_latency_ms']:.3f} ms")
    print(f"\nLATENCY_COSTS (ratio to max {max(b['avg_latency_ms'] for b in bench_dict.values()):.3f} ms):")
    for tier in TIER_NAMES:
        print(f"  {tier:9s}: {latency_costs[tier]:.4f}")

    df = pd.read_csv(LABELS_PATH)
    df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["best_tier"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["best_tier"], random_state=SEED
    )
    print(f"\nSplit: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    test_loader = DataLoader(
        CorrectnessGateDataset(test_df, eval_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
    )

    ckpt_path = BASE_DIR / "training" / "checkpoints" / "correctness_gate_best.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = CorrectnessGateNet(num_tiers=3).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = nn.BCELoss()
    _, _, _, _, test_preds, test_true = evaluate(model, test_loader, criterion)
    print(f"Raw test predictions: {test_preds.shape}")

    costs = np.array([latency_costs[t] for t in TIER_NAMES])
    sweep_df = sweep_routing(test_preds, test_true, costs)

    # Augment with significance and expensive-tier usage
    sweep_df["pct_expensive"] = sweep_df["pct_balanced"] + sweep_df["pct_heavy"]
    sweep_df["lift"] = sweep_df["balanced_routing_acc"] - sweep_df["random_baseline_mean"]
    sweep_df["sigma"] = np.where(
        sweep_df["random_baseline_std"] > 0,
        sweep_df["lift"] / sweep_df["random_baseline_std"],
        0.0,
    )

    sweep_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved coarse sweep to {OUT_CSV}")

    print(f"\n--- Coarse sweep (lambda 0.05-0.95, step 0.05) with LATENCY_COSTS "
          f"(fast={costs[0]:.4f} balanced={costs[1]:.4f} heavy={costs[2]:.4f}):")
    print(sweep_df[[
        "lambda", "pct_fast", "pct_balanced", "pct_heavy", "pct_expensive",
        "balanced_routing_acc", "random_baseline_mean", "random_baseline_std", "sigma",
    ]].to_string(index=False))

    # ---- Fine-grid sweep over 0.15-0.20 in steps of 0.01 ----------------
    fine_lambdas = [round(0.15 + i * 0.01, 2) for i in range(6)]
    fine_rows = []
    for lam in fine_lambdas:
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
        fine_rows.append({
            "lambda": lam,
            "pct_fast": round(pcts[0], 2),
            "pct_balanced": round(pcts[1], 2),
            "pct_heavy": round(pcts[2], 2),
            "pct_expensive": round(pcts[1] + pcts[2], 2),
            "balanced_routing_acc": round(bal_acc, 4),
            "random_baseline_mean": round(rnd_mean, 4),
            "random_baseline_std": round(rnd_std, 4),
            "sigma": round(sigma, 4),
        })

    fine_df = pd.DataFrame(fine_rows)
    fine_df.to_csv(FINE_GRID_CSV, index=False)

    print(f"\n--- Fine-grid sweep (lambda 0.15-0.20, step 0.01):")
    print(fine_df[[
        "lambda", "pct_fast", "pct_balanced", "pct_heavy", "pct_expensive",
        "balanced_routing_acc", "random_baseline_mean", "random_baseline_std", "sigma",
    ]].to_string(index=False))
    print(f"\nSaved fine-grid sweep to {FINE_GRID_CSV}")

    # Best point: lowest pct_expensive among sigma >= 3.0
    sig3 = fine_df[fine_df["sigma"] >= 3.0]
    if len(sig3) > 0:
        best_fine = sig3.loc[sig3["pct_expensive"].idxmin()]
        print(f"\n*** BEST COST-EFFICIENT POINT (sigma >= 3.0, lowest expensive-tier usage):")
        print(f"    lambda={best_fine['lambda']:.2f}  "
              f"pct_expensive={best_fine['pct_expensive']:.1f}% "
              f"(balanced={best_fine['pct_balanced']:.1f}% + heavy={best_fine['pct_heavy']:.1f}%)  "
              f"acc={best_fine['balanced_routing_acc']:.4f}  "
              f"sigma={best_fine['sigma']:.1f}")
    else:
        print(f"\n*** No lambda in fine grid has sigma >= 3.0. Showing highest-sigma point instead:")
        best_fine = fine_df.loc[fine_df["sigma"].idxmax()]
        print(f"    lambda={best_fine['lambda']:.2f}  "
              f"pct_expensive={best_fine['pct_expensive']:.1f}%  "
              f"acc={best_fine['balanced_routing_acc']:.4f}  "
              f"sigma={best_fine['sigma']:.1f}")

    # Also show coarse-sweep best point for comparison
    best_idx = int(sweep_df["balanced_routing_acc"].idxmax())
    best = sweep_df.loc[best_idx]
    print(f"\n--- Coarse-sweep best balanced_routing_acc for reference:")
    print(f"    lambda={best['lambda']:.2f}  "
          f"pct_expensive={best['pct_expensive']:.1f}%  "
          f"acc={best['balanced_routing_acc']:.4f}  "
          f"sigma={best['sigma']:.1f}")


if __name__ == "__main__":
    main()
