"""
Area Under the Deferral Curve (AUDC) analysis for CorrectnessGateNet V1 and V2b.

AUDC measures average relative cost (1 - balanced_routing_acc) across the
entire tested lambda range, normalized by range width. Lower is better.
This is a read-only analysis script -- no existing files are modified.

Outputs saved to monitoring/reports/:
  - audc_comparison.csv   (comparison table)
  - deferral_curves.png   (lambda vs accuracy plot)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS = BASE_DIR / "training" / "checkpoints"
SUMMARY = BASE_DIR / "router" / "core_experiment_summary.csv"
OUTPUT_DIR = BASE_DIR / "monitoring" / "reports"

# Reference ceiling: always_heavy accuracy (best possible, per core_experiment_summary)
ACC_REF = 0.988


def load_and_merge(coarse_csv: Path, fine_csv: Path) -> pd.DataFrame:
    """Load coarse + fine-grid sweep CSVs, merge, deduplicate, sort by lambda."""
    frames = []
    for p in [coarse_csv, fine_csv]:
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"No sweep CSVs found for {coarse_csv.name} / {fine_csv.name}")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["lambda"], keep="last")  # fine overrides coarse at same lambda
    df = df.sort_values("lambda").reset_index(drop=True)
    return df


def compute_audc(df: pd.DataFrame) -> dict:
    """Compute AUDC metrics from a lambda-sorted sweep DataFrame.

    AUDC (relative cost):
        AUDC = trapz(1 - acc, lambda) / (lambda_max - lambda_min)
        Interpretation: average fraction of accuracy lost across the sweep range.

    AUDC (normalized to ceiling):
        AUDC_norm = AUDC / (1 - ACC_REF)
        Interpretation: relative to worst possible accuracy (1 - always_heavy).
        Lower = closer to optimal.
    """
    lambdas = df["lambda"].values
    accs = df["balanced_routing_acc"].values
    rel_cost = 1.0 - accs

    # Trapezoidal integration: area under (1 - acc) curve over lambda range
    # np.trapezoid(y, x) computes ∫ y dx using trapezoidal rule (NumPy 2.x)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area = trapz(rel_cost, lambdas)
    lam_range = lambdas[-1] - lambdas[0]

    audc_raw = area / lam_range if lam_range > 0 else 0.0
    audc_norm = audc_raw / (1.0 - ACC_REF) if (1.0 - ACC_REF) > 0 else 0.0

    # Naive best: highest accuracy regardless of statistical significance
    best_idx = int(np.argmax(accs))
    best_acc = float(accs[best_idx])
    best_lambda = float(lambdas[best_idx])
    best_sigma = float(df["sigma"].iloc[best_idx]) if "sigma" in df.columns else float("nan")

    # Best defensible: highest accuracy among rows with sigma >= 3.0
    # NaN sigma (from coarse CSVs lacking the column) treated as < 3.0
    has_sigma = "sigma" in df.columns
    if has_sigma:
        defensible_mask = df["sigma"].fillna(0) >= 3.0
        defensible_df = df[defensible_mask]
    else:
        defensible_df = pd.DataFrame(columns=df.columns)

    if len(defensible_df) > 0:
        def_idx = int(defensible_df["balanced_routing_acc"].idxmax())
        def_acc = float(defensible_df.loc[def_idx, "balanced_routing_acc"])
        def_lambda = float(defensible_df.loc[def_idx, "lambda"])
        def_sigma = float(defensible_df.loc[def_idx, "sigma"]) if has_sigma else float("nan")
    else:
        def_acc, def_lambda, def_sigma = None, None, None

    return {
        "audc_raw": round(audc_raw, 6),
        "audc_normalized": round(audc_norm, 4),
        "sweep_lambda_min": float(lambdas[0]),
        "sweep_lambda_max": float(lambdas[-1]),
        "best_naive_acc": round(best_acc, 4),
        "best_naive_lambda": round(best_lambda, 4),
        "best_naive_sigma": round(best_sigma, 4) if not np.isnan(best_sigma) else None,
        "best_defensible_acc": round(def_acc, 4) if def_acc is not None else None,
        "best_defensible_lambda": round(def_lambda, 4) if def_lambda is not None else None,
        "best_defensible_sigma": round(def_sigma, 4) if def_sigma is not None and not np.isnan(def_sigma) else None,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load sweep data for both models ---
    v1_coarse = CHECKPOINTS / "correctness_gate_lambda_sweep.csv"
    v1_fine = CHECKPOINTS / "correctness_gate_lambda_sweep_finegrid.csv"
    v2b_coarse = CHECKPOINTS / "correctness_gate_v2b_lambda_sweep_latencycost.csv"
    v2b_fine = CHECKPOINTS / "correctness_gate_v2b_lambda_sweep_finegrid.csv"

    v1_df = load_and_merge(v1_coarse, v1_fine)
    v2b_df = load_and_merge(v2b_coarse, v2b_fine)

    print(f"V1 sweep:  {len(v1_df)} unique lambdas, range [{v1_df['lambda'].min():.2f}, {v1_df['lambda'].max():.2f}]")
    print(f"V2b sweep: {len(v2b_df)} unique lambdas, range [{v2b_df['lambda'].min():.2f}, {v2b_df['lambda'].max():.2f}]")

    # --- Compute AUDC for both models ---
    v1_metrics = compute_audc(v1_df)
    v2b_metrics = compute_audc(v2b_df)

    # --- Reference points: rule_based and ml_router (single operating point) ---
    summary = pd.read_csv(SUMMARY)
    rule_row = summary[summary["strategy"] == "rule_based"].iloc[0]
    ml_row = summary[summary["strategy"] == "ml_router"].iloc[0]

    rule_acc = round(float(rule_row["accuracy"]), 4)
    ml_acc = round(float(ml_row["accuracy"]), 4)

    # --- Print comparison table ---
    def fmt_naive(m):
        sig = m["best_naive_sigma"]
        sig_str = f"(sigma={sig:.1f})" if sig is not None else "(sigma=N/A)"
        return f"{m['best_naive_acc']:.4f} @ λ={m['best_naive_lambda']:.2f} {sig_str}"

    def fmt_defensible(m):
        if m["best_defensible_acc"] is None:
            return "NONE (no sigma>=3.0)"
        sig = m["best_defensible_sigma"]
        return f"{m['best_defensible_acc']:.4f} @ λ={m['best_defensible_lambda']:.2f} (sigma={sig:.1f})"

    print("\n" + "=" * 110)
    print("AUDC COMPARISON  (lower AUDC = better average cost across lambda range)")
    print("=" * 110)
    print(f"{'Model':<28} {'AUDC_raw':>10} {'AUDC_norm':>10} "
          f"{'Naive best (any sigma)':>38} {'Best defensible (sigma>=3)':>38}")
    print("-" * 110)
    print(f"{'CorrectnessGateNet V1':<28} {v1_metrics['audc_raw']:>10.6f} {v1_metrics['audc_normalized']:>10.4f} "
          f"{fmt_naive(v1_metrics):>38} {fmt_defensible(v1_metrics):>38}")
    print(f"{'CorrectnessGateNetV2b':<28} {v2b_metrics['audc_raw']:>10.6f} {v2b_metrics['audc_normalized']:>10.4f} "
          f"{fmt_naive(v2b_metrics):>38} {fmt_defensible(v2b_metrics):>38}")
    print("-" * 110)
    print(f"{'rule_based (reference)':<28} {'—':>10} {'—':>10} "
          f"{rule_acc:>8.4f} (single point){'':>18} {'—':>38}")
    print(f"{'ml_router (reference)':<28} {'—':>10} {'—':>10} "
          f"{ml_acc:>8.4f} (single point){'':>18} {'—':>38}")
    print(f"{'always_heavy (ceiling)':<28} {'—':>10} {'—':>10} "
          f"{ACC_REF:>8.4f} (reference){'':>20} {'—':>38}")
    print("=" * 110)
    print(f"\nAUDC_raw = ∫(1 - acc(λ)) dλ / (λ_max - λ_min)   [avg relative cost across sweep]")
    print(f"AUDC_norm = AUDC_raw / (1 - ACC_REF)              [normalized to ceiling {ACC_REF}]")
    print("Naive best = highest balanced_routing_acc across entire sweep (may be degenerate)")
    print("Best defensible = highest balanced_routing_acc where sigma >= 3.0 (statistically validated)")

    # --- Save comparison CSV ---
    comparison_rows = [
        {"model": "CorrectnessGateNet_V1", "audc_raw": v1_metrics["audc_raw"],
         "audc_normalized": v1_metrics["audc_normalized"],
         "best_naive_acc": v1_metrics["best_naive_acc"],
         "best_naive_lambda": v1_metrics["best_naive_lambda"],
         "best_naive_sigma": v1_metrics["best_naive_sigma"],
         "best_defensible_acc": v1_metrics["best_defensible_acc"],
         "best_defensible_lambda": v1_metrics["best_defensible_lambda"],
         "best_defensible_sigma": v1_metrics["best_defensible_sigma"],
         "sweep_lambda_min": v1_metrics["sweep_lambda_min"],
         "sweep_lambda_max": v1_metrics["sweep_lambda_max"],
         "note": "swept curve, trapezoidal AUDC"},
        {"model": "CorrectnessGateNetV2b", "audc_raw": v2b_metrics["audc_raw"],
         "audc_normalized": v2b_metrics["audc_normalized"],
         "best_naive_acc": v2b_metrics["best_naive_acc"],
         "best_naive_lambda": v2b_metrics["best_naive_lambda"],
         "best_naive_sigma": v2b_metrics["best_naive_sigma"],
         "best_defensible_acc": v2b_metrics["best_defensible_acc"],
         "best_defensible_lambda": v2b_metrics["best_defensible_lambda"],
         "best_defensible_sigma": v2b_metrics["best_defensible_sigma"],
         "sweep_lambda_min": v2b_metrics["sweep_lambda_min"],
         "sweep_lambda_max": v2b_metrics["sweep_lambda_max"],
         "note": "swept curve, trapezoidal AUDC"},
        {"model": "rule_based", "audc_raw": None, "audc_normalized": None,
         "best_naive_acc": rule_acc, "best_naive_lambda": None,
         "best_naive_sigma": None,
         "best_defensible_acc": rule_acc, "best_defensible_lambda": None,
         "best_defensible_sigma": None,
         "sweep_lambda_min": None, "sweep_lambda_max": None,
         "note": "single operating point, not swept"},
        {"model": "ml_router", "audc_raw": None, "audc_normalized": None,
         "best_naive_acc": ml_acc, "best_naive_lambda": None,
         "best_naive_sigma": None,
         "best_defensible_acc": ml_acc, "best_defensible_lambda": None,
         "best_defensible_sigma": None,
         "sweep_lambda_min": None, "sweep_lambda_max": None,
         "note": "single operating point, not swept"},
    ]
    pd.DataFrame(comparison_rows).to_csv(OUTPUT_DIR / "audc_comparison.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'audc_comparison.csv'}")

    # --- Generate deferral curve plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sweep curves
    ax.plot(v1_df["lambda"], v1_df["balanced_routing_acc"],
            marker="o", markersize=3, linewidth=1.5, label="CorrectnessGateNet V1", color="#e74c3c")
    ax.plot(v2b_df["lambda"], v2b_df["balanced_routing_acc"],
            marker="s", markersize=3, linewidth=1.5, label="CorrectnessGateNetV2b", color="#2ecc71")

    # Reference horizontal lines
    ax.axhline(y=rule_acc, color="#3498db", linestyle="--", linewidth=1.2,
               label=f"rule_based ({rule_acc:.3f})")
    ax.axhline(y=ml_acc, color="#9b59b6", linestyle=":", linewidth=1.2,
               label=f"ml_router ({ml_acc:.3f})")
    ax.axhline(y=ACC_REF, color="gray", linestyle="-.", linewidth=0.8,
               label=f"always_heavy ({ACC_REF:.3f})")

    ax.set_xlabel("λ (lambda)", fontsize=12)
    ax.set_ylabel("balanced_routing_acc", fontsize=12)
    ax.set_title("Deferral Curves — CorrectnessGateNet V1 vs V2b\n(AUDC = avg area under 1-acc, lower = better)", fontsize=13)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "deferral_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'deferral_curves.png'}")


if __name__ == "__main__":
    main()
