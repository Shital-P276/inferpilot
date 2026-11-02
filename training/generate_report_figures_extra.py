"""
training/generate_report_figures_extra.py

Additional IEEE report figures (A + B2-B5). Creates NEW files only; does not
touch existing figures, CSVs, or generate_report_figures.py.

Outputs (reports/figures/):
  f4_confusion_pairs.png   top-8 confused pairs per tier, horizontal bars
  f9_dataset_composition.png  tier-label mix + capped-holdout support histogram
  f10_holdout_vs_ood.png   per-class support (holdout vs full OOD) + accuracy gap
  f11_tier_ladder.png      params vs accuracy ladder (holdout + full OOD)
  f12_latency_vs_load.png  avg latency / throughput heatmaps by load phase

Sources:
  reports/figures/tier_confusion/{tier}_top15_confused.csv + confusion_summary.csv
  router/utility_labels.csv, router/held_out_images.csv
  training/macro_f1_reports/{tier}_full_ood_per_class.csv  (numeric rows only;
      'macro avg'/'weighted avg' summary rows excluded)
  router/model_benchmarks.json
  router/concurrent_experiment_summary_v3a_full11.csv  (Sep 16 canonical run)

Run: source ~/venv_inferpilot/bin/activate && python training/generate_report_figures_extra.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG = REPO_ROOT / "reports" / "figures"
TC = FIG / "tier_confusion"

SINGLE_W, DOUBLE_W = 3.5, 7.16
TIERS = ["fast", "balanced", "heavy"]
HOLDOUT_ACC = {"fast": 0.2300, "balanced": 0.3428, "heavy": 0.5245}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "figure.dpi": 300})


def save(fig, name):
    out = FIG / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.relative_to(REPO_ROOT)}")


def load_ood(tier):
    d = pd.read_csv(REPO_ROOT / "training" / "macro_f1_reports" /
                    f"{tier}_full_ood_per_class.csv")
    d = d[d.class_label.astype(str).str.fullmatch(r"\d+")].copy()
    d["class_label"] = d.class_label.astype(int)
    return d


def full_ood_accuracy(d):
    return float((d.recall * d.support).sum() / d.support.sum())


def confusion_pairs():
    print("A: f4_confusion_pairs...")
    summary = pd.read_csv(TC / "confusion_summary.csv").set_index("tier")
    fig, axes = plt.subplots(3, 1, figsize=(SINGLE_W, 7.6))
    for ax, tier in zip(axes, TIERS):
        d = pd.read_csv(TC / f"{tier}_top15_confused.csv").head(8)
        labels = [f"{t} -> {p}" for t, p in zip(d.true_name, d.pred_name)]
        y = np.arange(len(d))
        ax.barh(y, d["count"], color="steelblue", height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=5.2)
        ax.invert_yaxis()
        ax.set_xlabel("count", fontsize=6)
        ax.tick_params(axis="x", labelsize=6)
        for yi, c in zip(y, d["count"]):
            ax.text(c + 0.5, yi, str(int(c)), va="center", fontsize=5)
        ax.set_title(f"{tier} (holdout acc={HOLDOUT_ACC[tier]:.3f})", fontsize=8)
    fig.suptitle("Most frequent confusion pairs (true -> predicted)", fontsize=9)
    fig.tight_layout()
    save(fig, "f4_confusion_pairs.png")


def dataset_composition():
    print("B2: f9_dataset_composition...")
    u = pd.read_csv(REPO_ROOT / "router" / "utility_labels.csv")
    h = pd.read_csv(REPO_ROOT / "router" / "held_out_images.csv")
    tier_counts = u.best_tier.value_counts().reindex(TIERS).fillna(0).astype(int)
    support = h.true_label.value_counts().values

    fig, axes = plt.subplots(2, 1, figsize=(SINGLE_W, 4.8))
    ax = axes[0]
    bars = ax.bar(TIERS, tier_counts.values, color="slategray")
    for b, v in zip(bars, tier_counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 30, f"{v:,}",
                ha="center", fontsize=6)
    ax.set_ylabel("images", fontsize=6)
    ax.set_title(f"Router training labels by best tier (n={len(u):,})", fontsize=8)
    ax.tick_params(labelsize=6)

    ax = axes[1]
    ax.hist(support, bins=np.arange(0, 105, 5), color="steelblue",
            edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.axvline(100, color="firebrick", ls="--", lw=0.8)
    ax.text(98, 2.0, "cap=100", ha="right", va="bottom", fontsize=5.5,
            color="firebrick")
    ax.set_xlabel("images per class", fontsize=6)
    ax.set_ylabel("classes (log)", fontsize=6)
    ax.set_title(f"Capped holdout class support (n={len(h):,}, "
                 f"{h.true_label.nunique()} classes)", fontsize=8)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    save(fig, "f9_dataset_composition.png")


def holdout_vs_ood():
    print("B5: f10_holdout_vs_ood...")
    h = pd.read_csv(REPO_ROOT / "router" / "held_out_images.csv")
    hold_support = np.sort(h.true_label.value_counts().values)[::-1]
    ood_support = np.sort(load_ood("heavy").support.values)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.7))
    ax = axes[0]
    ax.plot(np.arange(1, len(ood_support) + 1), ood_support, "o-", ms=2,
            lw=0.8, label=f"full OOD test (n={int(ood_support.sum()):,})")
    ax.plot(np.arange(1, len(hold_support) + 1), hold_support, "s-", ms=2,
            lw=0.8, label=f"capped holdout (n={int(hold_support.sum()):,})")
    ax.set_yscale("log")
    ax.set_xlabel("class rank", fontsize=6)
    ax.set_ylabel("images (log)", fontsize=6)
    ax.set_title("Class support: full OOD vs capped holdout", fontsize=8)
    ax.legend(fontsize=5.5)
    ax.tick_params(labelsize=6)

    ax = axes[1]
    x = np.arange(len(TIERS))
    ood_acc = [full_ood_accuracy(load_ood(t)) for t in TIERS]
    hold_acc = [HOLDOUT_ACC[t] for t in TIERS]
    ax.bar(x - 0.2, hold_acc, 0.4, label="capped holdout", color="steelblue")
    ax.bar(x + 0.2, ood_acc, 0.4, label="full OOD test", color="darkorange")
    for xi, a, b in zip(x, hold_acc, ood_acc):
        ax.text(xi - 0.2, a + 0.01, f"{a:.3f}", ha="center", fontsize=5.5)
        ax.text(xi + 0.2, b + 0.01, f"{b:.3f}", ha="center", fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels(TIERS, fontsize=6)
    ax.set_ylabel("accuracy", fontsize=6)
    ax.set_ylim(0, 0.85)
    ax.set_title("Tier accuracy on the two substrates", fontsize=8)
    ax.legend(fontsize=5.5)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    save(fig, "f10_holdout_vs_ood.png")


def tier_ladder():
    print("B3: f11_tier_ladder...")
    bench = {b["tier"]: b for b in
             json.load(open(REPO_ROOT / "router" / "model_benchmarks.json"))}
    params = [bench[t]["num_params"] for t in TIERS]
    ood_acc = [full_ood_accuracy(load_ood(t)) for t in TIERS]
    hold_acc = [HOLDOUT_ACC[t] for t in TIERS]

    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.8))
    ax.plot(params, hold_acc, "o-", color="steelblue", ms=4,
            label="capped holdout")
    ax.plot(params, ood_acc, "s--", color="darkorange", ms=4,
            label="full OOD test")
    for t, p, a in zip(TIERS, params, hold_acc):
        heavy = t == "heavy"
        ax.annotate(f"{t}\n{bench[t]['avg_latency_ms']:.1f} ms",
                    (p, a), textcoords="offset points",
                    xytext=(10, -8) if heavy else (0, -22),
                    ha="left" if heavy else "center", fontsize=5.5)
    ax.set_xscale("log")
    ax.set_xlim(params[0] * 0.5, params[-1] * 2.0)
    ax.set_xlabel("parameters (log)", fontsize=6)
    ax.set_ylabel("accuracy", fontsize=6)
    ax.set_title("Capacity ladder: params vs accuracy", fontsize=8)
    ax.legend(fontsize=5.5, loc="upper left")
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    save(fig, "f11_tier_ladder.png")


def latency_vs_load():
    print("B4: f12_latency_vs_load...")
    s = pd.read_csv(REPO_ROOT / "router" /
                    "concurrent_experiment_summary_v3a_full11.csv")
    order = ["always_fast", "round_robin", "rule_based", "ml_router",
             "single_shot_router", "correctness_gate_router",
             "correctness_gate_v3_router", "correctness_gate_v3a_router",
             "hybrid_gate", "hybrid_dumb", "always_heavy"]
    short = {k: k.replace("correctness_gate_", "gate_") for k in order}
    phases = ["light", "heavy", "burst"]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 3.2))
    for ax, col, title, fmt in [
            (axes[0], "avg_latency_ms", "avg latency (ms)", "{:.0f}"),
            (axes[1], "throughput_rps", "throughput (req/s)", "{:.1f}")]:
        mat = np.zeros((len(order), len(phases)))
        for i, strat in enumerate(order):
            for j, ph in enumerate(phases):
                row = s[(s.strategy == strat) & (s.concurrency_phase == ph)]
                mat[i, j] = float(row[col].iloc[0]) if len(row) else np.nan
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, fontsize=6)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([short[k] for k in order], fontsize=5.5)
        ax.set_title(title, fontsize=8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                        fontsize=5, color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    save(fig, "f12_latency_vs_load.png")


def main():
    print("Extra report figures -> reports/figures/")
    confusion_pairs()
    dataset_composition()
    holdout_vs_ood()
    tier_ladder()
    latency_vs_load()
    print("Done.")


if __name__ == "__main__":
    main()
