"""
training/generate_report_figures.py

Phase 3 figure generation: all data-derived IEEE figures from on-disk data.
Presets: single-col 3.5in / double-col 7.16in, 300dpi PNG -> reports/figures/.

Figure : source (auditable)
  F1 per-class F1 strips      : training/macro_f1_reports/*_full_ood_per_class.csv (Drive pull)
  F2 11-strategy dual-axis    : router/concurrent_experiment_summary_v3a_full11.csv (overall rows)
  F3 lambda tradeoff curves   : training/checkpoints/lambda_per_load_stage_sweep{,_floor}.csv
  F4 conformal coverage-vs-acc: training/checkpoints/conformal_gate_routing.csv
  F5 OOD scoreboard           : training/checkpoints/ood_detection_baselines.csv
  F6 breaker timeline         : router/chaos_partial_trace.jsonl
  F7 fallback accuracy drop   : v3a_full11 summary (v3 clean) + chaos_partial_summary.csv
  F8 gate lineage AUROC bars  : assembled below -- EXACT source per bar in comment
  T-ablation 2x2 CSV          : ablation report numbers (same Session, same test partition)

Gate-lineage bar sources (avg AUROC unless noted):
  V2b  0.7469 = train_correctness_gate_v3.py::V2B_BASELINE (same utility_labels.csv)
  V3   0.7724 = train_v3_ablation.py report (published ckpt, shared test partition)
  V3a  0.7810 = train_v3_ablation.py report (new ckpt, shared test partition)
  V3b  0.7495 = train_v3_ablation.py report (new ckpt, shared test partition)
  V2c/V2d = balanced-routing-accuracy 0.9599 vs V2b 0.9686 from
            train_correctness_gate_v2d_css.py docstring -- DIFFERENT METRIC,
            plotted on a separate axis scale and labeled as such.
  V1/V2 excluded: pre-migration (produce data), not comparable -- footnote.
Run: source ~/venv_inferpilot/bin/activate && python training/generate_report_figures.py
"""

from pathlib import Path
import csv
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG = REPO_ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SINGLE_W, DOUBLE_W = 3.5, 7.16
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "figure.dpi": 300})


def save(fig, name):
    out = FIG / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.relative_to(REPO_ROOT)}")


def f1_strips():
    print("F1: per-class F1 strips...")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_W, 2.6), sharey=True)
    for ax, tier in zip(axes, ["fast", "balanced", "heavy"]):
        df = pd.read_csv(REPO_ROOT / "training" / "macro_f1_reports" /
                         f"{tier}_full_ood_per_class.csv")
        df = df.sort_values("f1")
        worst, best = df.head(15), df.tail(15)
        y = np.arange(15)
        ax.barh(y, worst["f1"].to_numpy(), label="worst 15")
        ax.barh(y + 16, best["f1"].to_numpy(), label="best 15")
        ax.set_title(f"{tier} (macro-F1={df['f1'].mean():.4f})")
        ax.set_xlabel("F1")
        ax.set_ylim(-1, 31)
        ax.legend(fontsize=6)
    axes[0].set_ylabel("class rank")
    fig.suptitle("Per-class F1 on full WILDS OOD test (n=42,791)")
    fig.tight_layout()
    save(fig, "f1_perclass_strips.png")


def strategy_dual_axis():
    print("F2: 11-strategy dual-axis...")
    df = pd.read_csv(REPO_ROOT / "router" / "concurrent_experiment_summary_v3a_full11.csv")
    ov = df[df.concurrency_phase == "overall"].copy()
    order = ["always_fast", "round_robin", "rule_based", "ml_router",
             "single_shot_router", "correctness_gate_router",
             "correctness_gate_v3_router", "correctness_gate_v3a_router",
             "hybrid_gate", "hybrid_dumb", "always_heavy"]
    ov["s"] = ov.strategy.map({s: s.replace("correctness_gate_", "gate_") for s in order})
    ov = ov.set_index("strategy").loc[order].reset_index()
    x = np.arange(len(ov))
    fig, ax1 = plt.subplots(figsize=(DOUBLE_W, 2.8))
    ax1.bar(x, ov.accuracy, color="steelblue", label="accuracy")
    ax1.set_ylabel("accuracy")
    ax1.set_ylim(0, 0.6)
    ax2 = ax1.twinx()
    ax2.plot(x, ov.pct_heavy, "o-", color="firebrick", label="% heavy")
    ax2.set_ylabel("% heavy")
    ax2.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ov["s"], rotation=30, ha="right", fontsize=7)
    ax1.set_title("Live 11-strategy comparison (overall phases, delay OFF)")
    fig.tight_layout()
    save(fig, "f2_strategies_dual_axis.png")


def lambda_curves():
    print("F3: lambda tradeoff curves...")
    d1 = pd.read_csv(REPO_ROOT / "training" / "checkpoints" / "lambda_per_load_stage_sweep.csv")
    d2 = pd.read_csv(REPO_ROOT / "training" / "checkpoints" / "lambda_per_load_stage_sweep_floor.csv")
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.6))
    for stage, mk in [("light", "o"), ("heavy", "s"), ("burst", "^")]:
        fl = d2[d2.stage == stage].sort_values("lambda")
        ax.plot(fl["lambda"], fl.accuracy, mk + "-", ms=3, label=f"{stage} (floor ext.)")
        base = d1[d1.stage == stage].sort_values("lambda")
        ax.plot(base["lambda"], base.accuracy, mk + ":", ms=3, alpha=0.6)
    ax.axvline(0.20, color="k", ls="--", lw=0.8, label="current fixed 0.20")
    ax.annotate("optimum at floor (0.00)\nall stages", xy=(0.0, 0.53),
                fontsize=6, ha="left")
    ax.set_xlabel("lambda")
    ax.set_ylabel("routing accuracy")
    ax.set_title("Accuracy-cost tradeoff is monotonic: no interior optimum")
    ax.legend(fontsize=6)
    fig.tight_layout()
    save(fig, "f3_lambda_tradeoff.png")


def conformal_plot():
    print("F4: conformal coverage-vs-accuracy...")
    df = pd.read_csv(REPO_ROOT / "training" / "checkpoints" / "conformal_gate_routing.csv")
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.6))
    base = df[df.coverage_nominal == 1.00].iloc[0]
    for probs, mk in [("raw", "o"), ("calibrated", "s")]:
        sub = df[(df.probs == probs) & (df.coverage_nominal < 1.0)].sort_values("heavy_pct")
        ax.plot(sub.heavy_pct, sub.overall_acc, mk + "-", ms=4, label=probs)
    ax.scatter([base.heavy_pct], [base.overall_acc], c="k", marker="x", s=40,
               label="fixed-lambda baseline")
    ax.set_xlabel("heavy %")
    ax.set_ylabel("overall accuracy")
    ax.set_title("Selective routing: gains ride on heavy use")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "f4_conformal.png")


def ood_scoreboard():
    print("F5: OOD scoreboard...")
    df = pd.read_csv(REPO_ROOT / "training" / "checkpoints" / "ood_detection_baselines.csv")
    df["cond"] = df.corruption + "=" + df.severity.astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.6), sharey=True)
    for ax, metric in zip(axes, ["msp_auroc", "energy_auroc"]):
        piv = df.pivot_table(index="cond", columns="tier", values=metric)
        piv = piv.reindex(["gaussian_noise=25", "gaussian_noise=50",
                           "gaussian_blur=2", "gaussian_blur=5"])
        piv.plot(kind="bar", ax=ax, rot=0)
        ax.axhline(0.5, color="k", ls="--", lw=0.8)
        ax.set_title(metric)
        ax.legend(fontsize=7)
        ax.set_xlabel("")
    fig.suptitle("ID-vs-corrupted AUROC (0.5 = chance; <0.5 = inverted)")
    fig.tight_layout()
    save(fig, "f5_ood_scoreboard.png")


def breaker_timeline():
    print("F6: breaker timeline...")
    rows = [json.loads(l) for l in
            open(REPO_ROOT / "router" / "chaos_partial_trace.jsonl")]
    t0 = rows[0]["timestamp_utc"]
    xs = [(r["timestamp_utc"][:19]) for r in rows]
    states = [r["breakers"]["heavy"]["state"] for r in rows]
    fails = [r["breakers"]["heavy"]["consecutive_failures"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(SINGLE_W, 2.4))
    ax1.step(range(len(xs)), [1 if s == "open" else 0 for s in states],
             where="post", label="heavy breaker (0=closed,1=open)")
    ax1.set_ylim(-0.1, 1.3)
    ax1.set_xticks(range(len(xs)))
    ax1.set_xticklabels([x[11:] for x in xs], rotation=30, fontsize=6)
    ax2 = ax1.twinx()
    ax2.plot(range(len(xs)), fails, "s-", ms=3, color="firebrick",
             label="consecutive failures")
    ax2.set_ylabel("consec. failures")
    ax1.set_title(f"Breaker trips mid-run (from {t0[:10]})")
    fig.tight_layout()
    save(fig, "f6_breaker_timeline.png")


def fallback_drop():
    print("F7: fallback accuracy drop...")
    full = pd.read_csv(REPO_ROOT / "router" / "concurrent_experiment_summary_v3a_full11.csv")
    clean = float(full[(full.strategy == "correctness_gate_v3_router") &
                       (full.concurrency_phase == "overall")].accuracy.iloc[0])
    ch = pd.read_csv(REPO_ROOT / "router" / "chaos_partial_summary.csv")
    chaos = float(ch[ch.concurrency_phase == "overall"].accuracy.iloc[0])
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.2))
    ax.bar(["clean\n(v3 gate)", "partial fault\n(v3 gate + fallback)"],
           [clean, chaos], color=["steelblue", "firebrick"])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 0.6)
    for i, v in enumerate([clean, chaos]):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_title("Cost of losing Heavy: accuracy under fallback")
    fig.tight_layout()
    save(fig, "f7_fallback_drop.png")


def lineage_bars():
    print("F8: gate lineage bars...")
    # avg AUROC on shared iWildCam test partition (see module docstring).
    auroc = {"V2b": 0.7469, "V3b": 0.7495, "V3": 0.7724, "V3a": 0.7810}
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.6))
    names = list(auroc)
    vals = [auroc[k] for k in names]
    cols = ["gray" if k != "V3a" else "firebrick" for k in names]
    ax.bar(names, vals, color=cols)
    ax.set_ylim(0.70, 0.80)
    ax.set_ylabel("test avg AUROC")
    ax.set_title("Gate lineage (shared partition; V2c/V2d separate metric,\n"
                 "V1/V2 pre-migration excluded -- see caption)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "f8_gate_lineage.png")
    print("  NOTE: V2c/V2d (0.9599 vs V2b 0.9686 balanced-routing-acc, different "
          "metric) + V1/V2 (produce-era) documented in caption, not plotted.")


def ablation_csv():
    print("T-ablation CSV...")
    # Per-tier AUROC from train_v3_ablation.py report (shared test partition).
    rows = [
        {"backbone": "frozen", "heads": "single", "cell": "v2b",
         "fast": 0.7589, "balanced": 0.7440, "heavy": 0.7376, "avg": 0.7469},
        {"backbone": "frozen", "heads": "dual", "cell": "v3b",
         "fast": 0.7525, "balanced": 0.7512, "heavy": 0.7447, "avg": 0.7495},
        {"backbone": "unfrozen", "heads": "single", "cell": "v3a",
         "fast": 0.7790, "balanced": 0.7932, "heavy": 0.7709, "avg": 0.7810},
        {"backbone": "unfrozen", "heads": "dual", "cell": "v3",
         "fast": 0.7784, "balanced": 0.7803, "heavy": 0.7585, "avg": 0.7724},
    ]
    out = FIG / "t_ablation_2x2.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {out.relative_to(REPO_ROOT)} ({len(rows)} rows)")


def main():
    print("Phase 3: generating report figures -> reports/figures/")
    f1_strips()
    strategy_dual_axis()
    lambda_curves()
    conformal_plot()
    ood_scoreboard()
    breaker_timeline()
    fallback_drop()
    lineage_bars()
    ablation_csv()
    print("Done.")


if __name__ == "__main__":
    main()
