"""
router/router_eval_utils.py

Shared helpers for the random-routing-baseline diagnostic used to judge
whether a router's balanced_routing_acc reflects genuine per-image routing
skill or is just an artifact of tier base-rate imbalance.

The metric is the SAME one used in training/train_correctness_gate.py:
balanced_routing_acc = macro-average over the 3 chosen-tier classes -- for
each tier, mean({tier}_correct) among images actually routed to that tier,
then averaged across tiers (empty tiers excluded). NOT sklearn's
balanced_accuracy_score (macro-recall over true tier labels), which is what
the router training scripts reported originally.

run_router_diagnostic() loads an already-trained router pickle, reproduces
the exact train/test split used by its training script (same seed/stratify
on best_tier), re-runs inference on the test set ONLY, and prints the
baseline comparison. No retraining, no changes to routing decision logic.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TIER_NAMES = ["fast", "balanced", "heavy"]
TIER_CORRECT_COLUMNS = ["fast_correct", "balanced_correct", "heavy_correct"]


def random_baseline_routing_acc(tier_correct_labels, pct_fast, pct_balanced, pct_heavy, n_trials=20, seed=42):
    """
    Monte-Carlo random-routing baseline: assigns each test image to a tier at
    random with the given proportions (the same tier distribution the real
    model produced), then computes the same macro-averaged balanced_routing_acc
    per trial. The mean/std across trials show how much balanced_routing_acc a
    random policy with the SAME routing skew would achieve by chance alone --
    separating genuine per-image routing skill from tier base-rate imbalance.

    pct_* are percentages (0-100). Empty tiers are skipped exactly as in the
    real metric.
    """
    probs = np.array([pct_fast, pct_balanced, pct_heavy], dtype=float)
    probs = probs / probs.sum()
    n = len(tier_correct_labels)
    trial_accs = []
    for trial in range(n_trials):
        trial_rng = np.random.default_rng(seed + trial)
        chosen = trial_rng.choice(3, size=n, p=probs)
        per_tier_acc = []
        for t in range(3):
            mask = chosen == t
            if mask.sum() > 0:
                per_tier_acc.append(float(tier_correct_labels[mask, t].mean()))
        trial_accs.append(float(np.nanmean(per_tier_acc)) if per_tier_acc else float("nan"))
    trial_accs = np.array(trial_accs)
    return float(np.nanmean(trial_accs)), float(np.nanstd(trial_accs))


def balanced_routing_metrics(chosen, test_true):
    """
    Gate-style balanced_routing_acc from a router's tier predictions.

    chosen:    int array of tier index per test image (0=fast, 1=balanced, 2=heavy)
    test_true: [N, 3] array of {tier}_correct ground-truth columns

    Returns (pcts, bal_acc) where pcts = [pct_fast, pct_balanced, pct_heavy]
    of the actual tier distribution produced by the router.
    """
    pcts = [float((chosen == t).mean()) * 100 for t in range(3)]
    per_tier_acc = []
    for t in range(3):
        mask = chosen == t
        if mask.sum() > 0:
            per_tier_acc.append(float(test_true[mask, t].mean()))
    bal_acc = float(np.nanmean(per_tier_acc)) if per_tier_acc else float("nan")
    return pcts, bal_acc


def print_baseline_comparison(name, bal_acc, rnd_mean, rnd_std):
    """
    Prints the single comparison line plus the noise warning, and returns
    (lift, sigma). WARNING fires when the lift is < 2x the random baseline's
    std (i.e. significance < 2 sigma).
    """
    lift = bal_acc - rnd_mean
    sigma = lift / rnd_std if rnd_std > 0 else (float("inf") if lift > 0 else 0.0)
    print(f"Router: {name} | Model balanced_routing_acc: {bal_acc:.4f} | "
          f"Random baseline (matched proportions): {rnd_mean:.4f} +/- {rnd_std:.4f} | "
          f"Lift: +{lift:.4f} | Significance: {sigma:.1f}sigma")
    if sigma < 2:
        print(f"WARNING: model's advantage over random routing at matched proportions is not clearly "
              f"distinguishable from noise -- do not report this as a genuine routing-skill result "
              f"without further investigation.")
    return lift, sigma


def run_router_diagnostic(name, model_pkl_path, labels_path, target_col,
                          test_size, seed, n_trials=20, baseline_seed=42):
    """
    Read-only random-routing-baseline diagnostic for an already-trained router.

    Reproduces the exact test split the training script used (same seed,
    stratify on best_tier), re-runs inference on the test set ONLY, computes
    the gate-style balanced_routing_acc and the matched-proportions random
    baseline, then prints the comparison.

    model_pkl_path: pickled {"model", "label_encoder", "feature_cols"} bundle.
    """
    with open(model_pkl_path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    le = bundle["label_encoder"]
    feature_cols = bundle["feature_cols"]

    df = pd.read_csv(labels_path)
    X_raw = df[feature_cols]
    y = le.transform(df[target_col].values)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_raw, y, test_size=test_size, stratify=y, random_state=seed
    )

    y_pred = le.inverse_transform(model.predict(X_test))
    tier_index = {t: i for i, t in enumerate(TIER_NAMES)}
    chosen = np.array([tier_index[p] for p in y_pred])

    test_true = df.loc[X_test.index][TIER_CORRECT_COLUMNS].values

    pcts, bal_acc = balanced_routing_metrics(chosen, test_true)
    rnd_mean, rnd_std = random_baseline_routing_acc(
        test_true, pcts[0], pcts[1], pcts[2], n_trials=n_trials, seed=baseline_seed
    )

    print(f"\n{'='*60}")
    print(f"Random-routing baseline diagnostic: {name}")
    print(f"{'='*60}")
    print(f"  Test images: {len(chosen)}  |  tier distribution produced by router: "
          f"fast={pcts[0]:.1f}% balanced={pcts[1]:.1f}% heavy={pcts[2]:.1f}%")
    print_baseline_comparison(name, bal_acc, rnd_mean, rnd_std)
    return bal_acc, rnd_mean, rnd_std