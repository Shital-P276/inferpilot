"""
EXPERIMENTAL: Hybrid router combining Balanced Random Forest (primary) with
a rule-based override for the "heavy" class, motivated by the observation
that the naive confidence-threshold baseline outperformed every ML model
specifically on heavy_f1 (0.289 vs BRF's 0.187).

Logic: use BRF's prediction by default. If fast_confidence falls below the
LOW threshold (tuned on train, same as the baseline script), override to
"heavy" -- this targets exactly the case the naive rule was catching that
BRF was missing.

This is a standalone experiment, not wired into the main train_router.py.
If it meaningfully beats plain BRF on the same test set, promote it to
the main pipeline. If not, this file documents that the attempt was made
and didn't pan out -- still a reportable finding either way.

Output: printed comparison only, no files overwritten.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, precision_recall_fscore_support
)

# ---------------- Config ----------------
BASE_DIR = Path(__file__).resolve().parent
LABELS_PATH = BASE_DIR / "utility_labels.csv"
BEST_MODEL_PATH = BASE_DIR / "router_best_model.pkl"  # trained BRF pipeline from train_router.py

FEATURE_COLS_NUMERIC = ["width", "height", "file_size_kb", "blur_score",
                         "brightness", "edge_density", "fast_confidence"]
FEATURE_COL_CATEGORICAL = ["load_stage"]
TARGET_COL = "best_tier"

SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Same low threshold found during confidence-baseline tuning in train_router.py
# (re-tuned here on the same train split for consistency, not hardcoded blindly)
OVERRIDE_SEARCH_RANGE = np.arange(0.3, 0.7, 0.02)


def load_data():
    df = pd.read_csv(LABELS_PATH)
    return df


def evaluate(name, y_true, y_pred, label_classes):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_classes, zero_division=0
    )
    idx = {c: i for i, c in enumerate(label_classes)}

    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(f"Overall accuracy:  {acc:.4f}")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(classification_report(y_true, y_pred, labels=label_classes, zero_division=0))

    return {
        "model_name": name,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "fast_f1": round(f1[idx["fast"]], 4),
        "balanced_f1": round(f1[idx["balanced"]], 4),
        "heavy_f1": round(f1[idx["heavy"]], 4),
    }


def tune_override_threshold(X_train, y_train_labels, primary_preds_train):
    """
    Finds the low-confidence cutoff that, when used to override BRF's
    prediction to 'heavy', best improves TRAIN balanced accuracy.
    Tuned on train only -- test set stays untouched until final eval.
    """
    conf = X_train["fast_confidence"].values
    best_score = -1
    best_thresh = None

    for thresh in OVERRIDE_SEARCH_RANGE:
        hybrid_preds = np.where(conf < thresh, "heavy", primary_preds_train)
        score = balanced_accuracy_score(y_train_labels, hybrid_preds)
        if score > best_score:
            best_score = score
            best_thresh = thresh

    print(f"  Tuned override threshold: {best_thresh:.2f} (train balanced_acc={best_score:.4f})")
    return best_thresh


def main():
    df = load_data()
    X_raw = df[FEATURE_COLS_NUMERIC + FEATURE_COL_CATEGORICAL]

    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL].values)
    class_names = list(le.classes_)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, stratify=y_temp, random_state=SEED
    )

    y_train_labels = le.inverse_transform(y_train)
    y_test_labels = le.inverse_transform(y_test)

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}\n")

    # ---- Load the trained BRF pipeline from the main script's output ----
    with open(BEST_MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    brf_pipeline = saved["model"]
    print(f"Loaded primary model from {BEST_MODEL_PATH}")

    # ---- Baseline: plain BRF on test set ----
    brf_test_preds = le.inverse_transform(brf_pipeline.predict(X_test))
    brf_metrics = evaluate("Plain Balanced Random Forest (baseline)", y_test_labels, brf_test_preds, class_names)

    # ---- Tune override threshold on TRAIN predictions ----
    brf_train_preds = le.inverse_transform(brf_pipeline.predict(X_train))
    override_thresh = tune_override_threshold(X_train, y_train_labels, brf_train_preds)

    # ---- Apply hybrid override on TEST set ----
    conf_test = X_test["fast_confidence"].values
    hybrid_preds = np.where(conf_test < override_thresh, "heavy", brf_test_preds)
    hybrid_metrics = evaluate(
        f"Hybrid: BRF + heavy-override (thresh={override_thresh:.2f})",
        y_test_labels, hybrid_preds, class_names
    )

    # ---- Side-by-side comparison ----
    comparison = pd.DataFrame([brf_metrics, hybrid_metrics])
    print(f"\n{'='*60}\nSIDE-BY-SIDE COMPARISON\n{'='*60}")
    print(comparison.to_string(index=False))

    delta_bal_acc = hybrid_metrics["balanced_accuracy"] - brf_metrics["balanced_accuracy"]
    delta_heavy_f1 = hybrid_metrics["heavy_f1"] - brf_metrics["heavy_f1"]
    delta_fast_f1 = hybrid_metrics["fast_f1"] - brf_metrics["fast_f1"]

    print(f"\nDelta (hybrid - plain BRF):")
    print(f"  balanced_accuracy: {delta_bal_acc:+.4f}")
    print(f"  heavy_f1:          {delta_heavy_f1:+.4f}")
    print(f"  fast_f1:           {delta_fast_f1:+.4f}  (watch for regression here)")

    if delta_bal_acc > 0.01 and delta_fast_f1 > -0.03:
        print("\n>>> Hybrid looks like a genuine improvement. Consider promoting to main pipeline.")
    elif delta_bal_acc > 0:
        print("\n>>> Hybrid shows a small improvement but check fast_f1 regression before promoting.")
    else:
        print("\n>>> Hybrid did not improve on plain BRF. This is a valid negative result -- "
              "document the attempt, keep plain BRF as the production model.")


if __name__ == "__main__":
    main()