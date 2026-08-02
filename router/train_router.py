"""
Comprehensive router model comparison:
  - Logistic Regression, Random Forest, Gradient Boosting (class-weighted) -- baseline scoped set
  - SMOTE-augmented versions of the same 3 -- oversampling minority classes
  - Balanced Random Forest (imblearn) -- per-tree majority undersampling
  - LightGBM (class-weighted) -- stronger gradient boosting implementation
  - Naive confidence-threshold baseline -- validates ML routing beats a hand-written rule

Output:
  router/router_comparison.csv
  router/router_best_model.pkl
  router/confusion_matrices.png
  router/feature_importance.png
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, balanced_accuracy_score
)
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

import lightgbm as lgb

# ---------------- Config ----------------
BASE_DIR = Path(__file__).resolve().parent
LABELS_PATH = BASE_DIR / "utility_labels.csv"
COMPARISON_OUTPUT = BASE_DIR / "router_comparison.csv"
BEST_MODEL_OUTPUT = BASE_DIR / "router_best_model.pkl"
CONFUSION_PLOT = BASE_DIR / "confusion_matrices.png"
IMPORTANCE_PLOT = BASE_DIR / "feature_importance.png"

FEATURE_COLS_NUMERIC = ["width", "height", "file_size_kb", "blur_score",
                         "brightness", "edge_density", "fast_confidence"]
FEATURE_COL_CATEGORICAL = ["load_stage"]
TARGET_COL = "best_tier"

SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


# ---------------- Naive confidence-threshold baseline ----------------
class ConfidenceThresholdBaseline(BaseEstimator, ClassifierMixin):
    """
    Hand-written rule, no learning: routes purely on fast_confidence.
    high confidence -> fast, medium -> balanced, low -> heavy.
    Thresholds are tuned on the TRAIN set only (grid search over a small
    range), same discipline as fitting any other model -- not peeked at test.
    """
    def __init__(self, low_thresh=0.9, high_thresh=0.99, conf_col_idx=6):
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.conf_col_idx = conf_col_idx  # index of fast_confidence in the numeric block

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        conf = X[:, self.conf_col_idx]
        preds = np.where(conf >= self.high_thresh, "fast",
                 np.where(conf >= self.low_thresh, "balanced", "heavy"))
        return preds


def tune_confidence_baseline(X_train_raw, y_train_labels):
    """Small grid search over thresholds using train data only."""
    best_score = -1
    best_params = (0.9, 0.99)
    conf = X_train_raw["fast_confidence"].values

    for low in np.arange(0.5, 0.95, 0.05):
        for high in np.arange(low + 0.02, 1.0, 0.02):
            preds = np.where(conf >= high, "fast",
                     np.where(conf >= low, "balanced", "heavy"))
            score = balanced_accuracy_score(y_train_labels, preds)
            if score > best_score:
                best_score = score
                best_params = (low, high)

    print(f"  Tuned thresholds: low={best_params[0]:.2f} high={best_params[1]:.2f} "
          f"(train balanced_acc={best_score:.4f})")
    return best_params


def load_data():
    df = pd.read_csv(LABELS_PATH)
    print(f"Loaded {len(df)} labeled samples")
    print(f"Label distribution:\n{df[TARGET_COL].value_counts()}\n")
    return df


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), FEATURE_COLS_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_COL_CATEGORICAL),
    ])


def evaluate_model(name, y_test, y_pred, label_classes):
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=label_classes, zero_division=0
    )

    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(f"Overall accuracy:  {acc:.4f}")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(classification_report(y_test, y_pred, labels=label_classes, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=label_classes)
    idx = {c: i for i, c in enumerate(label_classes)}

    return {
        "model_name": name,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "fast_f1": round(f1[idx["fast"]], 4),
        "balanced_f1": round(f1[idx["balanced"]], 4),
        "heavy_f1": round(f1[idx["heavy"]], 4),
    }, cm


def main():
    df = load_data()

    X_raw = df[FEATURE_COLS_NUMERIC + FEATURE_COL_CATEGORICAL]
    y_labels = df[TARGET_COL].values

    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    class_names = list(le.classes_)  # e.g. ['balanced', 'fast', 'heavy']

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, stratify=y_temp, random_state=SEED
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}\n")

    y_train_labels = le.inverse_transform(y_train)
    y_test_labels = le.inverse_transform(y_test)

    preprocessor = build_preprocessor()
    results = []
    confusion_data = {}
    fitted_pipelines = {}

    # ---- 1-3: Baseline scoped set (class-weighted) ----
    baseline_models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=SEED),
    }
    for name, clf in baseline_models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("clf", clf)])
        if name == "Gradient Boosting":
            weights = compute_sample_weight("balanced", y_train)
            pipe.fit(X_train, y_train, clf__sample_weight=weights)
        else:
            pipe.fit(X_train, y_train)
        y_pred = le.inverse_transform(pipe.predict(X_test))
        metrics, cm = evaluate_model(name, y_test_labels, y_pred, class_names)
        results.append(metrics)
        confusion_data[name] = cm
        fitted_pipelines[name] = pipe

    # ---- 4-6: SMOTE-augmented versions ----
    smote_models = {
        "Logistic Regression + SMOTE": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest + SMOTE": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1),
        "Gradient Boosting + SMOTE": GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=SEED),
    }
    for name, clf in smote_models.items():
        pipe = ImbPipeline([
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=SEED, k_neighbors=5)),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        y_pred = le.inverse_transform(pipe.predict(X_test))
        metrics, cm = evaluate_model(name, y_test_labels, y_pred, class_names)
        results.append(metrics)
        confusion_data[name] = cm
        fitted_pipelines[name] = pipe

    # ---- 7: Balanced Random Forest (imblearn) ----
    brf_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("clf", BalancedRandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1,
            sampling_strategy="all", replacement=True, bootstrap=False
        )),
    ])
    brf_pipe.fit(X_train, y_train)
    y_pred = le.inverse_transform(brf_pipe.predict(X_test))
    metrics, cm = evaluate_model("Balanced Random Forest", y_test_labels, y_pred, class_names)
    results.append(metrics)
    confusion_data["Balanced Random Forest"] = cm
    fitted_pipelines["Balanced Random Forest"] = brf_pipe

    # ---- 8: LightGBM (class-weighted) ----
    lgb_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("clf", lgb.LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            class_weight="balanced", random_state=SEED, verbose=-1
        )),
    ])
    lgb_pipe.fit(X_train, y_train)
    y_pred = le.inverse_transform(lgb_pipe.predict(X_test))
    metrics, cm = evaluate_model("LightGBM", y_test_labels, y_pred, class_names)
    results.append(metrics)
    confusion_data["LightGBM"] = cm
    fitted_pipelines["LightGBM"] = lgb_pipe

    # ---- 9: Naive confidence-threshold baseline (no learning) ----
    print(f"\n{'='*60}\nConfidence-Threshold Baseline (tuning on train)\n{'='*60}")
    low, high = tune_confidence_baseline(X_train, y_train_labels)
    conf_test = X_test["fast_confidence"].values
    y_pred = np.where(conf_test >= high, "fast",
              np.where(conf_test >= low, "balanced", "heavy"))
    metrics, cm = evaluate_model("Confidence Threshold (rule-based)", y_test_labels, y_pred, class_names)
    results.append(metrics)
    confusion_data["Confidence Threshold (rule-based)"] = cm

    # ---- Summary ----
    results_df = pd.DataFrame(results).sort_values("balanced_accuracy", ascending=False)
    results_df.to_csv(COMPARISON_OUTPUT, index=False)
    print(f"\n{'='*60}\nFULL COMPARISON (sorted by balanced accuracy)\n{'='*60}")
    print(results_df.to_string(index=False))
    print(f"\nWritten to {COMPARISON_OUTPUT}")

    best_name = results_df.iloc[0]["model_name"]
    print(f"\nBest model overall: {best_name}")

    if best_name in fitted_pipelines:
        with open(BEST_MODEL_OUTPUT, "wb") as f:
            pickle.dump({
                "model": fitted_pipelines[best_name],
                "label_encoder": le,
                "feature_cols": FEATURE_COLS_NUMERIC + FEATURE_COL_CATEGORICAL,
            }, f)
        print(f"Saved to {BEST_MODEL_OUTPUT}")
    else:
        print("(Best model is the rule-based baseline -- no pickle needed, it's just two thresholds:"
              f" low={low:.3f}, high={high:.3f} on fast_confidence)")

    # ---- Confusion matrices: show top 4 models only (too many to plot all 9 legibly) ----
    top4 = results_df.head(4)["model_name"].tolist()
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, name in zip(axes, top4):
        sns.heatmap(confusion_data[name], annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(CONFUSION_PLOT)
    plt.close(fig)
    print(f"Confusion matrices (top 4) saved to {CONFUSION_PLOT}")

    # ---- Feature importance: RF, GB, LightGBM ----
    feature_names = (
        FEATURE_COLS_NUMERIC
        + list(fitted_pipelines["Random Forest"].named_steps["preprocess"]
               .named_transformers_["cat"].get_feature_names_out(FEATURE_COL_CATEGORICAL))
    )
    importance_models = [
        ("Random Forest", fitted_pipelines["Random Forest"]),
        ("Gradient Boosting", fitted_pipelines["Gradient Boosting"]),
        ("LightGBM", fitted_pipelines["LightGBM"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    for ax, (name, pipeline) in zip(axes, importance_models):
        importances = pipeline.named_steps["clf"].feature_importances_
        order = np.argsort(importances)[::-1]
        ax.barh([feature_names[i] for i in order], importances[order])
        ax.set_title(f"{name} — Feature Importance", fontsize=10)
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(IMPORTANCE_PLOT)
    plt.close(fig)
    print(f"Feature importance plot saved to {IMPORTANCE_PLOT}")


if __name__ == "__main__":
    main()