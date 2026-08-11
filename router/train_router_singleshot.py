"""
train_router_singleshot.py

Path A single-shot router: predicts the best tier directly from
request-level features ONLY -- no fast_confidence, no cascade, no
proxy call to Fast at all. Exactly one network hop at inference time.

This is a direct mirror of train_router.py, with two changes:
  1. "fast_confidence" removed from FEATURE_COLS_NUMERIC (the whole point
     of this script -- no cascade, no Fast pre-call).
  2. The "Confidence Threshold (rule-based)" baseline (#9 in the original)
     is REMOVED entirely, not just skipped -- it requires fast_confidence
     to threshold against, which doesn't exist in this feature set. There
     is no meaningful single-shot equivalent of that baseline.

Everything else -- the 8 remaining models, preprocessing (StandardScaler +
OneHotEncoder), train/val/test split sizes and seed, evaluation metrics,
confusion matrix / feature importance plots -- matches train_router.py
exactly, so results are a fair apples-to-apples comparison.

Output:
  router/router_singleshot_comparison.csv
  router/router_singleshot_model.pkl   -- SEPARATE from router_best_model.pkl
  router/singleshot_confusion_matrices.png
  router/singleshot_feature_importance.png

USAGE:
    python router/train_router_singleshot.py
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
COMPARISON_OUTPUT = BASE_DIR / "router_singleshot_comparison.csv"
BEST_MODEL_OUTPUT = BASE_DIR / "router_singleshot_model.pkl"
CONFUSION_PLOT = BASE_DIR / "singleshot_confusion_matrices.png"
IMPORTANCE_PLOT = BASE_DIR / "singleshot_feature_importance.png"

# "fast_confidence" deliberately dropped -- see module docstring.
FEATURE_COLS_NUMERIC = ["width", "height", "file_size_kb", "blur_score",
                         "brightness", "edge_density"]
FEATURE_COL_CATEGORICAL = ["load_stage"]
TARGET_COL = "best_tier"

SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


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
    class_names = list(le.classes_)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, stratify=y_temp, random_state=SEED
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}\n")

    y_test_labels = le.inverse_transform(y_test)

    preprocessor = build_preprocessor()
    results = []
    confusion_data = {}
    fitted_pipelines = {}

    # ---- 1-3: Baseline scoped set (class-weighted) ----
    baseline_models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=SEED, n_jobs=1),
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
        "Random Forest + SMOTE": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=SEED, n_jobs=1),
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
    # n_jobs=1 from the start (unlike the original cascade script's n_jobs=-1)
    # -- the cascade router's n_jobs=-1 was later found to cost 62.5-84.8%
    # extra single-row inference latency, purely from thread-pool spawn
    # overhead. Baking the fix in from the start here avoids repeating that.
    brf_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("clf", BalancedRandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=SEED, n_jobs=1,
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
            class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=1
        )),
    ])
    lgb_pipe.fit(X_train, y_train)
    y_pred = le.inverse_transform(lgb_pipe.predict(X_test))
    metrics, cm = evaluate_model("LightGBM", y_test_labels, y_pred, class_names)
    results.append(metrics)
    confusion_data["LightGBM"] = cm
    fitted_pipelines["LightGBM"] = lgb_pipe

    # NOTE: "Confidence Threshold (rule-based)" baseline from the original
    # script (#9) is intentionally NOT included here -- it requires
    # fast_confidence, which doesn't exist in this single-shot feature set.

    # ---- Summary ----
    results_df = pd.DataFrame(results).sort_values("balanced_accuracy", ascending=False)
    results_df.to_csv(COMPARISON_OUTPUT, index=False)
    print(f"\n{'='*60}\nFULL COMPARISON (sorted by balanced accuracy)\n{'='*60}")
    print(results_df.to_string(index=False))
    print(f"\nWritten to {COMPARISON_OUTPUT}")

    best_name = results_df.iloc[0]["model_name"]
    print(f"\nBest model overall: {best_name}")

    with open(BEST_MODEL_OUTPUT, "wb") as f:
        pickle.dump({
            "model": fitted_pipelines[best_name],
            "label_encoder": le,
            "feature_cols": FEATURE_COLS_NUMERIC + FEATURE_COL_CATEGORICAL,
        }, f)
    print(f"Saved to {BEST_MODEL_OUTPUT}")

    print(f"\n{'=' * 60}")
    print("COMPARISON TO CASCADE ROUTER (fill in from your records)")
    print(f"{'=' * 60}")
    print(f"  Cascade router (WITH fast_confidence), Balanced Random Forest: 0.5588 balanced accuracy")
    print(f"  THIS single-shot best model ({best_name}): {results_df.iloc[0]['balanced_accuracy']:.4f} balanced accuracy")
    print(f"\nExpected: notably lower than 0.5588, since fast_confidence carried real signal. "
          f"The question this answers isn't 'is single-shot as accurate' (it won't be) -- it's "
          f"whether the accuracy loss is small enough to be worth it once the cascade's ~50-100ms "
          f"network tax is removed entirely. That's answered by re-running the core experiment + "
          f"utility sweep with this model wired in as a true single-shot strategy, not by this "
          f"number alone.")

    # ---- Confusion matrices: top 4 (or fewer, since we only have 8 models not 9) ----
    top_n = min(4, len(results_df))
    top_models = results_df.head(top_n)["model_name"].tolist()
    fig, axes = plt.subplots(1, top_n, figsize=(5.5 * top_n, 5))
    if top_n == 1:
        axes = [axes]
    for ax, name in zip(axes, top_models):
        sns.heatmap(confusion_data[name], annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(CONFUSION_PLOT)
    plt.close(fig)
    print(f"Confusion matrices (top {top_n}) saved to {CONFUSION_PLOT}")

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
