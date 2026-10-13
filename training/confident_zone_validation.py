"""
Confident-zone validation analysis.

Checks whether rule_based's 0.7 confidence threshold correctly identifies the
"easy" images that hybrid_gate never sends to the correctness gate, by comparing
rule_based's per-image correctness against correctness_gate_router's on the
subset of images where fast_confidence is OUTSIDE [0.60, 0.80].

Data sources:
  - router/core_experiment_raw_results.csv (per-image results for all strategies)
  - Fast CNN checkpoint (to compute fast_confidence locally, no gateway calls)

Output:
  - monitoring/reports/confident_zone_validation.csv (full confusion table)
  - Console printout of confusion matrix and summary statistics
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_RESULTS = BASE_DIR / "router" / "core_experiment_raw_results.csv"
FAST_CKPT = BASE_DIR / "training" / "checkpoints" / "fast_cnn_best.pt"
OUTPUT_CSV = BASE_DIR / "monitoring" / "reports" / "confident_zone_validation.csv"

BAND_LOW = 0.60
BAND_HIGH = 0.80

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_fast_model():
    """Load FastCNN from checkpoint on CPU (no GPU needed for 250 images)."""
    from training.models.fast_cnn import FastCNN

    device = torch.device("cpu")
    ckpt = torch.load(FAST_CKPT, map_location=device)
    class_names = ckpt["class_names"]
    model = FastCNN(num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded Fast model from {FAST_CKPT} ({len(class_names)} classes, CPU)")
    return model, class_names, device


def compute_fast_confidences(model, class_names, device, image_paths):
    """Run Fast model on all images, return predicted_class and confidence for each."""
    results = []
    for img_path in image_paths:
        full_path = BASE_DIR / img_path
        try:
            image = Image.open(full_path).convert("RGB")
            tensor = eval_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)
            results.append({
                "fast_predicted_class": class_names[pred_idx.item()],
                "fast_confidence": confidence.item(),
            })
        except Exception as e:
            print(f"  WARNING: failed on {img_path}: {e}")
            results.append({
                "fast_predicted_class": None,
                "fast_confidence": None,
            })
    return results


def main():
    print("=" * 70)
    print("CONFIDENT ZONE VALIDATION ANALYSIS")
    print("=" * 70)

    # Step 1: Load raw results CSV
    print("\n[Step 1] Loading raw results CSV...")
    df = pd.read_csv(RAW_RESULTS)
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Strategies: {df['strategy'].unique().tolist()}")

    for strat in ["rule_based", "correctness_gate_router"]:
        sub = df[df["strategy"] == strat]
        print(f"  {strat}: {len(sub)} rows, accuracy={sub['correct'].mean():.4f}")

    # Step 1b: Compute fast_confidence for all images
    print("\n[Step 1b] Computing fast_confidence for all 250 images (local, no gateway)...")
    model, class_names, device = load_fast_model()

    # Get unique images (all strategies share the same 250 images)
    rule_df = df[df["strategy"] == "rule_based"].copy()
    # Normalize Windows backslash paths to forward slashes (WSL compat)
    rule_df["image_path"] = rule_df["image_path"].str.replace("\\", "/", regex=False)
    df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)
    image_paths = rule_df["image_path"].tolist()

    print(f"  Running Fast inference on {len(image_paths)} images...")
    fast_results = compute_fast_confidences(model, class_names, device, image_paths)
    fast_df = pd.DataFrame(fast_results)
    fast_df["image_path"] = image_paths

    print(f"  Computed confidence for {fast_df['fast_confidence'].notna().sum()}/{len(image_paths)} images")
    print(f"  Fast overall accuracy (on these images): "
          f"{(fast_df['fast_predicted_class'] == rule_df['true_label'].values).mean():.4f}")

    # Step 2: Classify into zones
    print(f"\n[Step 2] Classifying images into zones (band=[{BAND_LOW}, {BAND_HIGH}])...")
    fast_df["zone"] = fast_df["fast_confidence"].apply(
        lambda c: "uncertain" if (c is not None and BAND_LOW <= c <= BAND_HIGH) else "confident"
    )
    zone_counts = fast_df["zone"].value_counts()
    print(f"  Confident zone (outside band): {zone_counts.get('confident', 0)} images")
    print(f"  Uncertain zone (inside band):  {zone_counts.get('uncertain', 0)} images")

    # Merge fast confidences with rule_based results
    rule_df = rule_df.merge(fast_df[["image_path", "fast_confidence", "fast_predicted_class", "zone"]],
                            on="image_path", how="left")

    # Merge with correctness_gate_router results
    gate_df = df[df["strategy"] == "correctness_gate_router"][["image_path", "correct", "predicted_class"]].copy()
    gate_df.rename(columns={"correct": "gate_correct", "predicted_class": "gate_predicted_class"}, inplace=True)
    merged = rule_df.merge(gate_df, on="image_path", how="left")

    # Step 3: 2x2 confusion table for confident zone only
    print("\n[Step 3] Building confusion table for CONFIDENT ZONE only...")
    confident = merged[merged["zone"] == "confident"].copy()
    n_confident = len(confident)
    print(f"  Confident-zone images: {n_confident}")

    # 2x2 table: rule_based correct/incorrect x gate correct/incorrect
    rb_correct = confident["correct"].astype(int)
    gate_correct = confident["gate_correct"].astype(int)

    both_right = ((rb_correct == 1) & (gate_correct == 1)).sum()
    both_wrong = ((rb_correct == 0) & (gate_correct == 0)).sum()
    rb_right_gate_wrong = ((rb_correct == 1) & (gate_correct == 0)).sum()
    rb_wrong_gate_right = ((rb_correct == 0) & (gate_correct == 1)).sum()

    print(f"\n  2x2 Confusion Table (confident zone, n={n_confident}):")
    print(f"  {'':30s}  gate_correct=1  gate_correct=0")
    print(f"  {'rule_based correct=1':30s}  {both_right:>13d}  {rb_right_gate_wrong:>13d}")
    print(f"  {'rule_based correct=0':30s}  {rb_wrong_gate_right:>13d}  {both_wrong:>13d}")

    # Build the full output rows for the CSV
    output_rows = []
    output_rows.append({
        "metric": "confident_zone_images",
        "value": n_confident,
    })
    output_rows.append({
        "metric": "uncertain_zone_images",
        "value": zone_counts.get("uncertain", 0),
    })
    output_rows.append({
        "metric": "band_low",
        "value": BAND_LOW,
    })
    output_rows.append({
        "metric": "band_high",
        "value": BAND_HIGH,
    })
    output_rows.append({
        "metric": "both_right",
        "value": int(both_right),
    })
    output_rows.append({
        "metric": "both_wrong",
        "value": int(both_wrong),
    })
    output_rows.append({
        "metric": "rb_right_gate_wrong",
        "value": int(rb_right_gate_wrong),
    })
    output_rows.append({
        "metric": "rb_wrong_gate_right",
        "value": int(rb_wrong_gate_right),
    })

    # Step 4: Print specific cases where gate beats rule_based in confident zone
    print("\n[Step 4] Cases where correctness_gate_router is RIGHT and rule_based is WRONG (confident zone):")
    gate_wins = confident[(confident["correct"] == 0) & (confident["gate_correct"] == 1)].copy()
    if len(gate_wins) == 0:
        print("  NONE -- rule_based never misses an image that the gate would have caught in the confident zone.")
    else:
        print(f"  Found {len(gate_wins)} cases:")
        for _, row in gate_wins.iterrows():
            print(f"    {row['image_path']}")
            fc = row['fast_confidence']
            print(f"      true_label={row['true_label']}, fast_confidence={fc if fc is not None else 'N/A'}")
            print(f"      rule_based predicted={row['predicted_class']} (wrong), "
                  f"gate predicted={row['gate_predicted_class']} (right)")
            output_rows.append({
                "metric": f"gate_wins_image",
                "value": row["image_path"],
                "true_label": row["true_label"],
                "fast_confidence": round(float(fc), 4) if fc is not None else None,
                "rb_predicted": row["predicted_class"],
                "gate_predicted": row["gate_predicted_class"],
            })

    # Also print cases where rule_based beats gate (gate wrong, rb right) in confident zone
    rb_wins = confident[(confident["correct"] == 1) & (confident["gate_correct"] == 0)].copy()
    print(f"\n  (Reverse: rule_based RIGHT, gate WRONG in confident zone: {len(rb_wins)} cases)")
    for _, row in rb_wins.iterrows():
        fc = row['fast_confidence']
        print(f"    {row['image_path']}")
        print(f"      true_label={row['true_label']}, fast_confidence={fc if fc is not None else 'N/A'}")
        print(f"      rule_based predicted={row['predicted_class']} (right), "
              f"gate predicted={row['gate_predicted_class']} (wrong)")

    # Step 5: Summary statistics
    print("\n[Step 5] Summary statistics (confident zone only):")
    rb_acc = confident["correct"].mean()
    gate_acc = confident["gate_correct"].mean()
    print(f"  rule_based accuracy:    {rb_acc:.4f} ({int(confident['correct'].sum())}/{n_confident})")
    print(f"  correctness_gate accuracy: {gate_acc:.4f} ({int(confident['gate_correct'].sum())}/{n_confident})")
    print(f"  Difference (gate - rb): {gate_acc - rb_acc:+.4f}")

    output_rows.append({
        "metric": "rule_based_accuracy_confident_zone",
        "value": round(float(rb_acc), 4),
    })
    output_rows.append({
        "metric": "gate_accuracy_confident_zone",
        "value": round(float(gate_acc), 4),
    })
    output_rows.append({
        "metric": "accuracy_difference",
        "value": round(float(gate_acc - rb_acc), 4),
    })

    # Full-dataset comparison (for reference)
    print("\n  Full-dataset reference (all 250 images):")
    rule_full = df[df["strategy"] == "rule_based"]
    gate_full = df[df["strategy"] == "correctness_gate_router"]
    print(f"  rule_based accuracy (full):     {rule_full['correct'].mean():.4f}")
    print(f"  correctness_gate accuracy (full): {gate_full['correct'].mean():.4f}")

    output_rows.append({
        "metric": "rule_based_accuracy_full",
        "value": round(float(rule_full["correct"].mean()), 4),
    })
    output_rows.append({
        "metric": "gate_accuracy_full",
        "value": round(float(gate_full["correct"].mean()), 4),
    })

    # Step 6: Save output
    print(f"\n[Step 6] Saving output to {OUTPUT_CSV}...")
    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Written {len(out_df)} rows to {OUTPUT_CSV}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
