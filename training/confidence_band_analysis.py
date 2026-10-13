"""
Fast confidence-score distribution analysis on the held-out set.

Measures the distribution of Fast's max-softmax confidence to pick an
empirically justified uncertainty band width around rule_based's 0.7
threshold for a future hybrid routing strategy.

This is a read-only analysis script -- no existing files are modified.

Outputs saved to monitoring/reports/:
  - confidence_band_analysis.csv (raw histogram + band captures)
"""
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image

from training.models.fast_cnn import FastCNN

BASE_DIR = Path(__file__).resolve().parent.parent
HELD_OUT_PATH = BASE_DIR / "router" / "held_out_images.csv"
FAST_CHECKPOINT = BASE_DIR / "training" / "checkpoints" / "fast_cnn_best.pt"
OUTPUT_DIR = BASE_DIR / "monitoring" / "reports"

FAST_NUM_CLASSES = 28
FAST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FAST_CLASS_NAMES = [
    "Apple__Healthy", "Apple__Rotten",
    "Banana__Healthy", "Banana__Rotten",
    "Bellpepper__Healthy", "Bellpepper__Rotten",
    "Carrot__Healthy", "Carrot__Rotten",
    "Cucumber__Healthy", "Cucumber__Rotten",
    "Grape__Healthy", "Grape__Rotten",
    "Guava__Healthy", "Guava__Rotten",
    "Jujube__Healthy", "Jujube__Rotten",
    "Mango__Healthy", "Mango__Rotten",
    "Orange__Healthy", "Orange__Rotten",
    "Pomegranate__Healthy", "Pomegranate__Rotten",
    "Potato__Healthy", "Potato__Rotten",
    "Strawberry__Healthy", "Strawberry__Rotten",
    "Tomato__Healthy", "Tomato__Rotten",
]

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

RULE_BASED_THRESHOLD = 0.70


def load_model():
    model = FastCNN(num_classes=FAST_NUM_CLASSES)
    ckpt = torch.load(FAST_CHECKPOINT, map_location=FAST_DEVICE)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.to(FAST_DEVICE)
    model.eval()
    print(f"Fast model loaded on {FAST_DEVICE}")
    return model


def run_inference(model, held_out_df):
    """Run Fast on every held-out image, return list of confidence scores."""
    confidences = []
    for _, row in held_out_df.iterrows():
        img_path = BASE_DIR / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        tensor = _preprocess(image).unsqueeze(0).to(FAST_DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence = float(torch.max(probs, dim=1)[0])
        confidences.append(confidence)

    return np.array(confidences)


def build_histogram(confidences):
    """Bucket confidences in 0.05 increments from 0 to 1."""
    bins = np.arange(0, 1.05, 0.05)
    counts, edges = np.histogram(confidences, bins=bins)
    total = len(confidences)

    rows = []
    for i, count in enumerate(counts):
        lo = round(edges[i], 2)
        hi = round(edges[i + 1], 2)
        rows.append({
            "bin_low": lo,
            "bin_high": hi,
            "count": int(count),
            "fraction": round(count / total, 4),
        })
    return rows, total


def compute_band_captures(confidences, threshold=RULE_BASED_THRESHOLD):
    """What fraction of requests fall within +/-N of the threshold."""
    total = len(confidences)
    captures = {}
    for delta in [0.05, 0.10, 0.15, 0.20]:
        lo = threshold - delta
        hi = threshold + delta
        mask = (confidences >= lo) & (confidences <= hi)
        frac = mask.sum() / total
        captures[delta] = round(float(frac), 4)
    return captures


def recommend_band(captures):
    """Pick the band capturing 15-30% of traffic."""
    for delta in [0.05, 0.10, 0.15, 0.20]:
        frac = captures[delta]
        if 0.10 <= frac <= 0.35:
            return delta, frac
    # fallback: closest to 20%
    best = min(captures.items(), key=lambda x: abs(x[1] - 0.20))
    return best[0], best[1]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    held_out_df = pd.read_csv(HELD_OUT_PATH)
    # Normalize Windows backslashes to forward slashes for WSL/Linux
    held_out_df["image_path"] = held_out_df["image_path"].str.replace("\\", "/", regex=False)
    print(f"Loaded {len(held_out_df)} held-out images")

    # Run inference
    model = load_model()
    confidences = run_inference(model, held_out_df)
    print(f"Computed {len(confidences)} confidence scores")
    print(f"  min={confidences.min():.4f}  max={confidences.max():.4f}  "
          f"mean={confidences.mean():.4f}  median={np.median(confidences):.4f}")

    # Build histogram
    hist_rows, total = build_histogram(confidences)

    print(f"\n{'='*55}")
    print(f"FAST CONFIDENCE DISTRIBUTION ({total} images, 0.05 bins)")
    print(f"{'='*55}")
    print(f"{'Bin':>12} {'Count':>8} {'Fraction':>10}")
    print(f"{'-'*55}")
    for row in hist_rows:
        print(f"[{row['bin_low']:.2f}, {row['bin_high']:.2f}) {row['count']:>8} {row['fraction']:>10.4f}")
    print(f"{'='*55}")

    # Band captures
    captures = compute_band_captures(confidences)

    print(f"\nFraction of traffic within +/-N of threshold {RULE_BASED_THRESHOLD}:")
    print(f"{'-'*55}")
    for delta, frac in captures.items():
        pct = frac * 100
        print(f"  +/-{delta:.2f}  [{RULE_BASED_THRESHOLD - delta:.2f}, {RULE_BASED_THRESHOLD + delta:.2f}]"
              f"  ->  {frac:.4f}  ({pct:.1f}%)")

    # Recommendation
    rec_delta, rec_frac = recommend_band(captures)
    lo = RULE_BASED_THRESHOLD - rec_delta
    hi = RULE_BASED_THRESHOLD + rec_delta

    print(f"\n{'='*55}")
    print(f"RECOMMENDATION: band width = +/-{rec_delta:.2f}")
    print(f"  Range: [{lo:.2f}, {hi:.2f}]")
    print(f"  Captures: {rec_frac:.4f} ({rec_frac*100:.1f}% of traffic)")
    print(f"{'='*55}")

    # Save CSV
    csv_rows = []
    for row in hist_rows:
        csv_row = {**row, "threshold": RULE_BASED_THRESHOLD}
        for delta in [0.05, 0.10, 0.15, 0.20]:
            lo_b = RULE_BASED_THRESHOLD - delta
            hi_b = RULE_BASED_THRESHOLD + delta
            in_band = (confidences >= lo_b) & (confidences <= hi_b)
            csv_row[f"capture_+/-{delta:.2f}"] = round(float(in_band.sum() / total), 4)
        csv_rows.append(csv_row)

    # Add summary row
    summary = {
        "bin_low": "SUMMARY",
        "bin_high": "",
        "count": total,
        "fraction": 1.0,
        "threshold": RULE_BASED_THRESHOLD,
    }
    for delta in [0.05, 0.10, 0.15, 0.20]:
        summary[f"capture_+/-{delta:.2f}"] = captures[delta]
    summary["recommended_band_width"] = f"+/-{rec_delta:.2f}"
    summary["recommended_range"] = f"[{lo:.2f}, {hi:.2f}]"
    summary["recommended_capture"] = f"{rec_frac:.4f} ({rec_frac*100:.1f}%)"
    csv_rows.append(summary)

    out_df = pd.DataFrame(csv_rows)
    out_path = OUTPUT_DIR / "confidence_band_analysis.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
