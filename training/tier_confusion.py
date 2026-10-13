"""
training/tier_confusion.py

Phase 1 figure task: tier-confusion analysis for the report.
Runs all 3 frozen tier checkpoints over the 3,556-image capped holdout
(same inference pattern as training/gradcam_iwildcam.py) and outputs:
  - top-15 confused species pairs per tier (true vs predicted, count, %)
  - one 182x182 confusion heatmap PNG per tier (log1p-normalized counts;
    appendix-scale figure, not main body)

Outputs to reports/figures/tier_confusion/:
  {tier}_top15_confused.csv, {tier}_confusion_182.png, confusion_summary.csv

Run: source ~/venv_inferpilot/bin/activate && python training/tier_confusion.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.models.iwildcam_fast_mobilenetv3small import build_fast_mobilenetv3small
from training.models.iwildcam_balanced_resnet18 import build_balanced_resnet18
from training.models.iwildcam_heavy_convnextbase import build_heavy_convnextbase

CHECKPOINT_DIR = REPO_ROOT / "training" / "checkpoints"
HOLDOUT_DIR = REPO_ROOT / "data" / "iwildcam_final_holdout_subset"
HOLDOUT_MANIFEST = HOLDOUT_DIR / "manifest.csv"
OUT_DIR = REPO_ROOT / "reports" / "figures" / "tier_confusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
N_CLASSES = 182

TIER_CONFIG = {
    "fast": {"builder": build_fast_mobilenetv3small,
             "checkpoint": CHECKPOINT_DIR / "iwildcam_fast_mobilenetv3small_best.pt",
             "resize": 256, "crop": 224},
    "balanced": {"builder": build_balanced_resnet18,
                 "checkpoint": CHECKPOINT_DIR / "iwildcam_balanced_resnet18_best.pt",
                 "resize": 293, "crop": 256},
    "heavy": {"builder": build_heavy_convnextbase,
              "checkpoint": CHECKPOINT_DIR / "iwildcam_heavy_convnextbase_best.pt",
              "resize": 438, "crop": 384},
}


class ManifestImages(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        return self.transform(Image.open(img_path).convert("RGB")), label


def main():
    print("=" * 70)
    print("Phase 1: tier-confusion analysis (3 tiers x 3,556 holdout)")
    print("=" * 70)
    with open(HOLDOUT_MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    samples = [((HOLDOUT_DIR / r["image_path"].replace("\\", "/")).resolve(),
                int(r["label"])) for r in rows]
    print(f"Holdout: {len(samples)} images")

    summary = []
    for tier, cfg in TIER_CONFIG.items():
        print(f"\n--- {tier} ---")
        ckpt = torch.load(cfg["checkpoint"], map_location=DEVICE)
        n_classes = ckpt.get("num_classes", N_CLASSES)
        class_names = ckpt.get("class_names", [f"class_{i}" for i in range(n_classes)])
        model = cfg["builder"](num_classes=n_classes)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        tf = v2.Compose([
            v2.ToImage(), v2.Resize(cfg["resize"]), v2.CenterCrop(cfg["crop"]),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        loader = DataLoader(ManifestImages(samples, tf), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2,
                            pin_memory=DEVICE.type == "cuda")
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="    infer", unit="batch",
                                       leave=False):
                y_pred.extend(model(images.to(DEVICE)).argmax(dim=1).cpu().tolist())
                y_true.extend(labels.tolist())
        y_true = np.asarray(y_true, dtype=np.intp)
        y_pred = np.asarray(y_pred, dtype=np.intp)
        acc = float((y_true == y_pred).mean())
        print(f"  accuracy: {acc:.4f}")

        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        np.add.at(cm, (y_true, y_pred), 1)

        # Top-15 confused pairs (off-diagonal, by count)
        pairs = []
        for t in range(N_CLASSES):
            for p in range(N_CLASSES):
                if t != p and cm[t, p] > 0:
                    pairs.append((t, p, int(cm[t, p])))
        pairs.sort(key=lambda x: -x[2])
        top15 = pairs[:15]
        support = cm.sum(axis=1)
        with open(OUT_DIR / f"{tier}_top15_confused.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true_id", "true_name", "pred_id", "pred_name",
                        "count", "pct_of_true_class"])
            for t, p, c in top15:
                tn = class_names[t] if t < len(class_names) else str(t)
                pn = class_names[p] if p < len(class_names) else str(p)
                pct = 100.0 * c / max(support[t], 1)
                w.writerow([t, tn, p, pn, c, round(pct, 1)])
        print(f"  top confusion: true={class_names[top15[0][0]]} -> "
              f"pred={class_names[top15[0][1]]} (n={top15[0][2]})")
        print(f"  off-diagonal pairs: {len(pairs)}; "
              f"zero-diagonal classes: {int((np.diag(cm) == 0).sum())}/{N_CLASSES}")

        # 182x182 heatmap (log1p counts; appendix figure)
        fig, ax = plt.subplots(figsize=(7.16, 6.5))
        im = ax.imshow(np.log1p(cm), aspect="auto", cmap="magma")
        ax.set_xlabel("Predicted class id")
        ax.set_ylabel("True class id")
        ax.set_title(f"{tier}: confusion (log1p counts, n={len(samples)}, "
                     f"acc={acc:.3f})")
        fig.colorbar(im, ax=ax, label="log1p(count)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{tier}_confusion_182.png", dpi=300)
        plt.close(fig)
        summary.append({"tier": tier, "n": len(samples),
                        "accuracy": round(acc, 4),
                        "offdiag_pairs": len(pairs),
                        "zero_diag_classes": int((np.diag(cm) == 0).sum())})
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    with open(OUT_DIR / "confusion_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nSaved to {OUT_DIR} "
          f"({len(summary) * 2 + 1} files: 3 top15 CSVs + 3 heatmaps + summary)")


if __name__ == "__main__":
    main()
