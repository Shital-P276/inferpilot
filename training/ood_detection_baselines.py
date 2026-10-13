"""
training/ood_detection_baselines.py

Workstream C: OOD/shift detection baselines on the frozen iWildCam tiers
(read-only -- no training, no checkpoint modification).

Question: on a WILDS-substrate project (whose entire point is distribution
shift), can the tiers' own outputs *detect* shifted inputs? Two textbook,
zero-training baselines per tier (Hendrycks et al. MSP 2017; Liu et al.
Energy 2020):
  MSP:    max softmax probability (higher = more in-distribution)
  Energy: -logsumexp(logits);       we score -energy (higher = more ID)

ID set:  local capped holdout data/iwildcam_final_holdout_subset (3,556 imgs).
OOD set: synthetic corruptions of the SAME images (fully local, no download):
  gaussian_noise (sigma=25/50 on 0-255) and gaussian_blur (radius=2/5).
  Same-image pairing isolates shift response from content differences.

Metric: AUROC for ID-vs-OOD separation per tier x corruption x severity.
Also reports MSP/energy mean shift (ID mean vs OOD mean).

Run: source ~/venv_inferpilot/bin/activate && python training/ood_detection_baselines.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.models.iwildcam_fast_mobilenetv3small import build_fast_mobilenetv3small
from training.models.iwildcam_balanced_resnet18 import build_balanced_resnet18
from training.models.iwildcam_heavy_convnextbase import build_heavy_convnextbase

BASE_DIR = REPO_ROOT
CHECKPOINT_DIR = BASE_DIR / "training" / "checkpoints"
HOLDOUT_DIR = BASE_DIR / "data" / "iwildcam_final_holdout_subset"
HOLDOUT_MANIFEST = HOLDOUT_DIR / "manifest.csv"
OUT_CSV = BASE_DIR / "training" / "checkpoints" / "ood_detection_baselines.csv"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
SEED = 42

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

CORRUPTIONS = [
    ("gaussian_noise", 25),
    ("gaussian_noise", 50),
    ("gaussian_blur", 2),
    ("gaussian_blur", 5),
]


def make_transform(resize, crop):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(resize),
        v2.CenterCrop(crop),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def corrupt(pil_img, name, level, rng):
    if name == "gaussian_noise":
        arr = np.asarray(pil_img).astype(np.float32)
        arr = arr + rng.normal(0, level, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if name == "gaussian_blur":
        return pil_img.filter(ImageFilter.GaussianBlur(radius=level))
    raise ValueError(name)


class ShiftDataset(Dataset):
    """Clean (label 0=ID) or corrupted (label 1=OOD) holdout images."""

    def __init__(self, samples, transform, corruption=None):
        self.samples = samples
        self.transform = transform
        self.corruption = corruption  # None | (name, level)
        self.rng = np.random.default_rng(SEED)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        pil_img = Image.open(img_path).convert("RGB")
        is_ood = 0
        if self.corruption is not None:
            name, level = self.corruption
            # Per-index seeded RNG for reproducibility regardless of order
            rng = np.random.default_rng(SEED + idx)
            pil_img = corrupt(pil_img, name, level, rng)
            is_ood = 1
        return self.transform(pil_img), is_ood


@torch.no_grad()
def collect_scores(model, loader):
    msps, neg_energies, ood_flags = [], [], []
    for images, is_ood in tqdm(loader, desc="    batches", unit="batch", leave=False):
        logits = model(images.to(DEVICE))
        probs = torch.softmax(logits, dim=1)
        msps.extend(probs.max(dim=1).values.cpu().tolist())
        neg_energies.extend((-torch.logsumexp(logits, dim=1)).cpu().tolist())
        ood_flags.extend(is_ood.tolist())
    return (np.asarray(msps), np.asarray(neg_energies), np.asarray(ood_flags))


def main():
    print("=" * 70)
    print("Workstream C: OOD detection baselines (MSP + Energy), frozen tiers")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print("ID: capped holdout (clean); OOD: synthetic corruptions of same images.")

    with open(HOLDOUT_MANIFEST, newline="") as f:
        reader = csv.DictReader(f)
        samples = [(HOLDOUT_DIR / r["image_path"].replace("\\", "/")).resolve()
                   for r in reader]
    print(f"Holdout images: {len(samples)}")

    rows = []
    for tier, cfg in TIER_CONFIG.items():
        print(f"\n--- Tier: {tier} ---")
        ckpt = torch.load(cfg["checkpoint"], map_location=DEVICE)
        n_classes = ckpt.get("num_classes", 182)
        model = cfg["builder"](num_classes=n_classes)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        transform = make_transform(cfg["resize"], cfg["crop"])
        kwargs = {"batch_size": BATCH_SIZE, "shuffle": False, "num_workers": 2,
                  "pin_memory": DEVICE.type == "cuda"}

        id_loader = DataLoader(ShiftDataset(samples, transform), **kwargs)
        print("  ID (clean)...")
        id_msp, id_en, _ = collect_scores(model, id_loader)

        for cname, level in CORRUPTIONS:
            print(f"  OOD ({cname}={level})...")
            ood_loader = DataLoader(
                ShiftDataset(samples, transform, corruption=(cname, level)), **kwargs)
            ood_msp, ood_en, _ = collect_scores(model, ood_loader)
            y = np.concatenate([np.zeros_like(id_msp), np.ones_like(ood_msp)])
            # Higher score = more ID, so AUROC uses negated-to-ID convention:
            # P(score_ID > score_OOD) = roc_auc_score(y, -score)
            msp_auc = float(roc_auc_score(y, -np.concatenate([id_msp, ood_msp])))
            en_auc = float(roc_auc_score(y, -np.concatenate([id_en, ood_en])))
            print(f"    MSP AUROC={msp_auc:.4f} (ID mean={id_msp.mean():.3f} "
                  f"OOD mean={ood_msp.mean():.3f}) | Energy AUROC={en_auc:.4f} "
                  f"(ID mean={id_en.mean():.3f} OOD mean={ood_en.mean():.3f})")
            rows.append({"tier": tier, "corruption": cname, "severity": level,
                         "n_id": len(id_msp), "n_ood": len(ood_msp),
                         "msp_auroc": round(msp_auc, 4),
                         "energy_auroc": round(en_auc, 4),
                         "msp_id_mean": round(float(id_msp.mean()), 4),
                         "msp_ood_mean": round(float(ood_msp.mean()), 4)})
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")
    print("\n" + out.to_string(index=False))

    print("\n" + "=" * 70)
    best = out.loc[out["msp_auroc"].idxmax()]
    print(f"Best separation: {best['tier']} / {best['corruption']}={best['severity']} "
          f"MSP AUROC={best['msp_auroc']:.4f}")
    for m in ["msp_auroc", "energy_auroc"]:
        piv = out.pivot_table(index="tier", columns="corruption", values=m,
                              aggfunc="mean")
        print(f"\nMean {m} by tier (averaged over severities):")
        print(piv.round(4).to_string())
    print("VERDICT: if Heavy's AUROCs exceed Fast's, shift-detection improves with "
          "capacity (consistent with the macro-F1 finding); if flat, detection is "
          "capacity-independent on these corruptions -- report either way.")
    print("=" * 70)


if __name__ == "__main__":
    main()
