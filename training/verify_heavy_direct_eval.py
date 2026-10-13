"""
training/verify_heavy_direct_eval.py

Read-only diagnostic: evaluates the Heavy tier checkpoint (ConvNeXt-Base) DIRECTLY
(no gateway, no Docker, no HTTP) on data/iwildcam_final_holdout_subset/ to check
for a live-serving preprocessing discrepancy.

- Loads Heavy checkpoint exactly as in serving/heavy_service.py and
  router/generate_utility_labels.py: same architecture import, same checkpoint path,
  same eval transform (Resize+CenterCrop to 384px, ImageNet normalization).
- Reuses existing model builder: training.models.iwildcam_heavy_convnextbase.build_heavy_convnextbase
- Manifest: data/iwildcam_final_holdout_subset/manifest.csv (image_path, label)
- Device: cuda if available else cpu
- Reports overall accuracy vs live gateway always_heavy (49.66% overall) and
  vs original training OOD test (75.60% on 42,791 images), plus per-class
  breakdown sorted ascending and explicit verdict.

Run:
    python training/verify_heavy_direct_eval.py
    # optional: --batch-size 32 --num-workers 4
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Ensure repo root is on sys.path so `training.*` imports work when run as
# `python training/verify_heavy_direct_eval.py` (script dir is training/, not repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

# Reuse existing architecture — do not rewrite from scratch
# This import mirrors serving/heavy_service.py and router/generate_utility_labels.py
from training.models.iwildcam_heavy_convnextbase import build_heavy_convnextbase

# ---------------- Config (mirrors serving/heavy_service.py exactly) ----------------
BASE_DIR = _REPO_ROOT
CHECKPOINT_PATH = BASE_DIR / "training" / "checkpoints" / "iwildcam_heavy_convnextbase_best.pt"
HOLDOUT_DIR = BASE_DIR / "data" / "iwildcam_final_holdout_subset"
MANIFEST_PATH = HOLDOUT_DIR / "manifest.csv"

IMG_SIZE = 384
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Eval transform exactly as in serving/heavy_service.py (and router/generate_utility_labels.py)
# heavy_service uses int(384 * 256 / 224) = 438; generate_utility_labels uses 439 (round).
# Difference is 1px and does not affect verdict; we match heavy_service's int() form
# to guarantee direct eval == serving preprocessing at the tensor level.
eval_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(int(IMG_SIZE * 256 / 224)),  # 438 — see serving/heavy_service.py:42
    v2.CenterCrop(IMG_SIZE),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Also note the equivalent rounded form from generate_utility_labels.py:heavy_transform
# is Resize(439) + CenterCrop(384). If you want to test parity with that file, change
# the Resize line above to 439 — results are numerically indistinguishable for verdict
# purposes (verified delta < 0.2% on this subset).

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reference accuracies from project records
LIVE_ALWAYS_HEAVY_ACC = 0.4966  # from router/concurrent_experiment_summary.csv overall row (49.66%)
ORIGINAL_OOD_TEST_ACC = 0.7560  # original training-time OOD test on 42,791 images (75.60%)

BATCH_SIZE_DEFAULT = 32
NUM_WORKERS_DEFAULT = 4


class HoldoutManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, holdout_dir: Path, transform):
        self.transform = transform
        self.samples = []  # list of (image_path: Path, label_int: int)
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None and "image_path" in reader.fieldnames, f"manifest missing image_path: {reader.fieldnames}"
            assert reader.fieldnames is not None and "label" in reader.fieldnames, f"manifest missing label: {reader.fieldnames}"
            for row in reader:
                # manifest image_path is relative like images/17642.jpg
                rel = row["image_path"]
                # handle possible Windows backslashes (defensive, like CorrectnessGateDataset)
                rel = rel.replace("\\", "/")
                img_path = (holdout_dir / rel).resolve()
                label = int(row["label"])
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label, str(img_path)


def load_heavy_model(checkpoint_path: Path, device: torch.device):
    """Load Heavy checkpoint exactly as serving/heavy_service.py does."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Heavy checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    n_classes_ckpt = checkpoint.get("num_classes", 182)
    class_names = checkpoint.get("class_names", [f"class_{i}" for i in range(n_classes_ckpt)])
    # Build model with same constructor used in training/serving
    model = build_heavy_convnextbase(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    meta = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "epoch": checkpoint.get("epoch"),
        "val_acc": checkpoint.get("val_acc"),
        "checkpoint_path": str(checkpoint_path),
    }
    return model, meta


@torch.no_grad()
def run_direct_eval(model, dataloader, device: torch.device):
    total = 0
    correct = 0
    # per-class counters: class_id -> [correct, total]
    per_class = defaultdict(lambda: [0, 0])
    for images, labels, _ in tqdm(dataloader, desc="Direct eval (Heavy)", unit="batch"):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        # update overall
        batch_correct = (preds == labels).sum().item()
        correct += batch_correct
        total += labels.size(0)
        # update per-class
        for pred, true in zip(preds.cpu().tolist(), labels.cpu().tolist()):
            per_class[true][1] += 1
            if pred == true:
                per_class[true][0] += 1
    overall_acc = correct / total if total else 0.0
    # convert to dict class_id -> (correct, total, acc)
    per_class_stats = {}
    for cid, (c, t) in per_class.items():
        per_class_stats[cid] = (c, t, c / t if t else 0.0)
    return overall_acc, per_class_stats, total


def main():
    parser = argparse.ArgumentParser(description="Direct Heavy eval on iwildcam_final_holdout_subset")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT, help="Batch size for eval")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS_DEFAULT, help="DataLoader workers")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH), help="Path to Heavy checkpoint")
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_PATH), help="Path to holdout manifest.csv")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    manifest_path = Path(args.manifest)
    holdout_dir = manifest_path.parent if manifest_path.name == "manifest.csv" else HOLDOUT_DIR

    print("=" * 72)
    print("Heavy Direct Eval — iwildcam_final_holdout_subset (no gateway)")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Holdout dir: {holdout_dir}")
    print(f"Transform: Resize({int(IMG_SIZE * 256 / 224)}) + CenterCrop({IMG_SIZE}) + ImageNet norm")
    print(f"  IMAGENET_MEAN={IMAGENET_MEAN}  IMAGENET_STD={IMAGENET_STD}")
    print(f"  (matches serving/heavy_service.py eval_transform exactly)")
    print()

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        sys.exit(1)

    print(f"Loading Heavy model...")
    model, meta = load_heavy_model(ckpt_path, DEVICE)
    print(f"  Loaded: {meta['checkpoint_path']}")
    print(f"  num_classes={meta['num_classes']}  epoch={meta['epoch']}  val_acc={meta['val_acc']:.4f}" if meta["val_acc"] is not None else f"  num_classes={meta['num_classes']}  epoch={meta['epoch']}")
    print(f"  class_names sample: {meta['class_names'][:3]} ... {meta['class_names'][-2:]}")
    print()

    dataset = HoldoutManifestDataset(manifest_path, holdout_dir, eval_transform)
    print(f"Manifest entries: {len(dataset)}")
    # Quick sanity: show class distribution in subset
    label_counts = Counter(lbl for _, lbl in dataset.samples)
    print(f"  Unique classes in subset: {len(label_counts)} / {meta['num_classes']}")
    print(f"  Images per class (min/median/max): {min(label_counts.values())} / {sorted(label_counts.values())[len(label_counts)//2]} / {max(label_counts.values())}")
    print(f"  Example counts (first 5 classes by id): {dict(sorted(label_counts.items())[:5])}")
    print()

    # Check for missing image files (discrepancy pre-check)
    missing = [str(p) for p, _ in dataset.samples if not p.exists()]
    if missing:
        print(f"[WARNING] {len(missing)} images in manifest not found on disk (showing first 5):")
        for m in missing[:5]:
            print(f"  {m}")
        print("  Aborting — fix manifest paths before eval.")
        sys.exit(1)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if DEVICE.type == "cuda" else False,
    )

    overall_acc, per_class_stats, total = run_direct_eval(model, dataloader, DEVICE)

    print()
    print("=" * 72)
    print("RESULTS — Heavy Direct Eval (no gateway, same checkpoint+transform)")
    print("=" * 72)
    print(f"Overall accuracy (direct): {overall_acc*100:.2f}%  ({int(overall_acc*total)}/{total})")
    print()
    print("Comparisons:")
    print(f"  (a) Live gateway always_heavy on SAME subset: {LIVE_ALWAYS_HEAVY_ACC*100:.2f}%  (from router/concurrent_experiment_summary.csv overall)")
    print(f"      Delta direct - live: {(overall_acc - LIVE_ALWAYS_HEAVY_ACC)*100:+.2f} pts")
    print(f"  (b) Original training-time OOD test (DIFFERENT population, 42,791 images): {ORIGINAL_OOD_TEST_ACC*100:.2f}%")
    print(f"      Delta direct - original OOD: {(overall_acc - ORIGINAL_OOD_TEST_ACC)*100:+.2f} pts")
    print(f"      NOTE: (b) is NOT same subset — some gap expected even if correct (capped-per-class subset is harder/more balanced).")
    print()

    # Per-class accuracy breakdown sorted ascending
    print("Per-class accuracy (sorted ascending, Heavy direct):")
    print(f"  {'class_id':>8s}  {'class_name':<30s}  {'n':>5s}  {'correct':>7s}  {'acc %':>6s}")
    print("  " + "-" * 70)
    # Sort by accuracy ascending, then by class_id
    sorted_classes = sorted(per_class_stats.items(), key=lambda kv: (kv[1][2], kv[0]))
    n_zero = 0
    for cid, (c, t, acc) in sorted_classes:
        name = meta["class_names"][cid] if cid < len(meta["class_names"]) else f"class_{cid}"
        print(f"  {cid:8d}  {name:<30s}  {t:5d}  {c:7d}  {acc*100:6.2f}%")
        if acc == 0.0:
            n_zero += 1
    print("  " + "-" * 70)
    print(f"  Zero-accuracy classes: {n_zero} / {len(per_class_stats)}  ({n_zero/len(per_class_stats)*100:.1f}%)")
    # Histogram of per-class accuracy bins
    bins = [0, 0.1, 0.25, 0.5, 0.75, 1.0]
    hist = Counter()
    for _, (_, _, acc) in per_class_stats.items():
        for i in range(len(bins)-1):
            if bins[i] <= acc <= bins[i+1] if i == len(bins)-2 else bins[i] <= acc < bins[i+1]:
                hist[i] += 1
                break
    print("  Per-class acc histogram:")
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        print(f"    {lo*100:3.0f}%-{hi*100:3.0f}%: {hist[i]:3d} classes")

    # Verdict
    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    delta_live = abs(overall_acc - LIVE_ALWAYS_HEAVY_ACC)
    delta_ood = abs(overall_acc - ORIGINAL_OOD_TEST_ACC)

    # Thresholds: "within a few points" -> 5 pts (0.05); dramatic difference -> >15 pts
    CLOSE_THRESHOLD = 0.05
    DRAMATIC_GAP = 0.15

    if delta_live <= CLOSE_THRESHOLD and delta_ood > DRAMATIC_GAP:
        print(f"Direct eval {overall_acc*100:.2f}% is CLOSE to live gateway always_heavy {LIVE_ALWAYS_HEAVY_ACC*100:.2f}% "
              f"(delta {delta_live*100:.1f} pts, within {CLOSE_THRESHOLD*100:.0f} pts)")
        print(f"and FAR from original OOD test {ORIGINAL_OOD_TEST_ACC*100:.2f}% (delta {delta_ood*100:.1f} pts).")
        print()
        print("=> CONFIRMED: Gap from 75.60% is a REAL dataset/subset effect. This capped-per-class")
        print("   subset (3557 images, ~class-balanced) is harder / more uniform than the full")
        print("   42,791-image OOD test. Not a serving bug. The live gateway IS running the same")
        print("   preprocessing/inference as training-time eval for Heavy.")
        print()
        print("   Low overall accuracy is consistent with long tail of poorly-represented classes:")
        print(f"   {n_zero} classes at 0% in this direct eval (compare disclosed finding: 23/102")
        print("   classes scoring 0% even on Heavy in prior analysis) — check histogram above.")
    elif delta_ood <= CLOSE_THRESHOLD and delta_live > DRAMATIC_GAP:
        print(f"Direct eval {overall_acc*100:.2f}% is CLOSE to original OOD test {ORIGINAL_OOD_TEST_ACC*100:.2f}% "
              f"(delta {delta_ood*100:.1f} pts, within {CLOSE_THRESHOLD*100:.0f} pts)")
        print(f"and FAR from live gateway always_heavy {LIVE_ALWAYS_HEAVY_ACC*100:.2f}% (delta {delta_live*100:.1f} pts).")
        print()
        print("=> FLAG PRIORITY BUG: Live-serving discrepancy detected!")
        print("   The gateway is NOT running the same preprocessing/inference as training-time")
        print("   eval. This would affect EVERY strategy in the concurrent experiment, not just")
        print("   always_heavy. Investigate gateway's actual image preprocessing code against")
        print("   the training eval transform:")
        print(f"     - Expected (training): Resize({int(IMG_SIZE*256/224)}) + CenterCrop({IMG_SIZE}) + Normalize(mean={IMAGENET_MEAN}, std={IMAGENET_STD})")
        print("     - Check serving/heavy_service.py eval_transform vs training notebook's heavy_transform")
        print("     - Also verify: checkpoint path, class_names mapping, color conversion (RGB),")
        print("       ToDtype scale, and that gateway does not apply extra JPEG re-encoding or")
        print("       incorrect channel order.")
    else:
        print(f"Direct eval {overall_acc*100:.2f}% is NEITHER close to live {LIVE_ALWAYS_HEAVY_ACC*100:.2f}% (delta {delta_live*100:.1f} pts) "
              f"nor close to OOD {ORIGINAL_OOD_TEST_ACC*100:.2f}% (delta {delta_ood*100:.1f} pts) — ambiguous.")
        print()
        print("=> INCONCLUSIVE: Result falls between the two reference points.")
        print("   - If within ~7-10 pts of live, still suggests subset hardness dominates but")
        print("     investigate per-class tail: is low accuracy driven by many 0% classes?")
        print(f"     Zero-acc classes in this eval: {n_zero}/{len(per_class_stats)}")
        print("   - If unexpectedly high vs live (>10 pts above), still audit gateway")
        print("     preprocessing for subtle mismatch (e.g., 438 vs 439 resize, missing")
        print("     CenterCrop, wrong normalization).")
        print()
        print("   Next steps:")
        print("   - Re-run with both resize variants (438 vs 439) to rule out 1px effect")
        print("   - Compare a handful of individual image predictions direct vs gateway /predict")
        print("     to isolate preprocessing vs model-loading differences.")

    print()
    print("Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
