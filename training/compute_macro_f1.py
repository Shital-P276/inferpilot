"""
training/compute_macro_f1.py

Read-only diagnostic: computes macro-F1 (and per-class F1) for all three
iWildCam tiers (Fast / Balanced / Heavy) on two eval sets.

Tier details — reuses EXACT existing construction + checkpoint + eval transform:
  Fast:     MobileNetV3-Small, 224px, checkpoint iwildcam_fast_mobilenetv3small_best.pt
            transform: Resize(256) + CenterCrop(224) + Normalize (see serving/fast_service.py,
                       router/generate_utility_labels.py:fast_transform)
  Balanced: ResNet-18, 256px, checkpoint iwildcam_balanced_resnet18_best.pt
            transform: Resize(293) + CenterCrop(256) (see generate_utility_labels.py:balanced_transform)
  Heavy:    ConvNeXt-Base, 384px, checkpoint iwildcam_heavy_convnextbase_best.pt
            transform: Resize(int(384*256/224))=438 + CenterCrop(384) (see serving/heavy_service.py,
                       also generate_utility_labels.py:heavy_transform Resize 439). Reused from
                       training/verify_heavy_direct_eval.py.

Two evaluation sets:
  (a) FULL WILDS OOD test (42,791 images) via wilds.get_dataset(dataset="iwildcam").get_subset("test")
      exactly as in original training notebook. Reuses wilds library if installed.
  (b) Capped-per-class holdout data/iwildcam_final_holdout_subset/manifest.csv (3,556 images)

Metrics per tier x set:
  - Overall accuracy (sanity vs 49.87/58.03/75.60 for full OOD, 52.45 for Heavy on capped)
  - Macro-F1, Weighted-F1 via sklearn
  - Per-class summary (top5/worst5 by F1) to stdout; full CSV per tier/set to training/macro_f1_reports/

Final summary table (6 rows) + WILDS baseline comparison.

Read-only — no checkpoint modification, no retraining. Heavy @384px is slowest.

Run:
  python training/compute_macro_f1.py
  python training/compute_macro_f1.py --batch-size 32 --num-workers 4 --skip-full-ood  # dev quick
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

# Reuse existing model builders — do not redefine architectures inline
from training.models.iwildcam_fast_mobilenetv3small import build_fast_mobilenetv3small
from training.models.iwildcam_balanced_resnet18 import build_balanced_resnet18
from training.models.iwildcam_heavy_convnextbase import build_heavy_convnextbase

# sklearn for F1
from sklearn.metrics import f1_score, classification_report, accuracy_score

# ---------------- Config ----------------
BASE_DIR = _REPO_ROOT
CHECKPOINT_DIR = BASE_DIR / "training" / "checkpoints"
HOLDOUT_DIR = BASE_DIR / "data" / "iwildcam_final_holdout_subset"
HOLDOUT_MANIFEST = HOLDOUT_DIR / "manifest.csv"
REPORT_DIR = BASE_DIR / "training" / "macro_f1_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Eval transforms — EXACT reuse from serving/* and router/generate_utility_labels.py
fast_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),            # 224 * 256/224  — fast_service.py: int(224*256/224)=256
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

balanced_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(293),            # 256 * 256/224 ≈292.57 → 293  — generate_utility_labels.py:293
    v2.CenterCrop(256),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

heavy_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(int(384 * 256 / 224)),  # 438 — serving/heavy_service.py:42  (generate_utility_labels.py uses 439 round)
    v2.CenterCrop(384),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

TIER_CONFIG = {
    "fast": {
        "builder": build_fast_mobilenetv3small,
        "checkpoint": CHECKPOINT_DIR / "iwildcam_fast_mobilenetv3small_best.pt",
        "transform": fast_transform,
        "img_size": 224,
        "expected_full_ood_acc": 0.4987,
    },
    "balanced": {
        "builder": build_balanced_resnet18,
        "checkpoint": CHECKPOINT_DIR / "iwildcam_balanced_resnet18_best.pt",
        "transform": balanced_transform,
        "img_size": 256,
        "expected_full_ood_acc": 0.5803,
    },
    "heavy": {
        "builder": build_heavy_convnextbase,
        "checkpoint": CHECKPOINT_DIR / "iwildcam_heavy_convnextbase_best.pt",
        "transform": heavy_transform,
        "img_size": 384,
        "expected_full_ood_acc": 0.7560,
    },
}

# WILDS baseline published (ERM) — per task prompt
WILDS_ERM_OOD_ACC = 0.716
WILDS_ERM_OOD_MACRO_F1 = 0.310  # reported as 31.0

# Holdout expected (verified via verify_heavy_direct_eval.py)
CAPPED_HEAVY_EXPECTED = 0.5245

BATCH_SIZE_DEFAULT = 32
NUM_WORKERS_DEFAULT = 4


class HoldoutManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, holdout_dir: Path, transform):
        self.transform = transform
        self.samples = []
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None and "image_path" in reader.fieldnames
            assert reader.fieldnames is not None and "label" in reader.fieldnames
            for row in reader:
                rel = row["image_path"].replace("\\", "/")
                img_path = (holdout_dir / rel).resolve()
                label = int(row["label"])
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label


def load_model_for_tier(tier: str, device: torch.device):
    cfg = TIER_CONFIG[tier]
    ckpt_path = cfg["checkpoint"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for tier={tier}: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    n_classes = checkpoint.get("num_classes", 182)
    class_names = checkpoint.get("class_names", [f"class_{i}" for i in range(n_classes)])
    builder = cfg["builder"]
    model = builder(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    meta = {
        "tier": tier,
        "checkpoint": str(ckpt_path),
        "class_names": class_names,
        "num_classes": len(class_names),
        "val_acc": checkpoint.get("val_acc"),
        "epoch": checkpoint.get("epoch"),
        "img_size": cfg["img_size"],
    }
    return model, meta


@torch.no_grad()
def evaluate_dataloader(model, dataloader, device: torch.device, desc: str):
    """Run inference over dataloader, return y_true, y_pred lists."""
    y_true_all = []
    y_pred_all = []
    for images, labels in tqdm(dataloader, desc=desc, unit="batch"):
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().tolist()
        trues = labels.tolist() if isinstance(labels, torch.Tensor) else list(labels)
        y_true_all.extend(trues)
        y_pred_all.extend(preds)
    return y_true_all, y_pred_all


def compute_metrics(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    # per-class report dict
    report = classification_report(
        y_true, y_pred,
        target_names=class_names[: max(max(y_true, default=0), max(y_pred, default=0)) + 1] if len(class_names) > 0 else None,
        output_dict=True,
        zero_division=0,
    )
    # Extract per-class F1 for sorting; keys are class name strings, but we map to ids
    per_class = []
    for cid, name in enumerate(class_names):
        r = report.get(name)
        if r is None:
            # class not in report (no support in y_true nor y_pred) -> zero
            per_class.append((cid, name, 0.0, 0.0, 0.0, 0))
        else:
            per_class.append((cid, name, r["precision"], r["recall"], r["f1-score"], int(r["support"])))
    return acc, macro_f1, weighted_f1, report, per_class


def save_per_class_csv(per_class, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "precision", "recall", "f1", "support"])
        for cid, name, prec, rec, f1, sup in per_class:
            w.writerow([cid, name, f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", sup])


def print_per_class_summary(per_class, tier, eval_set_name, report_path: Path):
    # Sort by F1 ascending for worst, descending for best
    sorted_by_f1 = sorted(per_class, key=lambda x: x[4])
    worst5 = sorted_by_f1[:5]
    best5 = sorted(sorted_by_f1, key=lambda x: x[4], reverse=True)[:5]
    print(f"  Per-class F1 summary for {tier} / {eval_set_name} (full report: {report_path}):")
    print(f"    Worst 5 by F1:")
    for cid, name, prec, rec, f1, sup in worst5:
        print(f"      {cid:3d} {name:<30s}  F1={f1:.3f}  P={prec:.3f} R={rec:.3f}  sup={sup}")
    print(f"    Best 5 by F1:")
    for cid, name, prec, rec, f1, sup in best5:
        print(f"      {cid:3d} {name:<30s}  F1={f1:.3f}  P={prec:.3f} R={rec:.3f}  sup={sup}")


def try_load_wilds_full_ood(transform, root_dir: Path | None = None):
    """
    Attempt to load FULL WILDS OOD test set via wilds.get_dataset.
    Returns (dataset, n_images) or raises informative error.
    This mirrors original training notebook's wilds.get_dataset(...).get_subset("test").
    """
    try:
        from wilds import get_dataset
        from wilds.common.data.datasets import WILDSDataset
    except ImportError as e:
        raise RuntimeError(f"wilds library not installed: {e}. Install with `pip install wilds`") from e

    # Determine root_dir: default wilds cache location if not specified
    if root_dir is None:
        # Use project-local cache if desired, else default ~/wilds or wilds default
        # Let wilds use its default (usually <root>/data or ~/wilds)
        # We'll try with download=True and let wilds decide; also try explicit BASE_DIR/data/wilds
        pass

    # Try common WILDS dataset identifiers for iWildCam
    candidates = ["iwildcam", "iwildcam_v2.0", "iwildcam_v1.0"]
    last_err = None
    for ds_name in candidates:
        for try_root in [None, BASE_DIR / "data", BASE_DIR / "data" / "wilds", Path.home() / "wilds"]:
            try:
                kwargs = {"dataset": ds_name, "download": True}
                if try_root is not None:
                    kwargs["root_dir"] = str(try_root)
                dataset = get_dataset(**kwargs)
                # WILDS iWildCam: test split contains OOD test; some versions use "test" with metadata split
                # Try get_subset("test") first; if that fails try with split param.
                try:
                    subset = dataset.get_subset("test", transform=transform)
                except TypeError:
                    # older wilds API: get_subset(split="test", transform=...)
                    subset = dataset.get_subset(split="test", transform=transform)
                # Verify size close to 42,791
                n = len(subset)
                print(f"  WILDS loaded: dataset={ds_name} root={try_root or 'default'}  subset=test  n={n}")
                return subset, n
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Failed to load WILDS iWildCam test subset after trying candidates {candidates}: {last_err}")


def build_wilds_dataloader(subset, batch_size: int, num_workers: int):
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if DEVICE.type == "cuda" else False)


def evaluate_tier_on_wilds(tier: str, model, meta, batch_size: int, num_workers: int):
    transform = TIER_CONFIG[tier]["transform"]
    print(f"\n--- {tier.upper()} / FULL WILDS OOD test (42,791) ---")
    print(f"  Transform: Resize({ '256' if tier=='fast' else '293' if tier=='balanced' else '438' }) + CenterCrop({meta['img_size']}) + Normalize")
    print(f"  Checkpoint: {meta['checkpoint']}  val_acc={meta['val_acc']:.4f} epoch={meta['epoch']}" if meta['val_acc'] else f"  Checkpoint: {meta['checkpoint']}")
    try:
        subset, n = try_load_wilds_full_ood(transform)
    except Exception as e:
        print(f"  [WARNING] Could not load WILDS full OOD test: {e}")
        print("  Skipping full OOD for this tier (will report N/A).")
        print("  To enable, install wilds and ensure iWildCam data download completes:")
        print("    pip install wilds")
        print("    python -c \"from wilds import get_dataset; get_dataset(dataset='iwildcam', download=True)\"")
        return None

    loader = build_wilds_dataloader(subset, batch_size=batch_size, num_workers=num_workers)
    # Note: wilds dataset may return (x, y, metadata) tuple; our evaluate needs to handle that.
    # Detect by inspecting first batch.
    y_true_all = []
    y_pred_all = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {tier} full OOD", unit="batch"):
            # wilds batch can be (x, y, metadata) or (x, y)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, labels, _ = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
            else:
                raise RuntimeError(f"Unexpected batch format from WILDS: {type(batch)} len={len(batch) if isinstance(batch, (list,tuple)) else 'N/A'}")
            images = images.to(DEVICE)
            if isinstance(labels, torch.Tensor):
                labels = labels.to(DEVICE)
            else:
                labels = torch.tensor(labels).to(DEVICE)
            # Wilds labels may be one-hot or class indices; handle both
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)
            logits = model(images)
            preds = logits.argmax(dim=1)
            y_true_all.extend(labels.cpu().tolist())
            y_pred_all.extend(preds.cpu().tolist())

    acc, macro_f1, weighted_f1, report, per_class = compute_metrics(y_true_all, y_pred_all, meta["class_names"])
    save_name = f"{tier}_full_ood_per_class.csv"
    save_path = REPORT_DIR / save_name
    save_per_class_csv(per_class, save_path)
    print(f"  Full OOD n={len(y_true_all)}  Accuracy={acc*100:.2f}% (expected {TIER_CONFIG[tier]['expected_full_ood_acc']*100:.2f}%)")
    print(f"  Macro-F1={macro_f1:.4f}  Weighted-F1={weighted_f1:.4f}")
    print_per_class_summary(per_class, tier, "full_ood", save_path)
    return {
        "tier": tier,
        "eval_set": "full_ood",
        "n": len(y_true_all),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "report_path": str(save_path),
    }


def evaluate_tier_on_holdout(tier: str, model, meta, batch_size: int, num_workers: int):
    transform = TIER_CONFIG[tier]["transform"]
    print(f"\n--- {tier.upper()} / CAPPED HOLDOUT (data/iwildcam_final_holdout_subset) ---")
    print(f"  Transform: Resize({ '256' if tier=='fast' else '293' if tier=='balanced' else '438' }) + CenterCrop({meta['img_size']}) + Normalize")
    print(f"  Manifest: {HOLDOUT_MANIFEST}  n_expected≈3556")
    dataset = HoldoutManifestDataset(HOLDOUT_MANIFEST, HOLDOUT_DIR, transform)
    n_unique = len(set(lbl for _, lbl in dataset.samples))
    print(f"  Holdout n={len(dataset)}  unique_classes={n_unique}/{meta['num_classes']}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if DEVICE.type == "cuda" else False)
    y_true_all, y_pred_all = evaluate_dataloader(model, loader, DEVICE, desc=f"  {tier} capped holdout")
    acc, macro_f1, weighted_f1, report, per_class = compute_metrics(y_true_all, y_pred_all, meta["class_names"])
    save_name = f"{tier}_capped_holdout_per_class.csv"
    save_path = REPORT_DIR / save_name
    save_per_class_csv(per_class, save_path)
    expected = CAPPED_HEAVY_EXPECTED if tier == "heavy" else None
    exp_str = f" (expected {expected*100:.2f}%)" if expected is not None else ""
    print(f"  Capped holdout n={len(y_true_all)}  Accuracy={acc*100:.2f}%{exp_str}")
    print(f"  Macro-F1={macro_f1:.4f}  Weighted-F1={weighted_f1:.4f}")
    print_per_class_summary(per_class, tier, "capped_holdout", save_path)
    return {
        "tier": tier,
        "eval_set": "capped_holdout",
        "n": len(y_true_all),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "report_path": str(save_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute macro-F1 for iWildCam tiers on full OOD vs capped holdout")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS_DEFAULT, help="DataLoader workers")
    parser.add_argument("--skip-full-ood", action="store_true", help="Skip full WILDS OOD eval (only capped holdout)")
    parser.add_argument("--wilds-root", type=str, default=None, help="Optional root_dir for wilds dataset")
    args = parser.parse_args()

    print("=" * 80)
    print("iWildCam Tier Macro-F1 Diagnostic — Fast / Balanced / Heavy")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Report dir: {REPORT_DIR}")
    print(f"Batch size: {args.batch_size}  Workers: {args.num_workers}")
    print(f"WILDS baseline (published ERM): OOD acc={WILDS_ERM_OOD_ACC*100:.1f}%  OOD macro-F1={WILDS_ERM_OOD_MACRO_F1*100:.1f} (31.0)")
    print(f"Expected full OOD (training-time): Fast 49.87%  Balanced 58.03%  Heavy 75.60%  (n=42,791)")
    print(f"Expected capped holdout (verified): Heavy 52.45% (n=3,556)")
    if args.skip_full_ood:
        print("  --skip-full-ood: WILL SKIP full OOD, evaluating only capped holdout")
    print()

    results = []

    for tier in ["fast", "balanced", "heavy"]:
        print("\n" + "=" * 80)
        print(f"TIER: {tier.upper()}  ({TIER_CONFIG[tier]['builder'].__name__}  img={TIER_CONFIG[tier]['img_size']}px)")
        print("=" * 80)
        model, meta = load_model_for_tier(tier, DEVICE)
        print(f"Loaded {tier}: {meta['checkpoint']}")
        print(f"  num_classes={meta['num_classes']}  val_acc={meta['val_acc']:.4f} epoch={meta['epoch']}" if meta["val_acc"] else f"  num_classes={meta['num_classes']}")
        print(f"  class_names sample: {meta['class_names'][:2]} ... {meta['class_names'][-2:]}")

        # (a) Full OOD
        if not args.skip_full_ood:
            res_full = evaluate_tier_on_wilds(tier, model, meta, batch_size=args.batch_size, num_workers=args.num_workers)
            if res_full is not None:
                results.append(res_full)
            else:
                # Record N/A placeholder
                results.append({
                    "tier": tier,
                    "eval_set": "full_ood",
                    "n": 0,
                    "accuracy": float("nan"),
                    "macro_f1": float("nan"),
                    "weighted_f1": float("nan"),
                    "per_class": [],
                    "report_path": "N/A (wilds not available)",
                })
        else:
            print(f"\n--- {tier.upper()} / FULL WILDS OOD test — SKIPPED (--skip-full-ood) ---")
            results.append({
                "tier": tier,
                "eval_set": "full_ood",
                "n": 0,
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
                "weighted_f1": float("nan"),
                "per_class": [],
                "report_path": "skipped",
            })

        # (b) Capped holdout
        res_capped = evaluate_tier_on_holdout(tier, model, meta, batch_size=args.batch_size, num_workers=args.num_workers)
        results.append(res_capped)

        # Free model to save GPU memory before next tier
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # Final summary table: 6 combinations
    print("\n" + "=" * 80)
    print("FINAL SUMMARY TABLE  (3 tiers x 2 eval sets)")
    print("=" * 80)
    header = f"{'Tier':<10s} | {'Eval Set':<16s} | {'n':>6s} | {'Accuracy':>9s} | {'Macro-F1':>9s} | {'Weighted-F1':>11s} | Report CSV"
    print(header)
    print("-" * len(header))
    for r in results:
        tier = r["tier"]
        eval_set = r["eval_set"]
        n = r["n"]
        acc = r["accuracy"]
        mf1 = r["macro_f1"]
        wf1 = r["weighted_f1"]
        acc_str = f"{acc*100:6.2f}%" if acc == acc else "   N/A"
        mf1_str = f"{mf1:.4f}" if mf1 == mf1 else " N/A"
        wf1_str = f"{wf1:.4f}" if wf1 == wf1 else " N/A"
        print(f"{tier:<10s} | {eval_set:<16s} | {n:6d} | {acc_str:>9s} | {mf1_str:>9s} | {wf1_str:>11s} | {r['report_path']}")

    # Sanity check vs expected
    print("\nSanity checks (tolerance ±1.0% for OOD, ±2% for capped):")
    for r in results:
        tier = r["tier"]
        eval_set = r["eval_set"]
        acc = r["accuracy"]
        if acc != acc:
            print(f"  {tier}/{eval_set}: N/A — skipped or failed to load")
            continue
        if eval_set == "full_ood":
            exp = TIER_CONFIG[tier]["expected_full_ood_acc"]
            delta = (acc - exp) * 100
            ok = abs(delta) <= 1.0
            print(f"  {tier}/full_ood: {acc*100:.2f}% vs expected {exp*100:.2f}%  delta {delta:+.2f} pts  {'OK' if ok else 'MISMATCH (check transform/checkpoint)'}")
        elif eval_set == "capped_holdout" and tier == "heavy":
            delta = (acc - CAPPED_HEAVY_EXPECTED) * 100
            ok = abs(delta) <= 2.0
            print(f"  {tier}/capped_holdout: {acc*100:.2f}% vs expected {CAPPED_HEAVY_EXPECTED*100:.2f}%  delta {delta:+.2f} pts  {'OK' if ok else 'MISMATCH'}")

    # WILDS baseline comparison line (Heavy full OOD)
    print("\n" + "=" * 80)
    print("WILDS BASELINE COMPARISON")
    print("=" * 80)
    heavy_full = next((r for r in results if r["tier"] == "heavy" and r["eval_set"] == "full_ood"), None)
    # Strict format requested
    print(f"WILDS ERM baseline (published): OOD accuracy=71.6%, OOD macro-F1=31.0. "
          f"Our Heavy tier (full OOD test): accuracy={heavy_full['accuracy']*100:.2f}%"
          f", macro-F1={heavy_full['macro_f1']:.4f}." if heavy_full and heavy_full["accuracy"] == heavy_full["accuracy"] else
          f"WILDS ERM baseline (published): OOD accuracy=71.6%, OOD macro-F1=31.0. "
          f"Our Heavy tier (full OOD test): accuracy=N/A, macro-F1=N/A (full OOD not evaluated — see warnings above).")
    if heavy_full and heavy_full["accuracy"] == heavy_full["accuracy"]:
        # Determine exceeds/matches/falls short for macro-F1 vs 0.310 and accuracy vs 0.716
        acc_exceeds = heavy_full["accuracy"] > WILDS_ERM_OOD_ACC + 1e-9
        mf1 = heavy_full["macro_f1"]
        if mf1 > WILDS_ERM_OOD_MACRO_F1 + 0.005:
            mf1_verdict = "exceeds"
        elif mf1 < WILDS_ERM_OOD_MACRO_F1 - 0.005:
            mf1_verdict = "falls short of"
        else:
            mf1_verdict = "matches"
        print(f"  Heavy accuracy {heavy_full['accuracy']*100:.2f}% {'exceeds' if acc_exceeds else 'does NOT exceed'} WILDS ERM 71.6% "
              f"({'already reported' if acc_exceeds else 'unexpected'}).")
        print(f"  Heavy macro-F1 {mf1:.4f} ({mf1*100:.1f}) {mf1_verdict} WILDS ERM baseline 31.0 (0.310).")
        if mf1 > WILDS_ERM_OOD_MACRO_F1:
            print("  => Macro-F1 also exceeds baseline — holds once class imbalance is accounted for (more meaningful per WILDS paper).")
        elif mf1_verdict == "matches":
            print("  => Macro-F1 matches baseline — accuracy gain does not translate to F1 gain under class imbalance.")
        else:
            print("  => Macro-F1 falls short despite higher accuracy — accuracy gain is driven by majority classes; "
                  "WILDS paper's macro-F1 preference reveals class-imbalance weakness.")
    else:
        print("  Full OOD not available — cannot compare macro-F1 to WILDS baseline. Run without --skip-full-ood after `pip install wilds` and dataset download.")

    print("\nFull per-class CSVs saved to:", REPORT_DIR)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
