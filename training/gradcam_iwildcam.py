"""
training/gradcam_iwildcam.py

Grad-CAM for the iWildCam tiers (Fast/MobileNetV3-Small, Balanced/ResNet-18,
Heavy/ConvNeXt-Base). Replaces the dead produce-era training/generate_gradcam.py
(which imports FastCNN/BalancedMobileNet/HeavyEfficientNet that no longer exist).

Sampling is DISAGREEMENT-FOCUSED (not random): 4-5 images where the tiers'
predicted labels differ -- ideally Fast-wrong/Heavy-right, incl. >=1 rare
species case tying into the long-tail finding -- plus 2 agreement controls
(1 all-correct, 1 all-wrong). Rationale: accuracy tables already say the tiers
differ; Grad-CAM must explain WHY (what does Heavy attend to that Fast misses?).

Preprocess matches serving/*_service.py EXACTLY per tier:
  Resize(int(IMG_SIZE*256/224)) -> CenterCrop(IMG_SIZE) -> Normalize,
  with IMG_SIZE = 224 / 256 / 384 for fast / balanced / heavy.

Run: source ~/venv_inferpilot/bin/activate && python training/gradcam_iwildcam.py
Output: reports/gradcam_iwildcam/{tier}_{stem}.png + selection.csv
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

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.models.iwildcam_fast_mobilenetv3small import build_fast_mobilenetv3small
from training.models.iwildcam_balanced_resnet18 import build_balanced_resnet18
from training.models.iwildcam_heavy_convnextbase import build_heavy_convnextbase

CHECKPOINT_DIR = REPO_ROOT / "training" / "checkpoints"
HOLDOUT_DIR = REPO_ROOT / "data" / "iwildcam_final_holdout_subset"
HOLDOUT_MANIFEST = HOLDOUT_DIR / "manifest.csv"
OUT_DIR = REPO_ROOT / "reports" / "gradcam_iwildcam"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
SEED = 42

N_DISAGREE = 5
N_AGREE = 2

TIER_CONFIG = {
    "fast": {"builder": build_fast_mobilenetv3small,
             "checkpoint": CHECKPOINT_DIR / "iwildcam_fast_mobilenetv3small_best.pt",
             "img": 224},
    "balanced": {"builder": build_balanced_resnet18,
                 "checkpoint": CHECKPOINT_DIR / "iwildcam_balanced_resnet18_best.pt",
                 "img": 256},
    "heavy": {"builder": build_heavy_convnextbase,
              "checkpoint": CHECKPOINT_DIR / "iwildcam_heavy_convnextbase_best.pt",
              "img": 384},
}


def make_transforms(img_size):
    model_tf = v2.Compose([
        v2.ToImage(),
        v2.Resize(int(img_size * 256 / 224)),
        v2.CenterCrop(img_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    display_tf = v2.Compose([
        v2.Resize(int(img_size * 256 / 224)),
        v2.CenterCrop(img_size),
    ])
    return model_tf, display_tf


def get_target_layer(tier_name, model):
    """Architecture-specific Grad-CAM targets, asserted (halt on mismatch)."""
    if tier_name == "fast":
        layer = model.features[-1]
        print(f"  fast target: {type(layer).__name__}")
        if "conv" not in type(layer).__name__.lower() and not hasattr(layer, "block"):
            raise RuntimeError(f"Unexpected fast target layer: {type(layer)}")
        return layer
    if tier_name == "balanced":
        layer = model.layer4[-1]
        print(f"  balanced target: {type(layer).__name__}")
        if type(layer).__name__ != "BasicBlock":
            raise RuntimeError(f"Unexpected balanced target layer: {type(layer)}")
        return layer
    if tier_name == "heavy":
        stage = model.features[-1]
        print(f"  heavy stage: {type(stage).__name__}")
        # torchvision convnext_base: features[-1] is the last Sequential stage;
        # target its last CNBlock for a localized signal.
        import torch.nn as _nn
        layer = stage[-1] if isinstance(stage, _nn.Sequential) else stage
        print(f"  heavy target: {type(layer).__name__}")
        if layer is model or not any(True for _ in layer.parameters()):
            raise RuntimeError(f"Unexpected heavy target layer: {type(layer)}")
        return layer
    raise ValueError(tier_name)


class ManifestImages(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, _ = self.samples[idx]
        return self.transform(Image.open(img_path).convert("RGB"))


@torch.no_grad()
def predict_all(model, samples, transform):
    loader = DataLoader(ManifestImages(samples, transform), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=2,
                        pin_memory=DEVICE.type == "cuda")
    preds = []
    for batch in tqdm(loader, desc="    infer", unit="batch", leave=False):
        preds.extend(model(batch.to(DEVICE)).argmax(dim=1).cpu().tolist())
    return np.asarray(preds)


def main():
    print("=" * 70)
    print("Grad-CAM (iWildCam tiers, disagreement-focused sampling)")
    print("=" * 70)
    rng = np.random.default_rng(SEED)

    with open(HOLDOUT_MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    samples = [((HOLDOUT_DIR / r["image_path"].replace("\\", "/")).resolve(),
                int(r["label"])) for r in rows]
    print(f"Holdout: {len(samples)} images")

    models, preds, names = {}, {}, {}
    for tier, cfg in TIER_CONFIG.items():
        print(f"--- {tier} ---")
        ckpt = torch.load(cfg["checkpoint"], map_location=DEVICE)
        model = cfg["builder"](num_classes=ckpt.get("num_classes", 182))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        models[tier] = model
        names[tier] = ckpt.get("class_names", [f"class_{i}" for i in range(182)])
        model_tf, _ = make_transforms(cfg["img"])
        preds[tier] = predict_all(model, samples, model_tf)
        acc = float((preds[tier] == np.array([s[1] for s in samples])).mean())
        print(f"  holdout acc: {acc:.4f}")

    pf, pb, ph = preds["fast"], preds["balanced"], preds["heavy"]
    true = np.array([s[1] for s in samples])
    correct = {t: (preds[t] == true) for t in TIER_CONFIG}

    # Disagreement pool: tiers' predictions differ; prefer Fast-wrong/Heavy-right
    disagree_idx = np.nonzero((pf != ph) | (pf != pb) | (pb != ph))[0]
    print(f"\nDisagreement pool: {len(disagree_idx)} images")
    fwh_idx = [i for i in disagree_idx if not correct["fast"][i] and correct["heavy"][i]]
    print(f"  Fast-wrong/Heavy-right subset: {len(fwh_idx)}")
    # Rare-species preference: true labels with lowest holdout support
    label_counts = {l: int((true == l).sum()) for l in set(true.tolist())}

    def rarity(i):
        return label_counts[true[i]]

    fwh_sorted = sorted(fwh_idx, key=rarity)
    other_dis = [i for i in disagree_idx if i not in set(fwh_idx)]
    rng.shuffle(other_dis)
    selected, roles = [], []
    for i in (fwh_sorted + other_dis):
        if len(selected) >= N_DISAGREE:
            break
        selected.append(int(i))
        roles.append("disagree_fast-wrong_heavy-right" if i in set(fwh_idx) else "disagree")

    # Agreement controls: 1 all-correct, 1 all-wrong
    all_ok = np.nonzero(correct["fast"] & correct["balanced"] & correct["heavy"])[0]
    all_bad = np.nonzero(~correct["fast"] & ~correct["balanced"] & ~correct["heavy"])[0]
    rng.shuffle(all_ok)
    rng.shuffle(all_bad)
    if len(all_ok):
        selected.append(int(all_ok[0]))
        roles.append("agree_all-correct")
    if len(all_bad):
        selected.append(int(all_bad[0]))
        roles.append("agree_all-wrong")
    print(f"Selected {len(selected)} images: {roles}")

    sel_rows = []
    for tier, model in models.items():
        cfg = TIER_CONFIG[tier]
        model_tf, display_tf = make_transforms(cfg["img"])
        target = get_target_layer(tier, model)
        cam = GradCAM(model=model, target_layers=[target])
        for img_idx, role in zip(selected, roles):
            img_path, true_label = samples[img_idx]
            pil_img = Image.open(img_path).convert("RGB")
            inp = model_tf(pil_img).unsqueeze(0).to(DEVICE)
            inp.requires_grad_(True)  # gradient-capable graph (backbones frozen)
            gray = cam(input_tensor=inp)[0]
            rgb = np.array(display_tf(pil_img)).astype(np.float32) / 255.0
            vis = show_cam_on_image(rgb, gray, use_rgb=True)
            with torch.no_grad():
                pred_idx = int(model(inp).argmax(dim=1).item())
            tname = names[tier][true_label] if true_label < len(names[tier]) else str(true_label)
            pname = names[tier][pred_idx] if pred_idx < len(names[tier]) else str(pred_idx)
            mark = "OK" if pred_idx == true_label else "WRONG"
            out = OUT_DIR / f"{tier}_{img_path.stem}.png"
            Image.fromarray(vis).save(out)
            print(f"  [{tier}] {img_path.name} ({role}): true={tname} pred={pname} {mark}")
            sel_rows.append({"image": img_path.name, "role": role, "tier": tier,
                             "true": tname, "pred": pname, "correct": int(pred_idx == true_label),
                             "file": str(out.relative_to(REPO_ROOT))})

    with open(OUT_DIR / "selection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sel_rows[0].keys()))
        w.writeheader()
        w.writerows(sel_rows)
    print(f"\nSaved overlays + selection.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
