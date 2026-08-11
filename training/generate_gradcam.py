"""
generate_gradcam.py

Generates Grad-CAM visualizations for all 3 trained model tiers
(Fast/Balanced/Heavy), showing what each model actually attends to
when making a prediction. Required deliverable per the project plan's
"never drop" list.

BEFORE RUNNING:
    pip install grad-cam --break-system-packages
    (this is the `pytorch-grad-cam` package, imported as `pytorch_grad_cam`)

USAGE:
    python training/generate_gradcam.py

OUTPUT:
    reports/gradcam/{tier}_{class_name}_{n}.png -- one overlay per sampled image,
    for each of the 3 tiers, so you get a side-by-side sense of what each
    model is "looking at" for the same/similar inputs.
"""

import random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms import v2

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    raise ImportError(
        "pytorch-grad-cam not installed. Run: pip install grad-cam --break-system-packages"
    )

from models.fast_cnn import FastCNN
from models.balanced_mobilenet import BalancedMobileNet
from models.heavy_efficientnet import HeavyEfficientNet

# ---- CONFIG ---------------------------------------------------------
CHECKPOINTS = {
    "fast": {
        "path": "training/checkpoints/fast_cnn_best.pt",
        "model_class": FastCNN,
    },
    "balanced": {
        "path": "training/checkpoints/balanced_mobilenet_best.pt",
        "model_class": BalancedMobileNet,
    },
    "heavy": {
        "path": "training/checkpoints/heavy_efficientnet_best.pt",
        "model_class": HeavyEfficientNet,
    },
}

DATA_TEST_DIR = Path("data/test")
OUTPUT_DIR = Path("reports/gradcam")
N_SAMPLE_CLASSES = 6   # how many different classes to sample images from
N_IMAGES_PER_CLASS = 1  # how many images per sampled class

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Matches fast_service.py / balanced_service.py / heavy_service.py EXACTLY --
# confirmed identical across all 3 real serving files. Resize(256) then
# CenterCrop(224), NOT a direct square resize -- squashing distorts aspect
# ratio and can genuinely change predictions on non-square images (caught
# empirically: a pomegranate image predicted correctly via the live API
# flipped to incorrect when an earlier version of this script used a naive
# square resize instead of matching production preprocessing exactly).
MODEL_PREPROCESS = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Same Resize+CenterCrop geometry, but WITHOUT normalization -- used to
# build the visible overlay image, so the heatmap is drawn over the same
# crop the model actually saw, not a differently-cropped version.
DISPLAY_PREPROCESS = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
])
# NOTE: matches the real serving pipeline (see serving/heavy_service.py's
# eval_transform) exactly -- Resize(256) then CenterCrop(224), NOT a direct
# Resize((224,224)). A direct square resize distorts aspect ratio for any
# non-square source image, which can genuinely change model predictions on
# borderline cases -- confirmed this mattered on a real image in this
# project (a Pomegranate sample predicted differently between the two
# preprocessing paths). Keep this in sync with whatever serving/*.py
# actually uses if that ever changes.
# ------------------------------------------------------------------------


def get_target_layer(tier_name: str, model: nn.Module) -> nn.Module:
    """Explicit, architecture-specific Grad-CAM target layers -- confirmed
    against the real model source (fast_cnn.py, balanced_mobilenet.py,
    heavy_efficientnet.py), not inferred generically.

    fast: FastCNN.features is a flat nn.Sequential with no branching --
        Conv2d, BN, ReLU, MaxPool2d x4. The last Conv2d is at index 12
        (Conv2d(128,256,...), followed by BN/ReLU/MaxPool at 13-15).

    balanced/heavy: both wrap a torchvision backbone (mobilenet_v3_large /
        efficientnet_b0) via self.backbone. backbone.features[-1] is the
        final Conv2dNormActivation block (conv+BN+activation) before the
        classifier -- the standard Grad-CAM target for these architectures.
        Earlier blocks inside backbone.features have internal residual/skip
        connections (inverted residual blocks), which is exactly the kind
        of branching a generic "last Conv2d found by traversal" search
        could target incorrectly -- targeting the whole final block instead
        (not a specific conv inside it) sidesteps that risk entirely.
    """
    if tier_name == "fast":
        layer = model.features[12]
        if not isinstance(layer, nn.Conv2d):
            raise RuntimeError(
                f"Expected model.features[12] to be the last Conv2d in FastCNN, "
                f"got {type(layer)} instead. The architecture may have changed -- "
                f"share the updated fast_cnn.py rather than letting this guess."
            )
        return layer

    elif tier_name in ("balanced", "heavy"):
        return model.backbone.features[-1]

    else:
        raise ValueError(f"No target layer defined for tier '{tier_name}'")


def load_model_and_classes(tier_name: str):
    cfg = CHECKPOINTS[tier_name]
    checkpoint = torch.load(cfg["path"], map_location=DEVICE, weights_only=False)

    if not isinstance(checkpoint, dict) or "class_names" not in checkpoint:
        raise RuntimeError(
            f"Checkpoint at {cfg['path']} doesn't have a 'class_names' key. "
            f"Expected format: dict with model_state_dict/class_names/val_acc/epoch. "
            f"Got keys: {checkpoint.keys() if isinstance(checkpoint, dict) else type(checkpoint)}. "
            f"If your checkpoint format differs, tell me and I'll adjust this script "
            f"rather than guessing at the class list."
        )

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = cfg["model_class"](num_classes=num_classes)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, class_names


def sample_test_images(class_names, n_classes, n_per_class):
    """Picks a random subset of classes and one or more real images per
    class from data/test/. Uses the SAME sample across all 3 tiers (seeded)
    so the Grad-CAM comparison is apples-to-apples on identical inputs."""
    random.seed(42)

    available_classes = [c for c in class_names if (DATA_TEST_DIR / c).is_dir()]
    if not available_classes:
        raise RuntimeError(
            f"None of the checkpoint's class_names match subfolders in {DATA_TEST_DIR}. "
            f"Expected folder names like 'Apple__Healthy' etc. Checkpoint class_names: "
            f"{class_names[:5]}... Check the naming convention matches your actual "
            f"data/test/ folder structure."
        )

    sampled_classes = random.sample(available_classes, min(n_classes, len(available_classes)))

    samples = []
    for cls in sampled_classes:
        class_dir = DATA_TEST_DIR / cls
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if not images:
            continue
        chosen = random.sample(images, min(n_per_class, len(images)))
        for img_path in chosen:
            samples.append((cls, img_path))

    return samples


def generate_gradcam_for_tier(tier_name: str, samples: list):
    print(f"\n=== Generating Grad-CAM for {tier_name} ===")
    model, class_names = load_model_and_classes(tier_name)
    target_layer = get_target_layer(tier_name, model)
    print(f"  Target layer: {target_layer}")

    cam = GradCAM(model=model, target_layers=[target_layer])

    for cls, img_path in samples:
        pil_img = Image.open(img_path).convert("RGB")
        input_tensor = MODEL_PREPROCESS(pil_img).unsqueeze(0).to(DEVICE)
        # Balanced/Heavy freeze their backbone weights by default
        # (freeze_backbone=True), so with no gradient anywhere in the graph,
        # Grad-CAM's backward pass finds nothing to differentiate (grads
        # come back as None). Requiring grad on the INPUT instead keeps the
        # graph gradient-capable regardless of which weights are frozen --
        # gradients still flow back through frozen layers to the input,
        # they just don't update those frozen weights (irrelevant here,
        # we're not training).
        input_tensor.requires_grad_(True)

        # Normalized [0,1] float image for overlay visualization (not the
        # normalized-for-model version -- that would look washed out).
        # Same crop the model actually saw (Resize(256)->CenterCrop(224)),
        # just without normalization -- for a correctly-aligned overlay.
        display_img = DISPLAY_PREPROCESS(pil_img)
        rgb_img = np.array(display_img).astype(np.float32) / 255.0

        grayscale_cam = cam(input_tensor=input_tensor)[0]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        with torch.no_grad():
            logits = model(input_tensor)
            pred_idx = logits.argmax(dim=1).item()
            pred_class = class_names[pred_idx]

        out_name = f"{tier_name}_{cls}_{img_path.stem}.png"
        out_path = OUTPUT_DIR / out_name
        Image.fromarray(visualization).save(out_path)

        correct = "✓" if pred_class == cls else "✗"
        print(f"  {img_path.name}: true={cls}, pred={pred_class} {correct} -> saved {out_name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use Fast's class list to pick samples (all 3 tiers should share the
    # same class set/order since they're trained on the same dataset -- if
    # they don't, that's worth knowing, and this will surface it via the
    # per-tier class_names check inside load_model_and_classes).
    _, fast_class_names = load_model_and_classes("fast")
    samples = sample_test_images(fast_class_names, N_SAMPLE_CLASSES, N_IMAGES_PER_CLASS)

    if not samples:
        raise RuntimeError(
            f"No sample images found under {DATA_TEST_DIR}. Check the path and "
            f"that class subfolders actually contain .jpg/.png files."
        )

    print(f"Sampled {len(samples)} images across {len(set(c for c, _ in samples))} classes:")
    for cls, path in samples:
        print(f"  {cls}: {path.name}")

    for tier_name in CHECKPOINTS:
        generate_gradcam_for_tier(tier_name, samples)

    print(f"\nAll Grad-CAM visualizations saved to {OUTPUT_DIR}/")
    print("Compare the SAME image across fast_/balanced_/heavy_ prefixed files "
          "to see how attention differs by model size/architecture -- this "
          "comparison is good report content on its own (e.g. does Heavy "
          "attend to more localized defect regions than Fast?).")


if __name__ == "__main__":
    main()