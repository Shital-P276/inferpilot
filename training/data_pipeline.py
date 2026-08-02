"""
Data pipeline: augmentation + DataLoaders for train/val/test splits.

Uses torchvision.transforms.v2 + ImageFolder over data/train, data/val, data/test
(produced by make_splits.py).

Train transform is deliberately harder than a "clean" pipeline would need:
our eyeball test showed rotten-vs-healthy is mostly easily distinguishable,
so we inject blur, noise, and lighting variation to (a) regularize, and
(b) manufacture enough difficulty that model capacity (fast/balanced/heavy)
actually matters -- this data feeds the router's whole reason for existing.

Val/test transforms stay clean (deterministic resize + normalize only) --
we want to measure real generalization, not test-time augmentation effects.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from pathlib import Path

# ---- Config ----
BASE_DIR = Path(__file__).resolve().parent.parent  # repo root (training/ is one level down)
DATA_DIR = BASE_DIR / "data"
IMG_SIZE = 224          # standard for MobileNetV3 / EfficientNet-Lite / ViT-small
BATCH_SIZE = 32
NUM_WORKERS = 4         # tune down to 0-2 if you hit Windows multiprocessing issues
# ----------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---- Train transform: deliberately harder ----
train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=20),

    # Lighting / color variation -- simulates mixed packing-house lighting
    v2.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.3, hue=0.05),

    # Blur -- simulates motion blur / poor focus on a fast conveyor line
    v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.5, 2.5))], p=0.3),

    # Random erasing -- simulates partial occlusion (e.g. produce overlapping,
    # a hand/equipment briefly in frame)
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomErasing(p=0.2, scale=(0.02, 0.15)),

    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---- Val/test transform: clean, deterministic ----
eval_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(IMG_SIZE),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Builds train/val/test ImageFolder datasets and DataLoaders."""

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transform)
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transform)

    # Sanity check: class-to-index mapping must match across all three splits
    assert train_ds.classes == val_ds.classes == test_ds.classes, \
        "Class mismatch between train/val/test -- check make_splits.py output"

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes


if __name__ == "__main__":
    # Quick sanity check -- confirms loaders work end-to-end before training
    train_loader, val_loader, test_loader, classes = get_dataloaders()

    print(f"Classes ({len(classes)}): {classes}\n")
    print(f"Train batches: {len(train_loader)}  ({len(train_loader.dataset)} images)")
    print(f"Val batches:   {len(val_loader)}  ({len(val_loader.dataset)} images)")
    print(f"Test batches:  {len(test_loader)}  ({len(test_loader.dataset)} images)")

    # Pull one batch to confirm shapes are sane
    images, labels = next(iter(train_loader))
    print(f"\nSample batch -- images: {images.shape}, labels: {labels.shape}")
    print(f"Pixel range: [{images.min():.3f}, {images.max():.3f}]  (should be roughly normalized, not [0,1] or [0,255])")