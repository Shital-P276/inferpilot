"""
training/train_gate.py

Trains the routing gate model: image -> {fast, balanced, heavy}.
Reuses router/utility_labels.csv (image_path, best_tier) as ground truth --
same 4000 images the sklearn router was trained on, same labels, different
input representation (raw pixels instead of hand-crafted features).

Run: python training/train_gate.py
"""
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision.transforms import v2
import mlflow
import matplotlib.pyplot as plt

from models.gate_net import GateNet  # save the class above to training/models/gate_net.py

# ---------------- Config ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
LABELS_PATH = BASE_DIR / "router" / "utility_labels.csv"
CHECKPOINT_DIR = BASE_DIR / "training" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
EPOCHS = 25
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MLFLOW_EXPERIMENT = "inferpilot-model-tiers"

TIER_TO_IDX = {"fast": 0, "balanced": 1, "heavy": 2}
IDX_TO_TIER = {v: k for k, v in TIER_TO_IDX.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
eval_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(IMG_SIZE),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class GateDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = BASE_DIR / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = TIER_TO_IDX[row["best_tier"]]
        return image, label


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    per_class_correct = Counter()
    per_class_total = Counter()
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for p, l in zip(preds.tolist(), labels.tolist()):
            per_class_total[l] += 1
            if p == l:
                per_class_correct[l] += 1

    balanced_acc = sum(
        per_class_correct[c] / per_class_total[c] for c in per_class_total
    ) / len(per_class_total)

    return running_loss / total, correct / total, balanced_acc


def main():
    print(f"Using device: {DEVICE}")
    df = pd.read_csv(LABELS_PATH)
    print(f"Loaded {len(df)} labeled samples")
    print(f"Label distribution:\n{df['best_tier'].value_counts()}\n")

    # Stratified 70/15/15 split, same discipline as the sklearn router
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["best_tier"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["best_tier"], random_state=SEED
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    train_loader = DataLoader(GateDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(GateDataset(val_df, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(GateDataset(test_df, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = GateNet(num_classes=3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GateNet params: {n_params:,}")

    # Class-imbalance handling: inverse-frequency weighted loss, same
    # imbalance-aware spirit as the Balanced Random Forest's class_weight
    class_counts = df["best_tier"].value_counts()
    weights = torch.tensor(
        [1.0 / class_counts[IDX_TO_TIER[i]] for i in range(3)], dtype=torch.float32
    ).to(DEVICE)
    weights = weights / weights.sum() * 3  # normalize
    print(f"Class weights (fast, balanced, heavy): {weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_balanced_acc": []}
    best_val_balanced_acc = 0.0
    best_ckpt_path = CHECKPOINT_DIR / "gate_net_best.pt"

    with mlflow.start_run(run_name="gate_net"):
        mlflow.log_params({
            "architecture": "GateNet", "num_params": n_params, "epochs": EPOCHS,
            "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "class_weights": weights.tolist(),
        })

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, val_balanced_acc = evaluate(model, val_loader, criterion)
            scheduler.step()
            epoch_time = time.time() - t0

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_balanced_acc"].append(val_balanced_acc)

            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "val_balanced_acc": val_balanced_acc, "epoch_time_sec": epoch_time,
            }, step=epoch)

            print(f"Epoch {epoch:2d}/{EPOCHS} | train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
                  f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} val_balanced_acc {val_balanced_acc:.4f} | {epoch_time:.1f}s")

            if val_balanced_acc > best_val_balanced_acc:
                best_val_balanced_acc = val_balanced_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_names": ["fast", "balanced", "heavy"],
                    "val_balanced_acc": val_balanced_acc,
                    "epoch": epoch,
                }, best_ckpt_path)
                print(f"  -> new best val_balanced_acc {val_balanced_acc:.4f}, checkpoint saved")

        best_ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_loss, test_acc, test_balanced_acc = evaluate(model, test_loader, criterion)
        mlflow.log_metrics({"test_acc": test_acc, "test_balanced_acc": test_balanced_acc})
        print(f"\nFinal test_acc: {test_acc:.4f}, test_balanced_acc: {test_balanced_acc:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs_range = range(1, EPOCHS + 1)
        axes[0].plot(epochs_range, history["train_loss"], label="Train")
        axes[0].plot(epochs_range, history["val_loss"], label="Val")
        axes[0].set_title("Loss"); axes[0].legend()
        axes[1].plot(epochs_range, history["train_acc"], label="Train Acc")
        axes[1].plot(epochs_range, history["val_acc"], label="Val Acc")
        axes[1].plot(epochs_range, history["val_balanced_acc"], label="Val Balanced Acc")
        axes[1].set_title("Accuracy"); axes[1].legend()
        curve_path = CHECKPOINT_DIR / "gate_net_curves.png"
        fig.savefig(curve_path)
        plt.close(fig)
        mlflow.log_artifact(str(curve_path))
        mlflow.log_artifact(str(best_ckpt_path))

    print(f"\nDone. Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()