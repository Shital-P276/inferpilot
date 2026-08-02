"""
Training script for the Balanced tier (MobileNetV3, transfer learning).
Two-phase fine-tuning:
  Phase 1: backbone frozen, train classifier head only (fast, few epochs)
  Phase 2: unfreeze last few blocks, fine-tune at lower LR (slower, more epochs)

Run: python training/train_balanced.py
"""
import time
from pathlib import Path

import torch
import torch.nn as nn
import mlflow
import matplotlib.pyplot as plt

from models.balanced_mobilenet import BalancedMobileNet
from data_pipeline import get_dataloaders

# ---------------- Config ----------------
NUM_CLASSES = 28
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 10
PHASE1_LR = 1e-3
PHASE2_LR = 1e-4
WEIGHT_DECAY = 1e-4
UNFREEZE_BLOCKS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "training" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_EXPERIMENT = "inferpilot-model-tiers"


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
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def plot_curves(history, save_path, phase_boundary):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].axvline(phase_boundary, color="gray", linestyle="--", label="Phase 2 start")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].axvline(phase_boundary, color="gray", linestyle="--", label="Phase 2 start")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main():
    print(f"Using device: {DEVICE}")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    assert len(class_names) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(class_names)}"

    model = BalancedMobileNet(num_classes=NUM_CLASSES, freeze_backbone=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_ckpt_path = CHECKPOINT_DIR / "balanced_mobilenet_best.pt"
    global_epoch = 0

    with mlflow.start_run(run_name="balanced_mobilenet"):
        mlflow.log_params({
            "architecture": "MobileNetV3-Large",
            "num_classes": NUM_CLASSES,
            "phase1_epochs": PHASE1_EPOCHS,
            "phase2_epochs": PHASE2_EPOCHS,
            "phase1_lr": PHASE1_LR,
            "phase2_lr": PHASE2_LR,
            "weight_decay": WEIGHT_DECAY,
            "unfreeze_blocks": UNFREEZE_BLOCKS,
            "optimizer": "AdamW",
            "device": str(DEVICE),
        })

        # ---------------- Phase 1: frozen backbone ----------------
        print("\n=== Phase 1: training classifier head (backbone frozen) ===")
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=PHASE1_LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS)

        for epoch in range(1, PHASE1_EPOCHS + 1):
            global_epoch += 1
            t0 = time.time()

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            scheduler.step()

            epoch_time = time.time() - t0
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "epoch_time_sec": epoch_time, "phase": 1,
            }, step=global_epoch)

            print(f"[Phase 1] Epoch {epoch:2d}/{PHASE1_EPOCHS} | "
                  f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
                  f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | {epoch_time:.1f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "val_acc": val_acc,
                    "epoch": global_epoch,
                    "phase": 1,
                }, best_ckpt_path)
                print(f"  -> new best val_acc {val_acc:.4f}, checkpoint saved")

        phase_boundary = global_epoch + 0.5

        # ---------------- Phase 2: unfreeze + fine-tune ----------------
        print(f"\n=== Phase 2: unfreezing last {UNFREEZE_BLOCKS} blocks, fine-tuning ===")
        model.unfreeze_last_blocks(UNFREEZE_BLOCKS)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=PHASE2_LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params in phase 2: {trainable:,}")
        mlflow.log_param("phase2_trainable_params", trainable)

        for epoch in range(1, PHASE2_EPOCHS + 1):
            global_epoch += 1
            t0 = time.time()

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            scheduler.step()

            epoch_time = time.time() - t0
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "epoch_time_sec": epoch_time, "phase": 2,
            }, step=global_epoch)

            print(f"[Phase 2] Epoch {epoch:2d}/{PHASE2_EPOCHS} | "
                  f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
                  f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | {epoch_time:.1f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "val_acc": val_acc,
                    "epoch": global_epoch,
                    "phase": 2,
                }, best_ckpt_path)
                print(f"  -> new best val_acc {val_acc:.4f}, checkpoint saved")

        # ---------------- Final test evaluation ----------------
        best_ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
        print(f"\nFinal test_acc (best checkpoint, phase {best_ckpt['phase']}): {test_acc:.4f}")

        curve_path = CHECKPOINT_DIR / "balanced_mobilenet_curves.png"
        plot_curves(history, curve_path, phase_boundary)
        mlflow.log_artifact(str(curve_path))
        mlflow.log_artifact(str(best_ckpt_path))
        mlflow.log_metric("best_val_acc", best_val_acc)

    print(f"\nDone. Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()