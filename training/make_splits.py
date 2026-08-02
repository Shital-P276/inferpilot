"""
Stratified train/val/test split for the fruit & vegetable disease dataset.

Reads from data/raw/Fruit And Vegetable Diseases Dataset/<ClassName>/*.jpg
Writes (copies) into:
    data/train/<ClassName>/*.jpg
    data/val/<ClassName>/*.jpg
    data/test/<ClassName>/*.jpg

Split is stratified PER CLASS, so small classes (e.g. Grape__Healthy, 200 images)
get proportional representation in val/test instead of being randomly starved.

Default split: 70% train / 15% val / 15% test
"""

import random
import shutil
from pathlib import Path

# ---- Config ----
BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
RAW_DIR = BASE_DIR / "data" / "raw" / "Fruit And Vegetable Diseases Dataset"
OUT_DIR = BASE_DIR / "data"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
# ----------------

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "Ratios must sum to 1.0"

random.seed(SEED)

def main():
    class_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} classes in {RAW_DIR}\n")

    summary = []

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = [f for f in class_dir.iterdir() if f.is_file()]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # remainder goes to test, so rounding doesn't drop images
        n_test = n - n_train - n_val

        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            split_class_dir = OUT_DIR / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, split_class_dir / f.name)

        summary.append((class_name, n, n_train, n_val, n_test))
        print(f"{class_name:<25} total={n:<6} train={n_train:<6} val={n_val:<6} test={n_test:<6}")

    # Totals
    total_n = sum(s[1] for s in summary)
    total_train = sum(s[2] for s in summary)
    total_val = sum(s[3] for s in summary)
    total_test = sum(s[4] for s in summary)

    print("\n" + "-" * 60)
    print(f"{'TOTAL':<25} total={total_n:<6} train={total_train:<6} val={total_val:<6} test={total_test:<6}")
    print("\nDone. Splits written to data/train, data/val, data/test")

if __name__ == "__main__":
    main()