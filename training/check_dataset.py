import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
RAW_DIR = BASE_DIR / "data" / "raw" / "Fruit And Vegetable Diseases Dataset"

print(f"{'Class':<40}{'Count':>10}")
print("-" * 50)

total = 0
for class_name in sorted(os.listdir(RAW_DIR)):
    class_path = RAW_DIR / class_name
    if class_path.is_dir():
        count = len(os.listdir(class_path))
        total += count
        print(f"{class_name:<40}{count:>10}")

print("-" * 50)
print(f"{'TOTAL':<40}{total:>10}")