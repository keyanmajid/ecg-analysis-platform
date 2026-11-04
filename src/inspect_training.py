from pathlib import Path
from collections import Counter
import numpy as np

from src.split_config import EXPERIMENT_A

DATA_DIR = Path("data/processed/mit-bih-216")

print("=" * 75)
print("EXPERIMENT A — TRAINING RECORD AUDIT")
print("=" * 75)

total = Counter()

for record_id in EXPERIMENT_A["train"]:
    path = DATA_DIR / f"record_{record_id}.npz"

    if not path.exists():
        print(f"Missing: {path}")
        continue

    data = np.load(path, allow_pickle=False)
    labels = Counter(data["y"].tolist())

    print(f"\nRecord: {record_id}")

    for cls in ["N", "S", "V", "F"]:
        count = labels.get(cls, 0)
        total[cls] += count
        print(f"  {cls}: {count}")

print("\n" + "=" * 75)
print("TOTAL TRAINING DISTRIBUTION")
print("=" * 75)

for cls in ["N", "S", "V", "F"]:
    print(f"{cls}: {total[cls]}")