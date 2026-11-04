
from pathlib import Path
from collections import Counter

import numpy as np

from src.split_config import EXPERIMENT_A


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "mit-bih-216"
)

CLASS_NAMES = ["N", "S", "V", "F"]


def inspect_validation():

    validation_records = EXPERIMENT_A["validation"]

    print("=" * 75)
    print("EXPERIMENT A — VALIDATION RECORD AUDIT")
    print("=" * 75)

    total_counts = Counter()

    for record_id in validation_records:

        file_path = DATA_DIR / f"record_{record_id}.npz"

        if not file_path.exists():
            print(f"Missing file: {file_path}")
            continue

        data = np.load(file_path, allow_pickle=False)

        labels = data["y"]

        counts = Counter(
            str(label) for label in labels
        )

        total_counts.update(counts)

        print(f"\nRecord: {record_id}")
        print(f"Total beats: {len(labels)}")

        for class_name in CLASS_NAMES:
            print(
                f"  {class_name}: "
                f"{counts.get(class_name, 0)}"
            )

        if "symbols" in data.files:

            symbols = data["symbols"]

            print("  Original annotation symbols:")

            symbol_counts = Counter(
                str(symbol) for symbol in symbols
            )

            print(dict(symbol_counts))

    print("\n" + "=" * 75)
    print("TOTAL VALIDATION DISTRIBUTION")
    print("=" * 75)

    for class_name in CLASS_NAMES:
        print(
            f"{class_name}: "
            f"{total_counts.get(class_name, 0)}"
        )


if __name__ == "__main__":
    inspect_validation()
