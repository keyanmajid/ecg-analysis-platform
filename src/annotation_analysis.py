
"""
MIT-BIH Annotation Exploration
------------------------------
Analyzes expert ECG annotations across all 48 recordings.
"""

from pathlib import Path
from collections import Counter
import wfdb


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "raw" / "mit-bih-arrhythmia"


def analyze_annotations():

    records = sorted(DATA_DIR.glob("*.hea"))

    if not records:
        raise FileNotFoundError("MIT-BIH dataset not found.")

    overall_counts = Counter()
    record_counts = {}

    print("=" * 65)
    print("MIT-BIH ANNOTATION EXPLORATION")
    print("=" * 65)

    print(f"\nTotal recordings: {len(records)}")

    for header in records:

        record_id = header.stem

        annotation = wfdb.rdann(
            str(DATA_DIR / record_id),
            extension="atr"
        )

        counts = Counter(annotation.symbol)

        record_counts[record_id] = counts
        overall_counts.update(counts)

        print(
            f"Record {record_id}: "
            f"{len(annotation.sample)} annotations"
        )

    print("\n" + "=" * 65)
    print("OVERALL ANNOTATION DISTRIBUTION")
    print("=" * 65)

    print(f"\nTotal annotations: {sum(overall_counts.values())}")

    print(f"\n{'Symbol':<15}{'Count':>12}{'Percentage':>15}")
    print("-" * 42)

    total = sum(overall_counts.values())

    for symbol, count in sorted(
        overall_counts.items(),
        key=lambda item: item[1],
        reverse=True
    ):

        percentage = (count / total) * 100

        print(
            f"{symbol!r:<15}"
            f"{count:>12}"
            f"{percentage:>14.2f}%"
        )

    print("\n" + "=" * 65)
    print("ANNOTATION SYMBOLS FOUND")
    print("=" * 65)

    print(sorted(overall_counts.keys()))

    print("\n" + "=" * 65)
    print("ANNOTATION EXPLORATION COMPLETED")
    print("=" * 65)

    return overall_counts, record_counts


if __name__ == "__main__":

    analyze_annotations()
