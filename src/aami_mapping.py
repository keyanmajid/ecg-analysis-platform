
"""
AAMI Five-Class Annotation Mapping
----------------------------------
Maps MIT-BIH beat annotations into AAMI classes.

Classes:
N = Normal
S = Supraventricular
V = Ventricular
F = Fusion
Q = Unknown / Paced
"""

from pathlib import Path
from collections import Counter

import wfdb


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "raw" / "mit-bih-arrhythmia"


# MIT-BIH annotation symbols -> AAMI classes
AAMI_MAPPING = {

    # Normal
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",

    # Supraventricular
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",

    # Ventricular
    "V": "V",
    "E": "V",

    # Fusion
    "F": "F",

    # Unknown / Paced
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}


CLASS_NAMES = {
    "N": "Normal",
    "S": "Supraventricular",
    "V": "Ventricular",
    "F": "Fusion",
    "Q": "Unknown / Paced",
}


# Conventional De Chazal DS1/DS2 record lists
DS1_RECORDS = [
    "101", "106", "108", "109", "112", "114",
    "115", "116", "118", "119", "122", "124",
    "201", "203", "205", "207", "208", "209",
    "215", "220", "223", "230"
]

DS2_RECORDS = [
    "100", "103", "105", "111", "113", "117",
    "121", "123", "200", "202", "210", "212",
    "213", "214", "219", "221", "222", "228",
    "231", "232", "233", "234"
]

# Standard benchmark excludes these paced recordings
PACED_RECORDS = {"102", "104", "107", "217"}


def get_aami_counts(record_ids):
    """Count AAMI classes for a selected group of records."""

    counts = Counter()
    excluded = Counter()
    total_annotations = 0

    for record_id in record_ids:

        annotation = wfdb.rdann(
            str(DATA_DIR / record_id),
            extension="atr"
        )

        for symbol in annotation.symbol:

            total_annotations += 1

            if symbol in AAMI_MAPPING:
                counts[AAMI_MAPPING[symbol]] += 1
            else:
                excluded[symbol] += 1

    return counts, excluded, total_annotations


def print_distribution(title, counts, excluded, total):

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    print(f"\nTotal original annotations: {total}")
    print(f"Mapped heartbeat annotations: {sum(counts.values())}")
    print(f"Excluded annotations: {sum(excluded.values())}")

    print("\nAAMI CLASS DISTRIBUTION")
    print("-" * 50)

    print(f"{'Class':<10}{'Description':<25}{'Count':>12}")

    for label in ["N", "S", "V", "F", "Q"]:

        count = counts[label]

        print(
            f"{label:<10}"
            f"{CLASS_NAMES[label]:<25}"
            f"{count:>12,}"
        )

    print("\nEXCLUDED SYMBOLS")
    print("-" * 50)

    for symbol, count in sorted(
        excluded.items(),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"{symbol!r:<10} {count:>10,}")


def main():

    if not DATA_DIR.exists():
        raise FileNotFoundError("MIT-BIH dataset not found.")

    available = {
        path.stem for path in DATA_DIR.glob("*.hea")
    }

    # Check standard record lists
    missing = (
        set(DS1_RECORDS + DS2_RECORDS + list(PACED_RECORDS))
        - available
    )

    if missing:
        raise FileNotFoundError(
            f"Missing expected recordings: {sorted(missing)}"
        )

    # All 48 recordings
    all_records = sorted(available)

    counts, excluded, total = get_aami_counts(all_records)

    print_distribution(
        "ALL 48 MIT-BIH RECORDINGS",
        counts,
        excluded,
        total
    )

    # Standard DS1
    counts, excluded, total = get_aami_counts(DS1_RECORDS)

    print_distribution(
        "STANDARD DS1 - TRAINING RECORDS",
        counts,
        excluded,
        total
    )

    # Standard DS2
    counts, excluded, total = get_aami_counts(DS2_RECORDS)

    print_distribution(
        "STANDARD DS2 - TEST RECORDS",
        counts,
        excluded,
        total
    )

    print("\n" + "=" * 65)
    print("AAMI MAPPING ANALYSIS COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
