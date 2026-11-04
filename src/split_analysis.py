
from collections import Counter, defaultdict

from src.aami_mapping import (
    DATA_DIR,
    AAMI_MAPPING,
    CLASS_NAMES,
    DS1_RECORDS,
    DS2_RECORDS,
    PACED_RECORDS,
)

import wfdb


def analyze_record(record):
    annotation = wfdb.rdann(
        str(DATA_DIR / record),
        "atr"
    )

    counts = Counter(
        AAMI_MAPPING[symbol]
        for symbol in annotation.symbol
        if symbol in AAMI_MAPPING
    )

    return counts


def main():

    print("=" * 65)
    print("MIT-BIH PATIENT SPLIT AND Q-CLASS AUDIT")
    print("=" * 65)

    record_counts = {}

    for record in sorted(DS1_RECORDS + DS2_RECORDS + list(PACED_RECORDS)):
        record_counts[record] = analyze_record(record)

    print("\n1. RECORD-LEVEL CLASS DISTRIBUTION")

    for record, counts in sorted(record_counts.items()):
        print(
            f"{record}: "
            + " | ".join(
                f"{class_name}={counts.get(class_name, 0)}"
                for class_name in CLASS_NAMES
            )
        )

    print("\n2. DS1 / DS2 RECORD OVERLAP")

    overlap = set(DS1_RECORDS) & set(DS2_RECORDS)
    print("Overlapping records:", sorted(overlap) if overlap else "None")

    print("\n3. PATIENT-LEVEL OVERLAP")

    def patient_id(record):
        if record in ("201", "202"):
            return "patient_201_202"
        return f"patient_{record}"

    ds1_patients = {patient_id(r) for r in DS1_RECORDS}
    ds2_patients = {patient_id(r) for r in DS2_RECORDS}

    patient_overlap = ds1_patients & ds2_patients

    print("DS1 patients:", len(ds1_patients))
    print("DS2 patients:", len(ds2_patients))
    print("Shared patients:", sorted(patient_overlap))

    print("\n4. Q-CLASS DISTRIBUTION")

    for record, counts in sorted(record_counts.items()):
        q_count = counts.get("Q", 0)

        if q_count > 0:
            paced = " (PACED)" if record in PACED_RECORDS else ""
            print(f"Record {record}: Q={q_count}{paced}")

    print("\n5. SPLIT-LEVEL DISTRIBUTION")

    for name, records in [
        ("DS1", DS1_RECORDS),
        ("DS2", DS2_RECORDS),
        ("PACED", sorted(PACED_RECORDS)),
    ]:
        total = Counter()

        for record in records:
            total.update(record_counts[record])

        print(f"\n{name} — {len(records)} records")

        for class_name in CLASS_NAMES:
            print(f"{class_name}: {total.get(class_name, 0)}")

    print("\nAudit completed.")


if __name__ == "__main__":
    main()