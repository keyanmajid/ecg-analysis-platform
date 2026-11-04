"""
ECG Experimental Split Configuration
-------------------------------------

Experiment A:
Conventional MIT-BIH four-class benchmark.
Patient overlap between records 201 and 202 is removed.

Experiment B:
Extended five-class patient-grouped experiment.
All 48 recordings are assigned without patient leakage.
"""

from collections import Counter

from src.aami_mapping import (
    AAMI_MAPPING,
    CLASS_NAMES,
    DS1_RECORDS,
    DS2_RECORDS,
    PACED_RECORDS,
    DATA_DIR,
)

import wfdb


# ============================================================
# EXPERIMENT A — FOUR-CLASS BENCHMARK
# ============================================================

EXPERIMENT_A_VALIDATION_RECORDS = [
    "108", "114", "124", "205"
]

EXPERIMENT_A = {

    "train": [
        r for r in DS1_RECORDS
        if r not in EXPERIMENT_A_VALIDATION_RECORDS
    ],

    "validation": EXPERIMENT_A_VALIDATION_RECORDS,

    "test": [
        r for r in DS2_RECORDS
        if r != "202"
    ],

    "classes": ["N", "S", "V", "F"],
}

# ============================================================
# EXPERIMENT B — FIVE-CLASS PATIENT-GROUPED
# ============================================================

# Patient groups must never cross partitions.
# Records 201 and 202 are treated as one patient.

PATIENT_GROUPS = {
    "patient_201_202": ["201", "202"],
}

all_records = (
    set(DS1_RECORDS)
    | set(DS2_RECORDS)
    | set(PACED_RECORDS)
)

for record in all_records:

    if record not in ("201", "202"):
        PATIENT_GROUPS[f"patient_{record}"] = [record]


# Initial deterministic assignment.
# This is a provisional allocation, not a class-balanced split.

EXPERIMENT_B = {

    "train": [
        "101", "106", "108", "109", "112", "114",
        "115", "116", "118", "119", "122", "124",
        "200", "201", "202", "203", "207",
        "208", "209", "102", "104"
    ],

    "validation": [
        "111", "113", "117", "121", "123",
        "210", "212", "214", "215", "219", "220",
        "107", "205", "232"
    ],

    "test": [
        "100", "103", "105", "221", "222",
        "223", "228", "230", "231", "233",
        "234", "217", "213"
    ],

    "classes": ["N", "S", "V", "F", "Q"],
}


# ============================================================
# VALIDATION
# ============================================================

def patient_id(record):

    if record in ("201", "202"):
        return "patient_201_202"

    return f"patient_{record}"


def validate_split(name, split):

    print("\n" + "=" * 65)
    print(f"{name} VALIDATION")
    print("=" * 65)

    train = set(split["train"])
    validation = set(split["validation"])
    test = set(split["test"])

    # Check record overlap.

    assert not train & validation, "Train/validation record overlap!"
    assert not train & test, "Train/test record overlap!"
    assert not validation & test, "Validation/test record overlap!"

    # Check that every record exists.

    available = {
        path.stem for path in DATA_DIR.glob("*.hea")
    }

    assigned = train | validation | test

    missing = assigned - available

    assert not missing, f"Missing records: {sorted(missing)}"

    # Check patient leakage.

    train_patients = {patient_id(r) for r in train}
    validation_patients = {patient_id(r) for r in validation}
    test_patients = {patient_id(r) for r in test}

    assert not train_patients & validation_patients
    assert not train_patients & test_patients
    assert not validation_patients & test_patients

    print("Train records:", len(train))
    print("Validation records:", len(validation))
    print("Test records:", len(test))

    print("Patient overlap: NONE")

    # Calculate class distributions.

    for partition_name, records in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:

        counts = Counter()

        for record in sorted(records):

            annotation = wfdb.rdann(
                str(DATA_DIR / record),
                "atr"
            )

            for symbol in annotation.symbol:

                if symbol in AAMI_MAPPING:

                    label = AAMI_MAPPING[symbol]

                    if label in split["classes"]:
                        counts[label] += 1

        print(f"\n{partition_name} DISTRIBUTION")

        for label in split["classes"]:
            print(
                f"{label} ({CLASS_NAMES[label]}): "
                f"{counts[label]:,}"
            )

    print("\nSplit validation successful.")


def main():

    validate_split("EXPERIMENT A", EXPERIMENT_A)
    validate_split("EXPERIMENT B", EXPERIMENT_B)


if __name__ == "__main__":
    main()