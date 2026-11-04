
from pathlib import Path
from collections import Counter
import csv

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt

from src.aami_mapping import DATA_DIR, AAMI_MAPPING


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = DATA_DIR.parent.parent / "processed" / "mit-bih-216"

FS = 360
LOWCUT = 0.5
HIGHCUT = 40.0
FILTER_ORDER = 4

PRE_SAMPLES = int(0.2 * FS)   # 72 samples
POST_SAMPLES = int(0.4 * FS)  # 144 samples
WINDOW_SIZE = PRE_SAMPLES + POST_SAMPLES  # 216 samples

ANNOTATION_EXTENSION = "atr"

# Prefer MLII. Use V5 only when MLII is unavailable.
FALLBACK_LEAD = "V5"


# ============================================================
# SIGNAL FILTER
# ============================================================

def filter_signal(signal):
    sos = butter(
        FILTER_ORDER,
        [LOWCUT, HIGHCUT],
        btype="bandpass",
        fs=FS,
        output="sos",
    )

    return sosfiltfilt(sos, signal).astype(np.float32)


# ============================================================
# LEAD SELECTION
# ============================================================

def select_lead(record_path):
    header = wfdb.rdheader(str(record_path))

    lead_names = [name.strip() for name in header.sig_name]

    for index, name in enumerate(lead_names):
        if name.upper() == "MLII":
            return index, name

    for index, name in enumerate(lead_names):
        if name.upper() == FALLBACK_LEAD:
            return index, name

    raise ValueError(
        f"No MLII or V5 lead found in {record_path.name}: {lead_names}"
    )


# ============================================================
# PROCESS ONE RECORD
# ============================================================

def process_record(record_id):

    record_path = DATA_DIR / record_id

    # Select lead
    lead_index, lead_name = select_lead(record_path)

    # Load physical ECG signal (mV)
    record = wfdb.rdrecord(
        str(record_path),
        channels=[lead_index],
        physical=True,
    )

    signal = record.p_signal[:, 0].astype(np.float64)

    if record.fs != FS:
        raise ValueError(
            f"Unexpected sampling rate in record {record_id}: {record.fs}"
        )

    if not np.isfinite(signal).all():
        raise ValueError(f"Non-finite signal values in record {record_id}")

    # Filter the complete recording before extracting beats
    filtered_signal = filter_signal(signal)

    # Load expert beat annotations
    annotation = wfdb.rdann(
        str(record_path),
        ANNOTATION_EXTENSION,
    )

    beats = []
    labels = []
    symbols = []
    rpeak_positions = []

    unmapped_count = 0
    boundary_count = 0

    for sample, symbol in zip(
        annotation.sample,
        annotation.symbol,
    ):

        # Keep only symbols included in our AAMI mapping
        if symbol not in AAMI_MAPPING:
            unmapped_count += 1
            continue

        # Exclude beats whose windows cross recording boundaries
        start = int(sample) - PRE_SAMPLES
        end = int(sample) + POST_SAMPLES

        if start < 0 or end > len(filtered_signal):
            boundary_count += 1
            continue

        beat = filtered_signal[start:end]

        if len(beat) != WINDOW_SIZE:
            raise ValueError(
                f"Invalid beat length in record {record_id}"
            )

        beats.append(beat)
        labels.append(AAMI_MAPPING[symbol])
        symbols.append(symbol)
        rpeak_positions.append(int(sample))

    if not beats:
        raise ValueError(f"No valid beats extracted from {record_id}")

    X = np.asarray(beats, dtype=np.float32)
    y = np.asarray(labels, dtype="U1")

    # Save one file per record to preserve record-level separation
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"record_{record_id}.npz"

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        symbols=np.asarray(symbols, dtype="U2"),
        rpeaks=np.asarray(rpeak_positions, dtype=np.int64),
        record_id=np.asarray(record_id),
        selected_lead=np.asarray(lead_name),
        sampling_rate=np.asarray(FS),
    )

    return {
        "record": record_id,
        "lead": lead_name,
        "beats": len(X),
        "unmapped": unmapped_count,
        "boundary": boundary_count,
        "N": int(np.sum(y == "N")),
        "S": int(np.sum(y == "S")),
        "V": int(np.sum(y == "V")),
        "F": int(np.sum(y == "F")),
        "Q": int(np.sum(y == "Q")),
    }


# ============================================================
# PROCESS ALL RECORDS
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    record_files = sorted(DATA_DIR.glob("*.hea"))

    if len(record_files) != 48:
        raise ValueError(
            f"Expected 48 MIT-BIH records, found {len(record_files)}"
        )

    results = []

    print("=" * 75)
    print("MIT-BIH BEAT PREPROCESSING")
    print("=" * 75)
    print(f"Records: {len(record_files)}")
    print(f"Sampling rate: {FS} Hz")
    print(f"Filter: {LOWCUT}-{HIGHCUT} Hz Butterworth")
    print(f"Beat window: {WINDOW_SIZE} samples")
    print(f"R-peak position: {PRE_SAMPLES}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 75)

    for index, path in enumerate(record_files, start=1):

        record_id = path.stem

        result = process_record(record_id)
        results.append(result)

        print(
            f"[{index:02d}/48] "
            f"{record_id} | "
            f"Lead: {result['lead']:4s} | "
            f"Beats: {result['beats']:5d} | "
            f"Unmapped: {result['unmapped']:4d} | "
            f"Boundary: {result['boundary']:3d}"
        )

    # Save a record-level audit summary
    summary_path = OUTPUT_DIR / "preprocessing_summary.csv"

    fieldnames = list(results[0].keys())

    with open(summary_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Aggregate class counts
    totals = Counter()

    for result in results:
        for class_name in ["N", "S", "V", "F", "Q"]:
            totals[class_name] += result[class_name]

    print("\n" + "=" * 75)
    print("PREPROCESSING COMPLETE")
    print("=" * 75)

    print(f"Records processed: {len(results)}")
    print(f"Total beats: {sum(r['beats'] for r in results):,}")

    print("\nAAMI CLASS DISTRIBUTION:")

    for class_name in ["N", "S", "V", "F", "Q"]:
        print(f"{class_name}: {totals[class_name]:,}")

    print("\nLEAD SELECTION:")

    for lead in sorted(set(r["lead"] for r in results)):
        count = sum(r["lead"] == lead for r in results)
        print(f"{lead}: {count} records")

    print("\nAUDIT TOTALS:")
    print(f"Unmapped annotations: {sum(r['unmapped'] for r in results):,}")
    print(f"Boundary exclusions: {sum(r['boundary'] for r in results):,}")

    print(f"\nSummary saved: {summary_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
