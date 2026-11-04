
"""
MIT-BIH Arrhythmia Database Downloader
---------------------------------------
Downloads the MIT-BIH Arrhythmia Database
and performs an initial recording inspection.

Run:
    python -m src.download
"""

from pathlib import Path
import argparse
import numpy as np
import wfdb


# Project root
ROOT_DIR = Path(__file__).resolve().parents[1]

# Default dataset location
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "raw" / "mit-bih-arrhythmia"


def download_dataset(output_dir: Path):
    """Download MIT-BIH Arrhythmia Database from PhysioNet."""

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MIT-BIH ARRHYTHMIA DATABASE")
    print("=" * 60)

    print("\nDownloading dataset...")
    print(f"Destination: {output_dir}")

    wfdb.dl_database(
        "mitdb",
        str(output_dir)
    )

    print("\nDownload completed!")

    return output_dir


def inspect_dataset(data_dir: Path):
    """Inspect downloaded recordings and annotations."""

    records = sorted(data_dir.glob("*.hea"))

    print("\n" + "=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    print(f"\nTotal recordings found: {len(records)}")

    if not records:
        raise FileNotFoundError(
            "No ECG header files found. Check the download."
        )

    # Inspect record 100, or the first available record
    record_id = "100" if (data_dir / "100.hea").exists() else records[0].stem

    record_path = data_dir / record_id

    # Read ECG signal
    record = wfdb.rdrecord(str(record_path))

    # Read expert annotations
    annotation = wfdb.rdann(
        str(record_path),
        extension="atr"
    )

    signal = record.p_signal

    print("\nRECORD INFORMATION")
    print("-" * 40)

    print(f"Record ID       : {record_id}")
    print(f"Sampling rate   : {record.fs} Hz")
    print(f"Number of leads : {record.n_sig}")
    print(f"Signal shape    : {signal.shape}")
    print(f"Signal names    : {record.sig_name}")
    print(f"Signal units    : {record.units}")
    print(f"Duration        : {len(signal) / record.fs / 60:.2f} minutes")

    print("\nANNOTATION INFORMATION")
    print("-" * 40)

    print(f"Total annotations: {len(annotation.sample)}")

    symbols, counts = np.unique(
        annotation.symbol,
        return_counts=True
    )

    print("\nAnnotation distribution:")

    for symbol, count in zip(symbols, counts):
        print(f"Symbol: {symbol!r:8} Count: {count}")

    print("\nDataset inspection completed!")


def main():

    parser = argparse.ArgumentParser(
        description="Download and inspect MIT-BIH ECG dataset"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR
    )

    args = parser.parse_args()

    data_dir = download_dataset(args.output_dir)

    inspect_dataset(data_dir)


if __name__ == "__main__":
    main()
