
from pathlib import Path
from collections import Counter
import argparse
import gc

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

from src.split_config import EXPERIMENT_A, EXPERIMENT_B


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    ROOT_DIR / "data" / "processed" / "mit-bih-216"
)

BATCH_SIZE = 256
NUM_WORKERS = 0

EXPERIMENTS = {
    "A": EXPERIMENT_A,
    "B": EXPERIMENT_B,
}


# ============================================================
# LOAD RECORDS
# ============================================================

def load_records(record_ids, class_names):

    all_signals = []
    all_labels = []

    for record_id in record_ids:

        path = PROCESSED_DIR / f"record_{record_id}.npz"

        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as data:

            X = data["X"]
            y = data["y"]

            mask = np.isin(y, class_names)

            X = X[mask]
            y = y[mask]

            all_signals.append(X)
            all_labels.append(y)

    X = np.concatenate(all_signals, axis=0)
    y = np.concatenate(all_labels, axis=0)

    return X.astype(np.float32), y


# ============================================================
# LABEL ENCODING
# ============================================================

def encode_labels(y, class_names):

    label_to_index = {
        label: index
        for index, label in enumerate(class_names)
    }

    encoded = np.asarray(
        [label_to_index[label] for label in y],
        dtype=np.int64,
    )

    return encoded


# ============================================================
# PYTORCH DATASET
# ============================================================

def create_tensor_dataset(X, y):

    # CNN input shape: (samples, channels, sequence_length)
    X_tensor = torch.from_numpy(
        X[:, np.newaxis, :]
    )

    y_tensor = torch.from_numpy(y)

    return TensorDataset(X_tensor, y_tensor)


# ============================================================
# BUILD EXPERIMENT
# ============================================================

def build_experiment(experiment_name):

    if experiment_name not in EXPERIMENTS:
        raise ValueError("Experiment must be A or B")

    config = EXPERIMENTS[experiment_name]

    class_names = config["classes"]

    print("\n" + "=" * 70)
    print(f"BUILDING EXPERIMENT {experiment_name}")
    print("=" * 70)

    # Load each split independently
    X_train, y_train_raw = load_records(
        config["train"], class_names
    )

    X_val, y_val_raw = load_records(
        config["validation"], class_names
    )

    X_test, y_test_raw = load_records(
        config["test"], class_names
    )

    # Encode labels
    y_train = encode_labels(y_train_raw, class_names)
    y_val = encode_labels(y_val_raw, class_names)
    y_test = encode_labels(y_test_raw, class_names)

    # Verify every expected class exists in training
    for index, class_name in enumerate(class_names):
        assert np.any(y_train == index), (
            f"Missing training class: {class_name}"
        )

    # --------------------------------------------------------
    # NORMALIZATION: FIT ON TRAINING DATA ONLY
    # --------------------------------------------------------

    train_mean = float(X_train.mean())
    train_std = float(X_train.std())

    if train_std < 1e-8:
        raise ValueError("Training signal standard deviation is too small")

    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    # --------------------------------------------------------
    # CLASS WEIGHTS: TRAINING DATA ONLY
    # --------------------------------------------------------

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=y_train,
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # CREATE PYTORCH DATASETS
    # --------------------------------------------------------

    train_dataset = create_tensor_dataset(X_train, y_train)
    val_dataset = create_tensor_dataset(X_val, y_val)
    test_dataset = create_tensor_dataset(X_test, y_test)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print(f"\nClasses: {class_names}")
    print(f"Training samples:   {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Test samples:       {len(test_dataset):,}")

    print("\nTraining class distribution:")

    train_counts = Counter(y_train_raw.tolist())

    for class_name in class_names:
        print(
            f"{class_name}: {train_counts[class_name]:,}"
        )

    print("\nTraining class weights:")

    for class_name, weight in zip(
        class_names, class_weights.tolist()
    ):
        print(f"{class_name}: {weight:.4f}")

    print("\nTraining normalization statistics:")
    print(f"Mean: {train_mean:.8f}")
    print(f"Std:  {train_std:.8f}")

    print("\nTensor shapes:")
    print(f"Input: (batch, 1, 216)")
    print(f"Output classes: {len(class_names)}")

    print("\nDataset loaders created successfully.")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": class_names,
        "class_weights": class_weights,
        "train_mean": train_mean,
        "train_std": train_std,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=["A", "B"],
        required=True,
    )

    args = parser.parse_args()

    result = build_experiment(args.experiment)

    print("\nBATCH TEST")

    signals, labels = next(
        iter(result["train_loader"])
    )

    print(f"Signals shape: {signals.shape}")
    print(f"Labels shape:  {labels.shape}")
    print(f"Signal dtype:  {signals.dtype}")
    print(f"Label dtype:   {labels.dtype}")

    print("\n" + "=" * 70)
    print(f"EXPERIMENT {args.experiment} DATASET READY")
    print("=" * 70)


if __name__ == "__main__":
    main()
