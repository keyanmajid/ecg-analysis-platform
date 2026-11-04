
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader

from src.dataset import PROCESSED_DIR
from src.split_config import EXPERIMENT_A

CLASS_NAMES = ["N", "S", "V", "F"]


def load_split(records):
    signals, labels, rhythms = [], [], []

    for record in records:
        path = PROCESSED_DIR / f"record_{record}.npz"

        with np.load(path, allow_pickle=False) as data:
            X = data["X"]
            y = data["y"]
            peaks = data["rpeaks"]

        rr = np.diff(peaks).astype(np.float32) / 360.0

        previous = np.full(len(y), np.nan, dtype=np.float32)
        following = np.full(len(y), np.nan, dtype=np.float32)

        previous[1:] = rr
        following[:-1] = rr

        mean_rr = (previous + following) / 2.0
        difference = np.abs(previous - following)

        features = np.column_stack([
            previous,
            following,
            mean_rr,
            difference,
        ])

        valid = np.isin(y, CLASS_NAMES)

        signals.append(X[valid])
        labels.append(y[valid])
        rhythms.append(features[valid])

    return (
        np.concatenate(signals).astype(np.float32),
        np.concatenate(labels),
        np.concatenate(rhythms).astype(np.float32),
    )


def encode_labels(labels):
    mapping = {name: i for i, name in enumerate(CLASS_NAMES)}
    return np.array([mapping[x] for x in labels], dtype=np.int64)


def build_rhythm_experiment(batch_size=256):

    config = EXPERIMENT_A

    X_train, y_train, R_train = load_split(config["train"])
    X_val, y_val, R_val = load_split(config["validation"])
    X_test, y_test, R_test = load_split(config["test"])

    # Fit waveform normalization on training data only.
    x_mean = float(X_train.mean())
    x_std = float(X_train.std())

    X_train = (X_train - x_mean) / x_std
    X_val = (X_val - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std

    # Fit rhythm normalization on training data only.
    # Ignore boundary NaNs when calculating statistics.
    r_mean = np.nanmean(R_train, axis=0)
    r_std = np.nanstd(R_train, axis=0)
    r_std = np.maximum(r_std, 1e-6)

    # Replace unavailable boundary intervals with training means.
    def normalize_rhythm(R):
        R = np.where(np.isfinite(R), R, r_mean)
        return ((R - r_mean) / r_std).astype(np.float32)

    R_train = normalize_rhythm(R_train)
    R_val = normalize_rhythm(R_val)
    R_test = normalize_rhythm(R_test)

    def make_dataset(X, R, y):
        return TensorDataset(
            torch.tensor(X[:, None, :], dtype=torch.float32),
            torch.tensor(R, dtype=torch.float32),
            torch.tensor(encode_labels(y), dtype=torch.long),
        )

    train_ds = make_dataset(X_train, R_train, y_train)
    val_ds = make_dataset(X_val, R_val, y_val)
    test_ds = make_dataset(X_test, R_test, y_test)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=pin
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=pin
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=pin
    )

    print("Rhythm feature normalization:")
    print("Mean:", np.round(r_mean, 4))
    print("Std: ", np.round(r_std, 4))
    print("\nDataset sizes:")
    print("Train:", len(train_ds))
    print("Validation:", len(val_ds))
    print("Test:", len(test_ds))

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": CLASS_NAMES,
        "x_mean": x_mean,
        "x_std": x_std,
        "r_mean": r_mean,
        "r_std": r_std,
    }
