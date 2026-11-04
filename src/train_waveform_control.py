
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

from src.rhythm_dataset import build_rhythm_experiment
from src.control_model import ECGWaveformControl
from src.dataset import PROCESSED_DIR
from src.split_config import EXPERIMENT_A


SEED = 42
EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3
PATIENCE = 6
POLARITY_PROB = 0.50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_record_sampler(dataset, class_names):
    record_ids = []

    for record in EXPERIMENT_A["train"]:
        with np.load(
            PROCESSED_DIR / f"record_{record}.npz",
            allow_pickle=False
        ) as archive:
            y = archive["y"]

        mask = np.isin(y, class_names)
        record_ids.extend([record] * int(mask.sum()))

    record_ids = np.asarray(record_ids)
    labels = dataset.tensors[2].numpy()

    if len(record_ids) != len(labels):
        raise ValueError(
            f"Record ID mismatch: {len(record_ids)} vs {len(labels)}"
        )

    f_index = class_names.index("F")
    f_mask = labels == f_index

    f_records = record_ids[f_mask]
    unique_records, counts = np.unique(
        f_records, return_counts=True
    )

    weights = np.ones(len(labels), dtype=np.float64)

    if len(unique_records):
        target = f_mask.sum() / len(unique_records)

        for record, count in zip(unique_records, counts):
            mask = f_mask & (record_ids == record)
            weights[mask] = target / count

    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED)
    )


def train_one_epoch(model, loader, optimizer, criterion, x_mean, x_std):
    model.train()
    total_loss = 0.0

    inversion_shift = 2.0 * x_mean / x_std

    for waveform, rhythm, labels in loader:
        waveform = waveform.to(DEVICE)
        labels = labels.to(DEVICE)

        invert = torch.rand(
            waveform.size(0), device=DEVICE
        ) < POLARITY_PROB

        waveform[invert] = -waveform[invert] - inversion_shift

        optimizer.zero_grad()

        logits = model(waveform)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    targets, predictions = [], []

    for waveform, rhythm, labels in loader:
        waveform = waveform.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(waveform)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)

        predictions.extend(logits.argmax(dim=1).cpu().numpy())
        targets.extend(labels.cpu().numpy())

    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(
        targets, predictions,
        average="macro",
        zero_division=0
    )

    return (
        total_loss / len(loader.dataset),
        accuracy,
        macro_f1,
        np.asarray(targets),
        np.asarray(predictions)
    )


def main():
    set_seed(SEED)

    data = build_rhythm_experiment(batch_size=BATCH_SIZE)

    train_ds = data["train_loader"].dataset
    sampler = build_record_sampler(train_ds, data["class_names"])

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = data["val_loader"]

    model = ECGWaveformControl(
        num_classes=len(data["class_names"])
    ).to(DEVICE)

    labels = train_ds.tensors[2].numpy()

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(data["class_names"])),
        y=labels
    )

    weights = np.sqrt(weights)
    weights = weights / weights.mean()

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            weights, dtype=torch.float32, device=DEVICE
        )
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    output_dir = Path("models/experiment_A_waveform_control")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    print("=" * 65)
    print("EXPERIMENT A — HYBRID-MATCHED WAVEFORM-ONLY CONTROL")
    print("=" * 65)
    print("Device:", DEVICE)
    print("Training samples:", len(train_ds))
    print("Validation samples:", len(val_loader.dataset))
    print("Class weights:", np.round(weights, 4))
    print("Polarity inversion:", POLARITY_PROB)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            data["x_mean"],
            data["x_std"]
        )

        val_loss, val_acc, val_f1, y_true, y_pred = validate(
            model, val_loader, criterion
        )

        scheduler.step(val_f1)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": data["class_names"],
                "x_mean": data["x_mean"],
                "x_std": data["x_std"],
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_f1,
                "seed": SEED,
                "experiment": "A",
                "architecture": "ECGWaveformControl",
                "sampling": "record_balanced_F",
                "polarity_augmentation": POLARITY_PROB,
                "class_weights": weights.tolist(),
            }, output_dir / "best_model.pt")

            print("  Saved best checkpoint.")

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

    checkpoint = torch.load(
        output_dir / "best_model.pt",
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    _, accuracy, macro_f1, y_true, y_pred = validate(
        model, val_loader, criterion
    )

    print("\\n" + "=" * 65)
    print("BEST WAVEFORM CONTROL — VALIDATION RESULTS")
    print("=" * 65)
    print("Best epoch:", checkpoint["best_epoch"])
    print("Accuracy:", round(accuracy, 4))
    print("Macro-F1:", round(macro_f1, 4))

    print("\\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(data["class_names"])),
        target_names=data["class_names"],
        zero_division=0,
        digits=4
    ))

    print("Confusion matrix:")
    print(confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(data["class_names"]))
    ))

    np.savez_compressed(
        output_dir / "validation_predictions.npz",
        y_true=y_true,
        y_pred=y_pred
    )


if __name__ == "__main__":
    main()
