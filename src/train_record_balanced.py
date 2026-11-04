
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)

from src.dataset import build_experiment, PROCESSED_DIR
from src.split_config import EXPERIMENT_A
from src.model import ECGCNN
from src.train import (
    set_seed,
    train_one_epoch,
    validate,
    SEED,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    POLARITY_INVERSION_PROBABILITY,
)

from pathlib import Path


def build_record_balanced_loader(data):
    class_names = data["class_names"]
    dataset = data["train_loader"].dataset

    labels = dataset.tensors[1].numpy()
    f_index = class_names.index("F")

    # Reconstruct record IDs in exactly the same order as load_records().
    record_ids = []

    for record in EXPERIMENT_A["train"]:
        path = PROCESSED_DIR / f"record_{record}.npz"

        with np.load(path, allow_pickle=False) as archive:
            y = archive["y"]

        mask = np.isin(y, class_names)
        record_ids.extend([record] * int(mask.sum()))

    record_ids = np.asarray(record_ids)

    if len(record_ids) != len(labels):
        raise ValueError(
            f"Record ID mismatch: {len(record_ids)} vs {len(labels)}"
        )

    f_records = sorted(set(record_ids[labels == f_index].tolist()))

    if not f_records:
        raise ValueError("No F-class records found.")

    f_counts = {
        record: int(np.sum(
            (record_ids == record) & (labels == f_index)
        ))
        for record in f_records
    }

    total_f = sum(f_counts.values())
    target = total_f / len(f_records)

    # Non-F samples retain weight 1.
    # F samples receive weights that equalize record contributions.
    weights = np.ones(len(labels), dtype=np.float64)

    for record, count in f_counts.items():
        mask = (record_ids == record) & (labels == f_index)
        weights[mask] = target / count

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    loader = DataLoader(
        dataset,
        batch_size=data["train_loader"].batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("\nRECORD-BALANCED F SAMPLING")
    print("-" * 55)
    print(f"Total original F samples: {total_f}")
    print(f"F-containing records: {len(f_records)}")
    print(f"Expected F contribution per record: {target:.2f}")

    for record in f_records:
        print(
            f"Record {record}: "
            f"{f_counts[record]} original F beats | "
            f"Expected sampled contribution: {target:.2f}"
        )

    print(
        "\nExpected total F sampling weight:",
        round(sum(weights[labels == f_index]), 2),
    )

    return loader


def train():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("EXPERIMENT A — RECORD-BALANCED F + POLARITY AUGMENTATION")
    print("=" * 70)
    print("Device:", device)

    data = build_experiment("A")

    train_loader = build_record_balanced_loader(data)
    val_loader = data["val_loader"]

    class_names = data["class_names"]
    class_weights = data["class_weights"]
    train_mean = data["train_mean"]
    train_std = data["train_std"]

    model = ECGCNN(
        num_classes=len(class_names),
        dropout=0.3,
    ).to(device)

    softened_weights = torch.sqrt(class_weights)
    softened_weights /= softened_weights.mean()

    criterion = nn.CrossEntropyLoss(
        weight=softened_weights.to(device)
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    save_dir = (
        Path("models") / "experiment_A_record_balanced"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / "best_model.pt"

    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    print("\nTraining configuration:")
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print(
        "Polarity inversion probability:",
        POLARITY_INVERSION_PROBABILITY,
    )

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train_mean,
            train_std,
        )

        (
            val_loss,
            val_accuracy,
            val_macro_f1,
            _,
            _,
        ) = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(val_macro_f1)

        improved = val_macro_f1 > best_f1

        if improved:
            best_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_macro_f1": best_f1,
                    "class_names": class_names,
                    "train_mean": train_mean,
                    "train_std": train_std,
                    "seed": SEED,
                    "experiment": "A",
                    "weighting": "sqrt_balanced",
                    "class_weights": softened_weights.cpu(),
                    "augmentation": "random_polarity_inversion_50_percent",
                    "polarity_inversion_probability": 0.5,
                    "sampling": "record_balanced_F_weighted_random_sampler",
                },
                checkpoint_path,
            )

            status = "BEST"
        else:
            patience_counter += 1
            status = ""

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"LR: {current_lr:.2e} | {status}"
        )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping triggered.")
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    (
        _,
        val_accuracy,
        val_macro_f1,
        targets,
        predictions,
    ) = validate(
        model,
        val_loader,
        criterion,
        device,
    )

    print("\n" + "=" * 70)
    print("BEST RECORD-BALANCED CHECKPOINT")
    print("=" * 70)
    print("Best epoch:", best_epoch)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation macro-F1: {val_macro_f1:.4f}")

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        targets,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    ))

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(
        targets,
        predictions,
        labels=np.arange(len(class_names)),
    ))

    print("\nCheckpoint:", checkpoint_path)


if __name__ == "__main__":
    train()
