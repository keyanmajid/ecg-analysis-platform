
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)

from src.dataset import build_experiment
from src.model import ECGCNN


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 6

POLARITY_INVERSION_PROBABILITY = 0.5

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train_mean,
    train_std,
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    # Correct inversion after training-set normalization:
    # normalized_inverted = -normalized_original - 2*mean/std

    polarity_offset = 2.0 * train_mean / train_std

    for signals, labels in loader:

        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ----------------------------------------------------
        # RANDOM POLARITY INVERSION
        # Applied independently to 50% of training beats
        # ----------------------------------------------------

        flip_mask = (
            torch.rand(signals.size(0), device=device)
            < POLARITY_INVERSION_PROBABILITY
        )

        if flip_mask.any():

            signals = signals.clone()

            signals[flip_mask] = (
                -signals[flip_mask] - polarity_offset
            )

        optimizer.zero_grad(set_to_none=True)

        outputs = model(signals)

        loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_targets = []

    for signals, labels in loader:

        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # No augmentation during validation
        outputs = model(signals)

        loss = criterion(outputs, labels)

        predictions = outputs.argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_targets.extend(
            labels.cpu().numpy()
        )

    avg_loss = total_loss / total_samples

    accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return (
        avg_loss,
        accuracy,
        macro_f1,
        np.asarray(all_targets),
        np.asarray(all_predictions),
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train(experiment):

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print(f"ECG CNN — EXPERIMENT {experiment}")
    print("=" * 70)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # BUILD DATASETS
    # --------------------------------------------------------

    data = build_experiment(experiment)

    train_loader = data["train_loader"]
    val_loader = data["val_loader"]

    class_names = data["class_names"]
    class_weights = data["class_weights"]

    train_mean = data["train_mean"]
    train_std = data["train_std"]

    # --------------------------------------------------------
    # INITIALIZE MODEL
    # --------------------------------------------------------

    model = ECGCNN(
        num_classes=len(class_names),
        dropout=0.3,
    ).to(device)

    # --------------------------------------------------------
    # SOFTENED CLASS WEIGHTS
    # --------------------------------------------------------

    softened_weights = torch.sqrt(class_weights)

    softened_weights = (
        softened_weights / softened_weights.mean()
    )

    print("\nSoftened training class weights:")

    for name, weight in zip(
        class_names,
        softened_weights.tolist(),
    ):
        print(f"{name}: {weight:.4f}")

    criterion = nn.CrossEntropyLoss(
        weight=softened_weights.to(device)
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CHECKPOINT CONFIGURATION
    # --------------------------------------------------------

    # Separate directory preserves the previous experiment
    save_dir = (
        MODEL_DIR /
        f"experiment_{experiment}_polarity50"
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = save_dir / "best_model.pt"

    best_f1 = -1.0
    best_epoch = 0

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # TRAINING CONFIGURATION
    # --------------------------------------------------------

    print("\nStarting training...")

    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")

    print(
        "Polarity inversion probability: "
        f"{POLARITY_INVERSION_PROBABILITY:.2f}"
    )

    print(f"Training mean: {train_mean:.8f}")
    print(f"Training std: {train_std:.8f}")

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

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

        current_lr = optimizer.param_groups[0]["lr"]

        improved = val_macro_f1 > best_f1

        if improved:

            best_f1 = val_macro_f1
            best_epoch = epoch

            epochs_without_improvement = 0

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
                    "experiment": experiment,

                    "weighting": "sqrt_balanced",

                    "class_weights": softened_weights.cpu(),

                    "augmentation": (
                        "random_polarity_inversion_50_percent"
                    ),

                    "polarity_inversion_probability": (
                        POLARITY_INVERSION_PROBABILITY
                    ),
                },
                checkpoint_path,
            )

            status = "BEST"

        else:

            epochs_without_improvement += 1
            status = ""

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"LR: {current_lr:.2e} | {status}"
        )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print("\nEarly stopping triggered.")
            break

    # --------------------------------------------------------
    # LOAD BEST CHECKPOINT
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    (
        val_loss,
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
    print("BEST CHECKPOINT RESULTS")
    print("=" * 70)

    print(f"Best epoch: {checkpoint['epoch']}")

    print(
        f"Validation accuracy: {val_accuracy:.4f}"
    )

    print(
        f"Validation macro-F1: {val_macro_f1:.4f}"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("\nCLASSIFICATION REPORT")

    print(
        classification_report(
            targets,
            predictions,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # PREDICTION DISTRIBUTION
    # --------------------------------------------------------

    print("\nPREDICTION DISTRIBUTION")

    for index, class_name in enumerate(class_names):

        actual_count = int(
            np.sum(targets == index)
        )

        predicted_count = int(
            np.sum(predictions == index)
        )

        print(
            f"{class_name}: "
            f"Actual={actual_count}, "
            f"Predicted={predicted_count}"
        )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    print("\nCONFUSION MATRIX")

    cm = confusion_matrix(
        targets,
        predictions,
        labels=np.arange(len(class_names)),
    )

    print(cm)

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print(f"\nCheckpoint saved: {checkpoint_path}")

    print("\nTraining complete.")

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=["A", "B"],
        required=True,
    )

    args = parser.parse_args()

    train(args.experiment)


if __name__ == "__main__":
    main()
