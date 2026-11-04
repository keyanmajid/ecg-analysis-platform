
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.dataset import build_experiment
from src.model import ECGCNN


def diagnose(experiment="A"):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    data = build_experiment(experiment)

    class_names = data["class_names"]
    class_weights = data["class_weights"].to(device)

    model = ECGCNN(
        num_classes=len(class_names),
        dropout=0.3,
    ).to(device)

    checkpoint_path = (
        f"models/experiment_{experiment}/best_model.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_targets = []
    all_raw = []
    all_corrected = []

    with torch.no_grad():

        for signals, labels in data["val_loader"]:

            signals = signals.to(device)

            logits = model(signals)

            # Original weighted-loss predictions
            raw_predictions = logits.argmax(dim=1)

            # Correct for the class-weight-induced shift
            adjusted_logits = (
                logits - torch.log(class_weights).unsqueeze(0)
            )

            corrected_predictions = adjusted_logits.argmax(dim=1)

            all_targets.extend(labels.numpy())
            all_raw.extend(raw_predictions.cpu().numpy())
            all_corrected.extend(
                corrected_predictions.cpu().numpy()
            )

    targets = np.asarray(all_targets)
    raw = np.asarray(all_raw)
    corrected = np.asarray(all_corrected)

    for title, predictions in [
        ("ORIGINAL PREDICTIONS", raw),
        ("PRIOR-CORRECTED PREDICTIONS", corrected),
    ]:

        print("\n" + "=" * 65)
        print(title)
        print("=" * 65)

        print(
            "Accuracy:",
            round(accuracy_score(targets, predictions), 4),
        )

        print("\nClassification report:")

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

        print("Prediction distribution:")

        for index, name in enumerate(class_names):
            print(
                f"{name}: "
                f"Actual={np.sum(targets == index)}, "
                f"Predicted={np.sum(predictions == index)}"
            )

        print("\nConfusion matrix:")

        print(
            confusion_matrix(
                targets,
                predictions,
                labels=np.arange(len(class_names)),
            )
        )


if __name__ == "__main__":
    diagnose("A")
