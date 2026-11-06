
from pathlib import Path

import numpy as np
import torch

from scipy.signal import butter, sosfiltfilt

from src.hybrid_model import ECGHybridCNN


PRE_SAMPLES = 72
POST_SAMPLES = 144
WINDOW_SIZE = 216
FS = 360
CLASS_NAMES = ["N", "S", "V", "F"]


class ECGPredictor:

    def __init__(self, checkpoint_path, device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        # Trusted project checkpoint containing NumPy metadata.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        if checkpoint.get("architecture") != "ECGHybridCNN":
            raise ValueError("Checkpoint is not an ECGHybridCNN model.")

        self.class_names = checkpoint["class_names"]
        self.x_mean = float(checkpoint["x_mean"])
        self.x_std = float(checkpoint["x_std"])
        self.r_mean = np.asarray(
            checkpoint["r_mean"], dtype=np.float32
        )
        self.r_std = np.asarray(
            checkpoint["r_std"], dtype=np.float32
        )

        if self.x_std <= 0 or np.any(self.r_std <= 0):
            raise ValueError("Invalid checkpoint normalization values.")

        self.model = ECGHybridCNN(
            num_classes=len(self.class_names)
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.checkpoint_metadata = {
            "architecture": checkpoint["architecture"],
            "best_epoch": checkpoint["best_epoch"],
            "best_val_macro_f1": checkpoint["best_val_macro_f1"],
            "sampling": checkpoint["sampling"],
        }

    @staticmethod
    def filter_signal(signal, sampling_rate=FS):
        signal = np.asarray(signal, dtype=np.float64)

        if signal.ndim != 1 or len(signal) < 30:
            raise ValueError("Expected a one-dimensional ECG signal.")

        if not np.isfinite(signal).all():
            raise ValueError("ECG signal contains non-finite values.")

        if sampling_rate <= 80:
            raise ValueError("Sampling rate must exceed 80 Hz.")

        sos = butter(
            4,
            [0.5, 40.0],
            btype="bandpass",
            fs=sampling_rate,
            output="sos"
        )

        return sosfiltfilt(sos, signal).astype(np.float32)

    def predict_beats(
        self,
        signal,
        rpeaks,
        sampling_rate=FS,
        already_filtered=False,
        batch_size=256
    ):
        """
        signal: one-dimensional ECG samples.
        rpeaks: detected R-peak sample positions in the same signal.
        sampling_rate: input signal sampling rate.
        already_filtered: True if the signal has already been filtered.
        """

        signal = np.asarray(signal, dtype=np.float32)
        rpeaks = np.asarray(rpeaks, dtype=np.int64)

        if signal.ndim != 1 or len(signal) < WINDOW_SIZE:
            raise ValueError("Invalid ECG signal length.")

        if not np.isfinite(signal).all():
            raise ValueError("ECG signal contains non-finite values.")

        if sampling_rate != FS:
            raise ValueError(
                "This checkpoint expects 360 Hz input. "
                "Resampling must be performed before inference."
            )

        if rpeaks.ndim != 1 or len(rpeaks) < 3:
            raise ValueError("At least three R-peaks are required.")

        if np.any(np.diff(rpeaks) <= 0):
            raise ValueError(
                "R-peaks must be unique and strictly increasing."
            )

        if rpeaks[0] < 0 or rpeaks[-1] >= len(signal):
            raise ValueError("An R-peak is outside the signal boundaries.")

        if not already_filtered:
            signal = self.filter_signal(signal, sampling_rate)

        # Extract beat windows.
        beats = []
        valid_peaks = []
        valid_peak_indices = []

        for i, peak in enumerate(rpeaks):
            start = int(peak) - PRE_SAMPLES
            end = int(peak) + POST_SAMPLES

            if start < 0 or end > len(signal):
                continue

            beats.append(signal[start:end])
            valid_peaks.append(int(peak))
            valid_peak_indices.append(i)

        if not beats:
            raise ValueError("No complete beat windows could be extracted.")

        X = np.asarray(beats, dtype=np.float32)

        # Reproduce training waveform normalization.
        X = (X - self.x_mean) / self.x_std

        # Calculate rhythm features from detected R-peaks.
        rr = np.diff(rpeaks).astype(np.float32) / FS

        previous = np.full(len(rpeaks), np.nan, dtype=np.float32)
        following = np.full(len(rpeaks), np.nan, dtype=np.float32)

        previous[1:] = rr
        following[:-1] = rr

        mean_rr = (previous + following) / 2.0
        difference = np.abs(previous - following)

        all_features = np.column_stack([
            previous,
            following,
            mean_rr,
            difference
        ])

        R = all_features[valid_peak_indices]

        # Match training behavior for missing boundary intervals.
        R = np.where(np.isfinite(R), R, self.r_mean)

        R = ((R - self.r_mean) / self.r_std).astype(np.float32)

        waveform_tensor = torch.from_numpy(
            X[:, None, :]
        ).to(self.device)

        rhythm_tensor = torch.from_numpy(R).to(self.device)

        probabilities = []

        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                end = start + batch_size

                logits = self.model(
                    waveform_tensor[start:end],
                    rhythm_tensor[start:end]
                )

                probs = torch.softmax(logits, dim=1)
                probabilities.append(probs.cpu().numpy())

        probabilities = np.concatenate(probabilities, axis=0)

        results = []

        for peak, probs in zip(valid_peaks, probabilities):
            predicted_index = int(np.argmax(probs))

            results.append({
                "rpeak_sample": peak,
                "predicted_class": self.class_names[predicted_index],
                "confidence": float(probs[predicted_index]),
                "probabilities": {
                    name: float(prob)
                    for name, prob in zip(self.class_names, probs)
                }
            })

        return {
            "sampling_rate_hz": sampling_rate,
            "total_detected_rpeaks": len(rpeaks),
            "classified_beats": len(results),
            "excluded_boundary_beats": len(rpeaks) - len(results),
            "results": results
        }
