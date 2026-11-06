
from fractions import Fraction
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from src.signal_quality import assess_signal_quality
from src.rpeak_detection import detect_rpeaks
from src.predict import ECGPredictor


TARGET_FS = 360


def resample_ecg(samples, original_fs, target_fs=TARGET_FS):
    samples = np.asarray(samples, dtype=np.float64)

    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("Invalid ECG samples.")

    if original_fs < 80:
        raise ValueError(
            "Sampling rate below 80 Hz is not supported for "
            "this 0.5–40 Hz model pipeline."
        )

    if original_fs <= 0:
        raise ValueError("Invalid original sampling rate.")

    if abs(original_fs - target_fs) < 1e-9:
        return samples.astype(np.float32)

    ratio = Fraction(
        str(target_fs / original_fs)
    ).limit_denominator(10000)

    resampled = resample_poly(
        samples,
        ratio.numerator,
        ratio.denominator
    )

    return resampled.astype(np.float32)


def analyze_ecg_signal(ecg, predictor, allow_review=True):
    """
    Analyze an ECGSignal created by src.signal_adapter.

    Returns quality information, detected peaks, HR and beat predictions.
    This is a research/portfolio prototype, not a clinical diagnosis.
    """

    quality = assess_signal_quality(ecg)

    if quality["status"] == "reject":
        raise ValueError({
            "message": "ECG signal failed quality checks.",
            "quality": quality
        })

    if quality["status"] == "review" and not allow_review:
        raise ValueError({
            "message": "ECG requires manual quality review.",
            "quality": quality
        })

    original_fs = float(ecg.sampling_rate_hz)

    samples_360 = resample_ecg(
        ecg.samples,
        original_fs,
        TARGET_FS
    )

    rpeaks = detect_rpeaks(
        samples_360,
        sampling_rate=TARGET_FS
    )

    if len(rpeaks) < 3:
        raise ValueError(
            f"Only {len(rpeaks)} R-peaks detected. "
            "At least 3 are required for rhythm features."
        )

    prediction = predictor.predict_beats(
        signal=samples_360,
        rpeaks=rpeaks,
        sampling_rate=TARGET_FS,
        already_filtered=False
    )

    rr_seconds = np.diff(rpeaks) / TARGET_FS
    heart_rate = 60.0 / np.median(rr_seconds)

    return {
        "status": "review" if quality["status"] == "review" else "processed",
        "input": ecg.summary(),
        "quality": quality,
        "processing": {
            "original_sampling_rate_hz": original_fs,
            "model_sampling_rate_hz": TARGET_FS,
            "resampled_samples": len(samples_360),
            "resampled_duration_seconds": round(
                (len(samples_360) - 1) / TARGET_FS, 3
            ),
            "detector": "experimental QRS detector",
        },
        "rhythm": {
            "median_heart_rate_bpm": round(float(heart_rate), 2),
            "median_rr_interval_seconds": round(
                float(np.median(rr_seconds)), 4
            ),
            "detected_rpeaks": int(len(rpeaks)),
        },
        "prediction": prediction
    }
