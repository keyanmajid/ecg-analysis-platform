
import numpy as np


def assess_signal_quality(
    ecg,
    min_duration_seconds=2.0,
    flatline_duration_seconds=1.0,
    amplitude_limits_mV=None,
):
    samples = np.asarray(ecg.samples, dtype=np.float64)
    fs = float(ecg.sampling_rate_hz)

    errors = []
    warnings = []

    # Basic validity
    if not np.isfinite(fs) or fs <= 0:
        errors.append("Invalid sampling rate.")

    if len(samples) < 2:
        errors.append("Insufficient ECG samples.")

    if errors:
        return {
            "status": "reject",
            "errors": errors,
            "warnings": warnings,
            "checks": {},
        }

    duration = (len(samples) - 1) / fs

    if duration < min_duration_seconds:
        errors.append(
            f"Recording too short: {duration:.2f} seconds."
        )

    # Missing and infinite samples
    finite_mask = np.isfinite(samples)
    invalid_count = int(np.sum(~finite_mask))

    if invalid_count:
        errors.append(
            f"{invalid_count} missing or non-finite samples. "
            "Repair or reject before model inference."
        )

    # Consecutive identical samples (flatline detection)
    max_flatline = 0
    current_run = 1

    for i in range(1, len(samples)):
        if finite_mask[i] and finite_mask[i - 1]:
            if samples[i] == samples[i - 1]:
                current_run += 1
            else:
                max_flatline = max(max_flatline, current_run)
                current_run = 1
        else:
            max_flatline = max(max_flatline, current_run)
            current_run = 1

    max_flatline = max(max_flatline, current_run)

    flatline_seconds = max_flatline / fs

    if flatline_seconds >= flatline_duration_seconds:
        errors.append(
            f"Possible flatline detected: "
            f"{flatline_seconds:.2f} seconds."
        )

    # Clipping detection only when device limits are known
    clipping_count = 0

    if amplitude_limits_mV is not None and np.all(finite_mask):
        lower, upper = amplitude_limits_mV

        if lower >= upper:
            raise ValueError(
                "Invalid amplitude limits: lower must be below upper."
            )

        clipping_count = int(np.sum(
            (samples <= lower) | (samples >= upper)
        ))

        if clipping_count:
            warnings.append(
                f"{clipping_count} samples reached the configured "
                "amplitude limits. Possible clipping."
            )

    checks = {
        "duration_seconds": round(duration, 3),
        "finite_samples": int(np.sum(finite_mask)),
        "invalid_samples": invalid_count,
        "max_identical_samples": int(max_flatline),
        "flatline_duration_seconds": round(flatline_seconds, 3),
        "possible_clipping_samples": clipping_count,
    }

    if errors:
        status = "reject"
    elif warnings:
        status = "review"
    else:
        status = "acceptable"

    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }
