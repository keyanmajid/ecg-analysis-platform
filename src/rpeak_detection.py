
import numpy as np

from scipy.signal import butter, sosfiltfilt, find_peaks


def detect_rpeaks(signal, sampling_rate=360):
    """
    Preliminary QRS/R-peak detector.

    Returns sample positions in the original input signal.
    Requires a finite, one-dimensional ECG signal.
    """

    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1 or len(signal) < int(2 * sampling_rate):
        raise ValueError("At least 2 seconds of ECG are required.")

    if not np.isfinite(signal).all():
        raise ValueError("Signal contains missing or non-finite samples.")

    if sampling_rate <= 40:
        raise ValueError("Sampling rate must exceed 40 Hz.")

    # QRS-focused bandpass.
    sos = butter(
        3, [5.0, min(18.0, sampling_rate * 0.4)],
        btype="bandpass",
        fs=sampling_rate,
        output="sos"
    )

    filtered = sosfiltfilt(sos, signal)

    # Derivative energy and moving-window integration.
    derivative = np.diff(filtered, prepend=filtered[0])
    energy = derivative ** 2

    window = max(1, int(0.12 * sampling_rate))
    integrated = np.convolve(
        energy,
        np.ones(window, dtype=np.float64) / window,
        mode="same"
    )

    median = np.median(integrated)
    mad = np.median(np.abs(integrated - median))
    threshold = median + 3.0 * max(1.4826 * mad, 1e-12)

    candidates, properties = find_peaks(
        integrated,
        height=threshold,
        distance=int(0.25 * sampling_rate),
        prominence=max(1.5 * mad, 1e-12)
    )

    if len(candidates) == 0:
        raise ValueError(
            "No candidate QRS complexes detected. "
            "Check signal quality and lead selection."
        )

    # Refine each candidate against the filtered ECG.
    search_radius = max(1, int(0.10 * sampling_rate))
    refined = []

    for candidate in candidates:
        left = max(0, candidate - search_radius)
        right = min(len(filtered), candidate + search_radius + 1)

        local = filtered[left:right]

        # Absolute amplitude accommodates either QRS polarity.
        peak = left + int(np.argmax(np.abs(local)))
        refined.append(peak)

    # Remove duplicate or too-close refined peaks.
    refined = sorted(set(refined))
    min_distance = int(0.25 * sampling_rate)

    final_peaks = []

    for peak in refined:
        if not final_peaks or peak - final_peaks[-1] >= min_distance:
            final_peaks.append(peak)
        elif abs(filtered[peak]) > abs(filtered[final_peaks[-1]]):
            final_peaks[-1] = peak

    return np.asarray(final_peaks, dtype=np.int64)
