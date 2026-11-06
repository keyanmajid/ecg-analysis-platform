
# Paste the complete Python code from my previous message here.



from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


UNIT_TO_MV = {
    "mv": 1.0,
    "millivolt": 1.0,
    "millivolts": 1.0,
    "v": 1000.0,
    "volt": 1000.0,
    "volts": 1000.0,
    "uv": 0.001,
    "µv": 0.001,
    "μv": 0.001,
    "microvolt": 0.001,
    "microvolts": 0.001,
}


@dataclass
class ECGSignal:
    samples: np.ndarray
    sampling_rate_hz: float
    original_sampling_rate_hz: float
    lead: Optional[str]
    unit: str
    source: Optional[str]
    duration_seconds: float
    input_format: str
    extraction_method: str
    warnings: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def summary(self):
        samples = np.asarray(self.samples)

        return {
            "samples": int(len(samples)),
            "sampling_rate_hz": self.sampling_rate_hz,
            "original_sampling_rate_hz":
                self.original_sampling_rate_hz,
            "lead": self.lead,
            "unit": self.unit,
            "source": self.source,
            "duration_seconds": round(self.duration_seconds, 3),
            "input_format": self.input_format,
            "extraction_method": self.extraction_method,
            "missing_samples": int(
                np.count_nonzero(~np.isfinite(samples))
            ),
            "warnings": self.warnings,
            "metadata": self.metadata,
        }




def _infer_lead(column):
    normalized = (
        str(column).upper()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )

    known_leads = {
        "LEADI": "I",
        "LEADII": "II",
        "LEADIII": "III",
        "MLII": "MLII",
        "V1": "V1",
        "V2": "V2",
        "V3": "V3",
        "V4": "V4",
        "V5": "V5",
        "V6": "V6",
        "AVR": "aVR",
        "AVL": "aVL",
        "AVF": "aVF",
    }

    return known_leads.get(normalized)


def _convert_to_mv(samples, unit):
    if unit is None:
        raise ValueError(
            "Signal unit is required. Specify mV, uV, or V."
        )

    normalized = str(unit).strip().lower().replace(" ", "")

    if normalized not in UNIT_TO_MV:
        raise ValueError(
            f"Unsupported signal unit: {unit}. "
            "Supported units: mV, uV, V."
        )

    return (
        np.asarray(samples, dtype=np.float64)
        * UNIT_TO_MV[normalized]
    )


def from_samples(
    samples,
    sampling_rate_hz,
    unit,
    lead=None,
    source=None,
    input_format="CSV",
    extraction_method="direct",
    metadata=None,
):
    """
    Create the common ECG representation from extracted samples.

    PDF digitization can use this same function later.
    """

    samples = np.asarray(samples, dtype=np.float64)

    if samples.ndim != 1:
        raise ValueError("ECG samples must be a one-dimensional array.")

    if len(samples) < 2:
        raise ValueError("ECG recording must contain at least 2 samples.")

    if (sampling_rate_hz is None or not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0):
        raise ValueError("A valid sampling rate is required.")

    samples = _convert_to_mv(samples, unit)

    warnings = []

    missing = int(np.count_nonzero(~np.isfinite(samples)))

    if missing:
        warnings.append(
            f"{missing} missing or non-numeric samples detected."
        )

    duration = (len(samples) - 1) / float(sampling_rate_hz)

    return ECGSignal(
        samples=samples,
        sampling_rate_hz=float(sampling_rate_hz),
        original_sampling_rate_hz=float(sampling_rate_hz),
        lead=lead,
        unit="mV",
        source=source,
        duration_seconds=duration,
        input_format=input_format.upper(),
        extraction_method=extraction_method,
        warnings=warnings,
        metadata=metadata or {},
    )


def load_ecg_csv(
    file_path,
    sampling_rate_hz=None,
    unit=None,
    signal_column=None,
    timestamp_column=None,
    lead=None,
    source=None,
):
    """
    Supports:
      A. Single ECG column
      B. Timestamp + ECG column
      C. Timestamp + multiple lead columns

    Timestamp values must be expressed in seconds.
    If timestamps are supplied, their spacing determines
    the effective sampling rate.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CSV not found: {file_path}")

    if file_path.suffix.lower() != ".csv":
        raise ValueError("Expected a .csv file.")

    if unit is None:
        raise ValueError(
            "Please specify the ECG signal unit: mV, uV, or V."
        )

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}")

    if df.empty or len(df.columns) == 0:
        raise ValueError("CSV contains no usable data.")

    df.columns = [str(c).strip() for c in df.columns]

    # Identify the ECG signal column.
    if signal_column is not None:
        if signal_column not in df.columns:
            raise ValueError(
                f"Signal column '{signal_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        selected_column = signal_column

    else:
        candidates = []

        for column in df.columns:
            if column == timestamp_column:
                continue

            numeric = pd.to_numeric(
                df[column], errors="coerce"
            )

            if numeric.notna().sum() >= 2:
                candidates.append(column)

        if len(candidates) == 0:
            raise ValueError("No numeric ECG column was found.")

        if len(candidates) > 1:
            raise ValueError(
                "Multiple numeric columns found. "
                f"Select one using signal_column. "
                f"Candidates: {candidates}"
            )

        selected_column = candidates[0]

    samples = pd.to_numeric(
        df[selected_column], errors="coerce"
    ).to_numpy(dtype=np.float64)

    if len(samples) < 2:
        raise ValueError("Not enough ECG samples.")

    warnings = []

    missing = int(np.count_nonzero(~np.isfinite(samples)))

    if missing:
        warnings.append(
            f"{missing} missing or non-numeric samples detected. "
            "Samples were preserved; no imputation was performed."
        )

    # Determine sampling rate from timestamps when available.
    if timestamp_column is not None:
        if timestamp_column not in df.columns:
            raise ValueError(
                f"Timestamp column '{timestamp_column}' not found."
            )

        timestamps = pd.to_numeric(
            df[timestamp_column], errors="coerce"
        ).to_numpy(dtype=np.float64)

        if not np.all(np.isfinite(timestamps)):
            raise ValueError(
                "Timestamp column contains missing or invalid values."
            )

        differences = np.diff(timestamps)

        if np.any(differences <= 0):
            raise ValueError(
                "Timestamps must be strictly increasing."
            )

        median_interval = float(np.median(differences))

        if median_interval <= 0:
            raise ValueError("Invalid timestamp intervals.")

        inferred_rate = 1.0 / median_interval

        # Flag timestamp irregularity above 2%.
        relative_jitter = (
            np.median(np.abs(differences - median_interval))
            / median_interval
        )

        if relative_jitter > 0.02:
            warnings.append(
                "Irregular timestamp spacing detected. "
                "Review signal quality before model inference."
            )

        if sampling_rate_hz is not None:
            relative_difference = abs(
                sampling_rate_hz - inferred_rate
            ) / inferred_rate

            if relative_difference > 0.01:
                warnings.append(
                    "Provided sampling rate differs from the "
                    "timestamp-derived rate. Timestamp-derived "
                    "rate was used."
                )

        sampling_rate_hz = inferred_rate

    if (sampling_rate_hz is None or not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0):
        raise ValueError(
            "Provide sampling_rate_hz or a timestamp_column."
        )

    # Convert explicitly declared units to millivolts.
    samples = _convert_to_mv(samples, unit)

    duration = (len(samples) - 1) / float(sampling_rate_hz)

    return ECGSignal(
        samples=samples,
        sampling_rate_hz=float(sampling_rate_hz),
        original_sampling_rate_hz=float(sampling_rate_hz),
        lead=lead or _infer_lead(selected_column),
        unit="mV",
        source=source,
        duration_seconds=duration,
        input_format="CSV",
        extraction_method="direct",
        warnings=warnings,
        metadata={
            "file_name": file_path.name,
            "signal_column": selected_column,
            "timestamp_column": timestamp_column,
            "original_unit": unit,
            "csv_rows": int(len(df)),
            "csv_columns": list(df.columns),
        },
    )
