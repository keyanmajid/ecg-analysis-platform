
from pathlib import Path

import cv2
import numpy as np


def detect_waveform_regions(
    image_path,
    min_region_height=30,
    row_activity_threshold=0.015,
    merge_gap=20,
):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold handles varying page brightness.
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )

    # Ignore very small isolated noise.
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
    )

    # Calculate dark-pixel activity per row.
    row_activity = np.mean(binary > 0, axis=1)

    active_rows = np.where(
        row_activity >= row_activity_threshold
    )[0]

    regions = []

    if len(active_rows):
        start = previous = int(active_rows[0])

        for row in active_rows[1:]:
            row = int(row)

            if row - previous > merge_gap:
                if previous - start + 1 >= min_region_height:
                    regions.append((start, previous + 1))

                start = row

            previous = row

        if previous - start + 1 >= min_region_height:
            regions.append((start, previous + 1))

    results = []

    for index, (y1, y2) in enumerate(regions):
        crop = image[y1:y2, :]

        results.append({
            "region_id": index + 1,
            "x1": 0,
            "y1": y1,
            "x2": image.shape[1],
            "y2": y2,
            "width": image.shape[1],
            "height": y2 - y1,
            "mean_row_activity": float(
                np.mean(row_activity[y1:y2])
            ),
        })

    return {
        "image_path": str(image_path),
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "candidate_count": len(results),
        "regions": results,
        "warning": (
            "Candidate regions require visual or algorithmic "
            "verification. Text and grid lines may be detected."
        ),
    }
