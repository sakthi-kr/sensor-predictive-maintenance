from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

try:
    from src.data_loader import (
        BearingSignal,
        load_dataset_from_manifest,
    )
except ModuleNotFoundError:
    from data_loader import (
        BearingSignal,
        load_dataset_from_manifest,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

MODEL_FEATURE_COLUMNS = [
    "mean",
    "std",
    "rms",
    "min",
    "max",
    "peak_to_peak",
    "skewness",
    "kurtosis",
    "crest_factor",
    "shape_factor",
    "dominant_frequency",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_energy",
]

METADATA_COLUMNS = [
    "label",
    "source_recording",
    "file_name",
    "signal_key",
    "rpm",
    "fault_type",
    "fault_diameter_in",
    "outer_race_position",
    "load_hp",
    "approx_rpm",
    "original_sample_rate_hz",
    "sample_rate_hz",
    "dataset_section",
    "window_index",
    "window_start_sample",
    "window_end_sample",
    "window_start_seconds",
    "window_end_seconds",
]


def segment_signal_with_starts(
    signal: np.ndarray,
    window_size: int = 2048,
    step_size: int = 1024,
    max_windows: Optional[int] = None,
    selection: str = "first",
) -> tuple[np.ndarray, np.ndarray]:
    signal = (
        np.asarray(signal)
        .squeeze()
    )

    if signal.ndim != 1:
        raise ValueError(
            "Expected 1D signal, "
            f"got shape {signal.shape}"
        )

    if window_size <= 0:
        raise ValueError(
            "window_size must be positive"
        )

    if step_size <= 0:
        raise ValueError(
            "step_size must be positive"
        )

    if signal.size < window_size:
        raise ValueError(
            f"Signal length {signal.size} "
            "is smaller than window_size "
            f"{window_size}"
        )

    if selection not in {
        "first",
        "even",
    }:
        raise ValueError(
            "selection must be either "
            "'first' or 'even'"
        )

    starts = np.arange(
        0,
        signal.size - window_size + 1,
        step_size,
        dtype=int,
    )

    if max_windows is not None:
        if max_windows <= 0:
            raise ValueError(
                "max_windows must be positive"
            )

        if starts.size > max_windows:
            if selection == "first":
                starts = starts[
                    :max_windows
                ]

            else:
                selected_indices = np.linspace(
                    0,
                    starts.size - 1,
                    num=max_windows,
                    dtype=int,
                )

                starts = starts[
                    selected_indices
                ]

    windows = np.stack(
        [
            signal[
                start:
                start + window_size
            ]
            for start in starts
        ]
    )

    return windows, starts


def segment_signal(
    signal: np.ndarray,
    window_size: int = 2048,
    step_size: int = 1024,
    max_windows: Optional[int] = None,
) -> np.ndarray:
    windows, _ = (
        segment_signal_with_starts(
            signal=signal,
            window_size=window_size,
            step_size=step_size,
            max_windows=max_windows,
            selection="first",
        )
    )

    return windows


def extract_time_domain_features(
    window: np.ndarray,
) -> dict:
    window = (
        np.asarray(window)
        .squeeze()
    )

    mean_value = np.mean(window)
    std_value = np.std(window)

    rms_value = np.sqrt(
        np.mean(window ** 2)
    )

    min_value = np.min(window)
    max_value = np.max(window)

    peak_to_peak_value = np.ptp(
        window
    )

    abs_mean = np.mean(
        np.abs(window)
    )

    crest_factor = (
        np.max(np.abs(window))
        / rms_value
        if rms_value != 0
        else 0.0
    )

    shape_factor = (
        rms_value / abs_mean
        if abs_mean != 0
        else 0.0
    )

    return {
        "mean": mean_value,
        "std": std_value,
        "rms": rms_value,
        "min": min_value,
        "max": max_value,
        "peak_to_peak": (
            peak_to_peak_value
        ),
        "skewness": skew(window),
        "kurtosis": kurtosis(window),
        "crest_factor": crest_factor,
        "shape_factor": shape_factor,
    }


def extract_frequency_domain_features(
    window: np.ndarray,
    sample_rate: int = 12000,
) -> dict:
    window = (
        np.asarray(window)
        .squeeze()
    )

    if sample_rate <= 0:
        raise ValueError(
            "sample_rate must be positive"
        )

    centered_window = (
        window - np.mean(window)
    )

    fft_values = np.fft.rfft(
        centered_window
    )

    frequencies = np.fft.rfftfreq(
        window.size,
        d=1.0 / sample_rate,
    )

    magnitude = np.abs(fft_values)

    if magnitude.size > 1:
        nonzero_magnitude = magnitude[1:]
        nonzero_frequencies = (
            frequencies[1:]
        )

        dominant_frequency = (
            nonzero_frequencies[
                np.argmax(
                    nonzero_magnitude
                )
            ]
        )

    else:
        dominant_frequency = 0.0

    magnitude_sum = np.sum(magnitude)

    if magnitude_sum == 0:
        spectral_centroid = 0.0
        spectral_bandwidth = 0.0

    else:
        spectral_centroid = (
            np.sum(
                frequencies * magnitude
            )
            / magnitude_sum
        )

        spectral_bandwidth = np.sqrt(
            np.sum(
                (
                    (
                        frequencies
                        - spectral_centroid
                    )
                    ** 2
                )
                * magnitude
            )
            / magnitude_sum
        )

    spectral_energy = (
        np.sum(magnitude ** 2)
        / len(magnitude)
    )

    return {
        "dominant_frequency": (
            dominant_frequency
        ),
        "spectral_centroid": (
            spectral_centroid
        ),
        "spectral_bandwidth": (
            spectral_bandwidth
        ),
        "spectral_energy": (
            spectral_energy
        ),
    }


def extract_features_from_window(
    window: np.ndarray,
    sample_rate: int = 12000,
) -> dict:
    features = {}

    features.update(
        extract_time_domain_features(
            window
        )
    )

    features.update(
        extract_frequency_domain_features(
            window,
            sample_rate=sample_rate,
        )
    )

    return features


def build_feature_table(
    records: list[BearingSignal],
    window_size: int = 2048,
    step_size: int = 1024,
    sample_rate: Optional[int] = None,
    max_windows_per_file: Optional[
        int
    ] = 100,
    window_selection: str = "first",
) -> pd.DataFrame:
    rows = []

    for record in records:
        effective_sample_rate = (
            record.sample_rate_hz
            if sample_rate is None
            else sample_rate
        )

        windows, starts = (
            segment_signal_with_starts(
                record.signal,
                window_size=window_size,
                step_size=step_size,
                max_windows=(
                    max_windows_per_file
                ),
                selection=window_selection,
            )
        )

        for window_index, (
            window,
            start,
        ) in enumerate(
            zip(
                windows,
                starts,
                strict=True,
            )
        ):
            feature_row = (
                extract_features_from_window(
                    window,
                    sample_rate=(
                        effective_sample_rate
                    ),
                )
            )

            end = int(
                start + window_size
            )

            feature_row.update(
                {
                    "label": record.label,
                    "source_recording": (
                        record.source_recording
                        or record.file_path.name
                    ),
                    "file_name": (
                        record.file_path.name
                    ),
                    "signal_key": (
                        record.signal_key
                    ),
                    "rpm": record.rpm,
                    "fault_type": (
                        record.fault_type
                    ),
                    "fault_diameter_in": (
                        record.fault_diameter_in
                    ),
                    "outer_race_position": (
                        record.outer_race_position
                    ),
                    "load_hp": (
                        record.load_hp
                    ),
                    "approx_rpm": (
                        record.approx_rpm
                    ),
                    "original_sample_rate_hz": (
                        record
                        .original_sample_rate_hz
                    ),
                    "sample_rate_hz": (
                        effective_sample_rate
                    ),
                    "dataset_section": (
                        record.dataset_section
                    ),
                    "window_index": (
                        window_index
                    ),
                    "window_start_sample": (
                        int(start)
                    ),
                    "window_end_sample": end,
                    "window_start_seconds": (
                        float(start)
                        / effective_sample_rate
                    ),
                    "window_end_seconds": (
                        float(end)
                        / effective_sample_rate
                    ),
                }
            )

            rows.append(feature_row)

    feature_table = pd.DataFrame(
        rows
    )

    if feature_table.empty:
        raise ValueError(
            "Feature table is empty. "
            "Check the input records."
        )

    ordered_columns = (
        MODEL_FEATURE_COLUMNS
        + [
            column
            for column in METADATA_COLUMNS
            if column
            in feature_table.columns
        ]
    )

    return feature_table[
        ordered_columns
    ]


def save_feature_preview(
    feature_table: pd.DataFrame,
) -> Path:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "feature_preview.csv"
    )

    feature_table.head(20).to_csv(
        output_path,
        index=False,
    )

    return output_path


def print_feature_summary(
    feature_table: pd.DataFrame,
) -> None:
    print(
        "\nFeature table created "
        "successfully."
    )

    print(
        f"Shape: {feature_table.shape}"
    )

    print("\nClass counts:")
    print(
        feature_table[
            "label"
        ].value_counts()
    )

    print("\nColumns:")

    for column in feature_table.columns:
        print(f"  - {column}")

    print("\nPreview:")
    print(feature_table.head())


if __name__ == "__main__":
    dataset = (
        load_dataset_from_manifest()
    )

    features = build_feature_table(
        records=dataset,
        window_size=2048,
        step_size=1024,
        sample_rate=None,
        max_windows_per_file=50,
        window_selection="even",
    )

    print_feature_summary(features)

    preview_path = (
        save_feature_preview(features)
    )

    print(
        "\nSaved feature preview to: "
        f"{preview_path}"
    )
