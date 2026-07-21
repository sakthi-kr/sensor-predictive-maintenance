

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


try:
    # Works when running as: python -m src.features
    from src.data_loader import BearingSignal, load_dataset
except ModuleNotFoundError:
    # Works when running directly from PyCharm: Run features.py
    from data_loader import BearingSignal, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def segment_signal(
    signal: np.ndarray,
    window_size: int = 2048,
    step_size: int = 1024,
    max_windows: Optional[int] = None,
) -> np.ndarray:
    """
    Split a 1D vibration signal into overlapping windows.

    Parameters
    ----------
    signal:
        1D vibration signal.
    window_size:
        Number of samples per window.
    step_size:
        Number of samples to move between windows.
        If step_size < window_size, windows overlap.
    max_windows:
        Optional limit on the number of windows returned.

    Returns
    -------
    windows:
        2D array with shape (number_of_windows, window_size).
    """
    signal = np.asarray(signal).squeeze()

    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}")

    if window_size <= 0:
        raise ValueError("window_size must be positive")

    if step_size <= 0:
        raise ValueError("step_size must be positive")

    if signal.size < window_size:
        raise ValueError(
            f"Signal length {signal.size} is smaller than window_size {window_size}"
        )

    windows = []

    for start in range(0, signal.size - window_size + 1, step_size):
        end = start + window_size
        windows.append(signal[start:end])

        if max_windows is not None and len(windows) >= max_windows:
            break

    return np.asarray(windows)


def extract_time_domain_features(window: np.ndarray) -> dict:
    """
    Extract time-domain statistical features from one vibration window.
    """
    window = np.asarray(window).squeeze()

    mean_value = np.mean(window)
    std_value = np.std(window)
    rms_value = np.sqrt(np.mean(window ** 2))
    min_value = np.min(window)
    max_value = np.max(window)
    peak_to_peak_value = np.ptp(window)

    # Avoid division by zero
    abs_mean = np.mean(np.abs(window))
    crest_factor = np.max(np.abs(window)) / rms_value if rms_value != 0 else 0.0
    shape_factor = rms_value / abs_mean if abs_mean != 0 else 0.0

    return {
        "mean": mean_value,
        "std": std_value,
        "rms": rms_value,
        "min": min_value,
        "max": max_value,
        "peak_to_peak": peak_to_peak_value,
        "skewness": skew(window),
        "kurtosis": kurtosis(window),
        "crest_factor": crest_factor,
        "shape_factor": shape_factor,
    }


def extract_frequency_domain_features(
    window: np.ndarray,
    sample_rate: int = 12000,
) -> dict:
    """
    Extract simple frequency-domain features from one vibration window.

    For the first CWRU version, we use 12 kHz because the selected fault files
    are from the 12k drive-end bearing fault dataset.
    """
    window = np.asarray(window).squeeze()

    # Remove DC offset before FFT
    centered_window = window - np.mean(window)

    # Real FFT because the signal is real-valued
    fft_values = np.fft.rfft(centered_window)
    frequencies = np.fft.rfftfreq(window.size, d=1.0 / sample_rate)
    magnitude = np.abs(fft_values)

    # Ignore the zero-frequency bin for dominant-frequency calculation
    if magnitude.size > 1:
        nonzero_magnitude = magnitude[1:]
        nonzero_frequencies = frequencies[1:]
        dominant_frequency = nonzero_frequencies[np.argmax(nonzero_magnitude)]
    else:
        dominant_frequency = 0.0

    magnitude_sum = np.sum(magnitude)

    if magnitude_sum == 0:
        spectral_centroid = 0.0
        spectral_bandwidth = 0.0
    else:
        spectral_centroid = np.sum(frequencies * magnitude) / magnitude_sum
        spectral_bandwidth = np.sqrt(
            np.sum(((frequencies - spectral_centroid) ** 2) * magnitude)
            / magnitude_sum
        )

    spectral_energy = np.sum(magnitude ** 2) / len(magnitude)

    return {
        "dominant_frequency": dominant_frequency,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_energy": spectral_energy,
    }


def extract_features_from_window(
    window: np.ndarray,
    sample_rate: int = 12000,
) -> dict:
    """
    Extract all features from one signal window.
    """
    features = {}
    features.update(extract_time_domain_features(window))
    features.update(extract_frequency_domain_features(window, sample_rate=sample_rate))
    return features


def build_feature_table(
    records: list[BearingSignal],
    window_size: int = 2048,
    step_size: int = 1024,
    sample_rate: int = 12000,
    max_windows_per_file: Optional[int] = 100,
) -> pd.DataFrame:
    """
    Convert loaded bearing signals into a machine-learning feature table.

    Each row = one window from one vibration file.
    Columns = extracted features + label + file metadata.
    """
    rows = []

    for record in records:
        windows = segment_signal(
            record.signal,
            window_size=window_size,
            step_size=step_size,
            max_windows=max_windows_per_file,
        )

        for window_index, window in enumerate(windows):
            feature_row = extract_features_from_window(
                window,
                sample_rate=sample_rate,
            )

            feature_row["label"] = record.label
            feature_row["file_name"] = record.file_path.name
            feature_row["signal_key"] = record.signal_key
            feature_row["rpm"] = record.rpm
            feature_row["window_index"] = window_index

            rows.append(feature_row)

    feature_table = pd.DataFrame(rows)

    if feature_table.empty:
        raise ValueError("Feature table is empty. Check the input records.")

    return feature_table


def save_feature_preview(feature_table: pd.DataFrame) -> Path:
    """
    Save a small preview of the feature table.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "feature_preview.csv"
    feature_table.head(20).to_csv(output_path, index=False)

    return output_path


def print_feature_summary(feature_table: pd.DataFrame) -> None:
    """
    Print basic information about the feature table.
    """
    print("\nFeature table created successfully.")
    print(f"Shape: {feature_table.shape}")

    print("\nClass counts:")
    print(feature_table["label"].value_counts())

    print("\nColumns:")
    for column in feature_table.columns:
        print(f"  - {column}")

    print("\nPreview:")
    print(feature_table.head())


if __name__ == "__main__":
    dataset = load_dataset()

    features = build_feature_table(
        records=dataset,
        window_size=2048,
        step_size=1024,
        sample_rate=12000,
        max_windows_per_file=100,
    )

    print_feature_summary(features)

    preview_path = save_feature_preview(features)
    print(f"\nSaved feature preview to: {preview_path}")