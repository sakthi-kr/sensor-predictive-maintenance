import numpy as np
import pandas as pd

from src.data_loader import load_dataset
from src.features import (
    build_feature_table,
    extract_features_from_window,
    segment_signal,
)


def test_segment_signal_shape():
    signal = np.arange(10000)

    windows = segment_signal(
        signal=signal,
        window_size=1000,
        step_size=500,
        max_windows=5,
    )

    assert windows.shape == (5, 1000)


def test_extract_features_from_window():
    window = np.sin(np.linspace(0, 10, 2048))

    features = extract_features_from_window(
        window=window,
        sample_rate=12000,
    )

    expected_features = {
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
    }

    assert expected_features.issubset(features.keys())

    for key, value in features.items():
        assert np.isfinite(value), f"Feature {key} is not finite: {value}"


def test_build_feature_table():
    records = load_dataset()

    feature_table = build_feature_table(
        records=records,
        window_size=2048,
        step_size=1024,
        sample_rate=12000,
        max_windows_per_file=5,
    )

    assert isinstance(feature_table, pd.DataFrame)
    assert not feature_table.empty

    expected_columns = {
        "mean",
        "std",
        "rms",
        "peak_to_peak",
        "skewness",
        "kurtosis",
        "dominant_frequency",
        "spectral_energy",
        "label",
        "file_name",
        "signal_key",
        "rpm",
        "window_index",
    }

    assert expected_columns.issubset(feature_table.columns)
    assert feature_table["label"].nunique() >= 4

    numeric_columns = feature_table.select_dtypes(include="number").columns
    assert np.isfinite(feature_table[numeric_columns].to_numpy()).all()
