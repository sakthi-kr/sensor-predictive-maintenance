from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    ALLOWED_LABELS,
    BearingSignal,
)
from src.features import (
    build_feature_table,
    extract_features_from_window,
    segment_signal,
)


@pytest.fixture
def synthetic_records() -> list[BearingSignal]:
    """
    Create four in-memory vibration signals representing the four labels.
    """
    sample_count = 8192

    time = np.linspace(
        0.0,
        1.0,
        sample_count,
        endpoint=False,
    )

    records = []

    for index, label in enumerate(
        sorted(ALLOWED_LABELS),
        start=1,
    ):
        frequency = 10.0 * index

        signal = (
            np.sin(
                2.0
                * np.pi
                * frequency
                * time
            )
            + 0.05
            * np.sin(
                2.0
                * np.pi
                * frequency
                * 3.0
                * time
            )
        )

        records.append(
            BearingSignal(
                file_path=Path(
                    f"{label}_{index}.mat"
                ),
                label=label,
                signal=signal.astype(float),
                signal_key=(
                    f"X{index:03d}_DE_time"
                ),
                rpm=1772.0,
            )
        )

    return records


def test_segment_signal_shape() -> None:
    signal = np.arange(10000)

    windows = segment_signal(
        signal=signal,
        window_size=1000,
        step_size=500,
        max_windows=5,
    )

    assert windows.shape == (5, 1000)


def test_segment_signal_rejects_short_signal() -> None:
    signal = np.arange(100)

    with pytest.raises(
        ValueError,
        match="smaller than window_size",
    ):
        segment_signal(
            signal=signal,
            window_size=200,
            step_size=100,
        )


def test_extract_features_from_window() -> None:
    window = np.sin(
        np.linspace(
            0.0,
            10.0,
            2048,
        )
    )

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

    assert expected_features.issubset(
        features.keys()
    )

    for key, value in features.items():
        assert np.isfinite(
            value
        ), f"Feature {key} is not finite: {value}"


def test_build_feature_table(
    synthetic_records: list[BearingSignal],
) -> None:
    feature_table = build_feature_table(
        records=synthetic_records,
        window_size=2048,
        step_size=1024,
        sample_rate=12000,
        max_windows_per_file=5,
    )

    assert isinstance(
        feature_table,
        pd.DataFrame,
    )

    assert feature_table.shape[0] == 20

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

    assert expected_columns.issubset(
        feature_table.columns
    )

    assert (
        feature_table["label"].nunique()
        == 4
    )

    assert set(
        feature_table["label"].unique()
    ) == ALLOWED_LABELS

    numeric_columns = (
        feature_table
        .select_dtypes(include="number")
        .columns
    )

    assert np.isfinite(
        feature_table[
            numeric_columns
        ].to_numpy()
    ).all()
