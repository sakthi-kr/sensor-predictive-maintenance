from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import (
    BearingSignal,
    standardize_sample_rate,
)
from src.dataset_manifest import (
    load_manifest,
)
from src.features import (
    MODEL_FEATURE_COLUMNS,
    build_feature_table,
    segment_signal_with_starts,
)


def test_load_manifest_resolves_paths(
    tmp_path: Path,
) -> None:
    manifest_path = (
        tmp_path
        / "manifest.csv"
    )

    manifest_path.write_text(
        "file_name,label,fault_type,"
        "fault_diameter_in,"
        "outer_race_position,load_hp,"
        "approx_rpm,sample_rate_hz,"
        "dataset_section,local_path,"
        "download_url\n"
        "97.mat,normal,normal,0.0,"
        "not_applicable,0,1797,48000,"
        "normal_baseline,"
        "data/raw/cwru/normal/97.mat,"
        "https://example.com/97.mat\n",
        encoding="utf-8",
    )

    records = load_manifest(
        manifest_path
    )

    assert len(records) == 1

    record = records[0]

    assert record.file_name == "97.mat"
    assert record.label == "normal"

    assert (
        record.sample_rate_hz
        == 48000
    )

    assert (
        record.local_path.name
        == "97.mat"
    )

    assert record.local_path.is_absolute()


def test_standardize_sample_rate(
) -> None:
    original_rate = 48000
    target_rate = 12000

    time = (
        np.arange(original_rate)
        / original_rate
    )

    signal = np.sin(
        2.0
        * np.pi
        * 1000.0
        * time
    )

    standardized = (
        standardize_sample_rate(
            signal,
            original_sample_rate_hz=(
                original_rate
            ),
            target_sample_rate_hz=(
                target_rate
            ),
        )
    )

    assert standardized.ndim == 1

    assert (
        standardized.size
        == target_rate
    )

    assert np.isfinite(
        standardized
    ).all()


def test_even_window_selection(
) -> None:
    signal = np.arange(
        20000,
        dtype=float,
    )

    windows, starts = (
        segment_signal_with_starts(
            signal,
            window_size=1000,
            step_size=500,
            max_windows=5,
            selection="even",
        )
    )

    assert windows.shape == (
        5,
        1000,
    )

    assert starts.size == 5
    assert starts[0] == 0
    assert starts[-1] == 19000

    assert np.all(
        np.diff(starts) > 0
    )


def test_feature_table_has_metadata(
) -> None:
    sample_rate = 12000
    sample_count = 60000

    time = (
        np.arange(sample_count)
        / sample_rate
    )

    signal = np.sin(
        2.0
        * np.pi
        * 200.0
        * time
    )

    record = BearingSignal(
        file_path=Path(
            "normal/97.mat"
        ),
        label="normal",
        signal=signal,
        signal_key="X097_DE_time",
        rpm=1797.0,
        source_recording="97.mat",
        fault_type="normal",
        fault_diameter_in=0.0,
        outer_race_position=(
            "not_applicable"
        ),
        load_hp=0,
        approx_rpm=1797,
        original_sample_rate_hz=48000,
        sample_rate_hz=12000,
        dataset_section=(
            "normal_baseline"
        ),
    )

    table = build_feature_table(
        records=[record],
        window_size=2048,
        step_size=1024,
        max_windows_per_file=10,
        window_selection="even",
    )

    assert isinstance(
        table,
        pd.DataFrame,
    )

    assert len(table) == 10

    assert set(
        MODEL_FEATURE_COLUMNS
    ).issubset(table.columns)

    assert (
        table[
            "source_recording"
        ].nunique()
        == 1
    )

    assert (
        table[
            "source_recording"
        ].iloc[0]
        == "97.mat"
    )

    assert (
        table["load_hp"].iloc[0]
        == 0
    )

    assert (
        table[
            "original_sample_rate_hz"
        ].iloc[0]
        == 48000
    )

    assert (
        table[
            "sample_rate_hz"
        ].iloc[0]
        == 12000
    )

    assert table[
        "window_start_sample"
    ].is_monotonic_increasing
