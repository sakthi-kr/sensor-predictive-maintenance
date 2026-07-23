from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.data_loader import (  # noqa: E402
    TARGET_SAMPLE_RATE_HZ,
    load_dataset_from_manifest,
)
from src.dataset_manifest import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
)
from src.features import (  # noqa: E402
    MODEL_FEATURE_COLUMNS,
    build_feature_table,
)


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

FEATURE_TABLE_PATH = (
    RESULTS_DIR
    / "cwru_load_features.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "cwru_load_feature_summary.json"
)

WINDOW_SIZE = 2048
STEP_SIZE = 1024
MAX_WINDOWS_PER_FILE = 50
WINDOW_SELECTION = "even"

EXPECTED_FILES = 16

EXPECTED_WINDOWS_PER_FILE = (
    MAX_WINDOWS_PER_FILE
)

EXPECTED_ROWS = (
    EXPECTED_FILES
    * EXPECTED_WINDOWS_PER_FILE
)


def validate_feature_table(
    feature_table: pd.DataFrame,
) -> dict:
    missing_feature_columns = (
        set(MODEL_FEATURE_COLUMNS)
        - set(feature_table.columns)
    )

    if missing_feature_columns:
        raise ValueError(
            "Feature table is missing "
            "model features: "
            f"{sorted(missing_feature_columns)}"
        )

    required_metadata = {
        "label",
        "source_recording",
        "load_hp",
        "fault_type",
        "fault_diameter_in",
        "original_sample_rate_hz",
        "sample_rate_hz",
        "window_index",
        "window_start_sample",
        "window_end_sample",
    }

    missing_metadata = (
        required_metadata
        - set(feature_table.columns)
    )

    if missing_metadata:
        raise ValueError(
            "Feature table is missing "
            "metadata columns: "
            f"{sorted(missing_metadata)}"
        )

    if len(feature_table) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"but received "
            f"{len(feature_table)}."
        )

    file_counts = (
        feature_table[
            "source_recording"
        ]
        .value_counts()
        .sort_index()
    )

    if file_counts.size != EXPECTED_FILES:
        raise ValueError(
            f"Expected {EXPECTED_FILES} "
            "source recordings, but "
            f"received {file_counts.size}."
        )

    if not (
        file_counts
        == EXPECTED_WINDOWS_PER_FILE
    ).all():
        raise ValueError(
            "Every source recording must "
            "contribute exactly "
            f"{EXPECTED_WINDOWS_PER_FILE} "
            "windows.\n"
            f"Observed counts:\n"
            f"{file_counts}"
        )

    class_counts = (
        feature_table["label"]
        .value_counts()
        .sort_index()
    )

    load_counts = (
        feature_table["load_hp"]
        .value_counts()
        .sort_index()
    )

    expected_per_class = (
        EXPECTED_ROWS // 4
    )

    expected_per_load = (
        EXPECTED_ROWS // 4
    )

    if not (
        class_counts
        == expected_per_class
    ).all():
        raise ValueError(
            "Feature table is not "
            "balanced across classes.\n"
            f"Observed counts:\n"
            f"{class_counts}"
        )

    if not (
        load_counts
        == expected_per_load
    ).all():
        raise ValueError(
            "Feature table is not "
            "balanced across loads.\n"
            f"Observed counts:\n"
            f"{load_counts}"
        )

    effective_rates = set(
        feature_table[
            "sample_rate_hz"
        ].unique()
    )

    if effective_rates != {
        TARGET_SAMPLE_RATE_HZ
    }:
        raise ValueError(
            "All signals must use the "
            "common effective sample rate "
            f"{TARGET_SAMPLE_RATE_HZ} Hz, "
            "but observed "
            f"{sorted(effective_rates)}."
        )

    numeric_values = (
        feature_table[
            MODEL_FEATURE_COLUMNS
        ]
        .to_numpy(dtype=float)
    )

    if not np.isfinite(
        numeric_values
    ).all():
        raise ValueError(
            "The model feature columns "
            "contain NaN or infinite values."
        )

    group_class_counts = (
        feature_table[
            [
                "source_recording",
                "label",
            ]
        ]
        .drop_duplicates()
        .groupby(
            "source_recording"
        )["label"]
        .nunique()
    )

    if not (
        group_class_counts == 1
    ).all():
        raise ValueError(
            "A source recording maps to "
            "more than one class label."
        )

    original_rate_counts = (
        feature_table[
            [
                "source_recording",
                "original_sample_rate_hz",
            ]
        ]
        .drop_duplicates()
        [
            "original_sample_rate_hz"
        ]
        .value_counts()
        .sort_index()
    )

    return {
        "feature_table_path": (
            FEATURE_TABLE_PATH
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
        "manifest_path": (
            DEFAULT_MANIFEST_PATH
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
        "n_rows": int(
            len(feature_table)
        ),
        "n_columns": int(
            feature_table.shape[1]
        ),
        "n_model_features": len(
            MODEL_FEATURE_COLUMNS
        ),
        "n_source_recordings": int(
            feature_table[
                "source_recording"
            ].nunique()
        ),
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "max_windows_per_file": (
            MAX_WINDOWS_PER_FILE
        ),
        "window_selection": (
            WINDOW_SELECTION
        ),
        "effective_sample_rate_hz": (
            TARGET_SAMPLE_RATE_HZ
        ),
        "class_counts": {
            str(key): int(value)
            for key, value
            in class_counts.items()
        },
        "load_counts": {
            str(key): int(value)
            for key, value
            in load_counts.items()
        },
        "windows_per_file": {
            str(key): int(value)
            for key, value
            in file_counts.items()
        },
        "original_sample_rate_counts": {
            str(key): int(value)
            for key, value
            in original_rate_counts.items()
        },
        "validation_status": "passed",
    }


def main() -> None:
    print(
        "Loading the 16-file "
        "CWRU benchmark..."
    )

    records = (
        load_dataset_from_manifest(
            manifest_path=(
                DEFAULT_MANIFEST_PATH
            ),
            target_sample_rate_hz=(
                TARGET_SAMPLE_RATE_HZ
            ),
        )
    )

    print(
        f"Loaded recordings: "
        f"{len(records)}"
    )

    print()
    print(
        "Sample-rate standardization:"
    )

    for record in records:
        print(
            f"  {record.file_path.name:>7}: "
            f"{record.original_sample_rate_hz:>5} "
            "-> "
            f"{record.sample_rate_hz:>5} Hz | "
            f"samples={record.signal.size}"
        )

    print()
    print(
        "Extracting metadata-rich "
        "window features..."
    )

    feature_table = (
        build_feature_table(
            records=records,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE,
            sample_rate=None,
            max_windows_per_file=(
                MAX_WINDOWS_PER_FILE
            ),
            window_selection=(
                WINDOW_SELECTION
            ),
        )
    )

    summary = validate_feature_table(
        feature_table
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_table.to_csv(
        FEATURE_TABLE_PATH,
        index=False,
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("=" * 72)
    print(
        "CWRU LOAD FEATURE TABLE"
    )
    print("=" * 72)

    print(
        f"Rows                 : "
        f"{summary['n_rows']}"
    )

    print(
        f"Columns              : "
        f"{summary['n_columns']}"
    )

    print(
        f"Model features       : "
        f"{summary['n_model_features']}"
    )

    print(
        f"Source recordings    : "
        f"{summary['n_source_recordings']}"
    )

    print(
        "Effective sample rate : "
        f"{summary['effective_sample_rate_hz']} "
        "Hz"
    )

    print()
    print("Class counts:")

    for label, count in (
        summary[
            "class_counts"
        ].items()
    ):
        print(
            f"  {label:<18}: {count}"
        )

    print()
    print("Load counts:")

    for load_hp, count in (
        summary[
            "load_counts"
        ].items()
    ):
        print(
            f"  {load_hp} HP: {count}"
        )

    print()
    print(
        f"Saved feature table  : "
        f"{FEATURE_TABLE_PATH}"
    )

    print(
        f"Saved summary        : "
        f"{SUMMARY_PATH}"
    )

    print()
    print(
        "PASS: The feature table is "
        "balanced, metadata-rich, finite, "
        "and standardized to 12 kHz."
    )


if __name__ == "__main__":
    main()
