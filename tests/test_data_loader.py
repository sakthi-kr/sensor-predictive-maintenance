from pathlib import Path

import numpy as np
import pytest

from src.data_loader import (
    ALLOWED_LABELS,
    DEFAULT_DATA_ROOT,
    find_mat_files,
    label_from_path,
    load_bearing_signal,
    load_dataset,
)


def test_data_root_exists():
    assert DEFAULT_DATA_ROOT.exists(), f"Data root does not exist: {DEFAULT_DATA_ROOT}"


def test_find_mat_files():
    mat_files = find_mat_files(DEFAULT_DATA_ROOT)

    assert len(mat_files) >= 4
    assert all(file_path.suffix == ".mat" for file_path in mat_files)


def test_label_from_path():
    test_cases = {
        "normal": "97.mat",
        "inner_race_fault": "105.mat",
        "ball_fault": "118.mat",
        "outer_race_fault": "130.mat",
    }

    for label, filename in test_cases.items():
        file_path = DEFAULT_DATA_ROOT / label / filename

        if not file_path.exists():
            pytest.skip(f"Missing expected test file: {file_path}")

        inferred_label = label_from_path(file_path, data_root=DEFAULT_DATA_ROOT)
        assert inferred_label == label


def test_load_single_bearing_signal():
    mat_files = find_mat_files(DEFAULT_DATA_ROOT)
    record = load_bearing_signal(mat_files[0], data_root=DEFAULT_DATA_ROOT)

    assert record.label in ALLOWED_LABELS
    assert record.file_path.exists()
    assert isinstance(record.signal, np.ndarray)
    assert record.signal.ndim == 1
    assert record.signal.size > 0
    assert record.signal_key.endswith(("DE_time", "FE_time"))


def test_load_dataset():
    records = load_dataset(DEFAULT_DATA_ROOT)

    labels = {record.label for record in records}

    assert len(records) >= 4
    assert labels.issubset(ALLOWED_LABELS)
    assert "normal" in labels
    assert "inner_race_fault" in labels
    assert "ball_fault" in labels
    assert "outer_race_fault" in labels
