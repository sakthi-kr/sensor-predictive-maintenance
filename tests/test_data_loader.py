from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from src.data_loader import (
    ALLOWED_LABELS,
    DEFAULT_DATA_ROOT,
    find_mat_files,
    label_from_path,
    load_bearing_signal,
    load_dataset,
)


@pytest.fixture
def synthetic_data_root(
    tmp_path: Path,
) -> Path:
    """
    Create a temporary CWRU-style folder containing four small .mat files.

    This makes the tests independent of the real external dataset.
    """
    data_root = tmp_path / "cwru"

    file_configuration = {
        "normal": (
            "97.mat",
            "X097_DE_time",
            "X097RPM",
            1.0,
        ),
        "inner_race_fault": (
            "105.mat",
            "X105_DE_time",
            "X105RPM",
            2.0,
        ),
        "ball_fault": (
            "118.mat",
            "X118_DE_time",
            "X118RPM",
            3.0,
        ),
        "outer_race_fault": (
            "130.mat",
            "X130_DE_time",
            "X130RPM",
            4.0,
        ),
    }

    time = np.linspace(
        0.0,
        2.0 * np.pi,
        4096,
        endpoint=False,
    )

    for label, configuration in file_configuration.items():
        (
            file_name,
            signal_key,
            rpm_key,
            frequency,
        ) = configuration

        label_directory = data_root / label
        label_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        signal = np.sin(
            frequency * time
        ).reshape(-1, 1)

        savemat(
            label_directory / file_name,
            {
                signal_key: signal,
                rpm_key: np.array([[1772.0]]),
            },
        )

    return data_root


def test_default_data_root_is_path() -> None:
    """
    The configured local data path should be represented by a Path.

    Its existence is not required in CI because the real dataset is external.
    """
    assert isinstance(DEFAULT_DATA_ROOT, Path)


def test_find_mat_files(
    synthetic_data_root: Path,
) -> None:
    mat_files = find_mat_files(
        synthetic_data_root
    )

    assert len(mat_files) == 4
    assert all(
        file_path.suffix == ".mat"
        for file_path in mat_files
    )


def test_label_from_path(
    synthetic_data_root: Path,
) -> None:
    expected_files = {
        "normal": "97.mat",
        "inner_race_fault": "105.mat",
        "ball_fault": "118.mat",
        "outer_race_fault": "130.mat",
    }

    for label, file_name in expected_files.items():
        file_path = (
            synthetic_data_root
            / label
            / file_name
        )

        inferred_label = label_from_path(
            file_path,
            data_root=synthetic_data_root,
        )

        assert inferred_label == label


def test_load_single_bearing_signal(
    synthetic_data_root: Path,
) -> None:
    file_path = (
        synthetic_data_root
        / "normal"
        / "97.mat"
    )

    record = load_bearing_signal(
        file_path,
        data_root=synthetic_data_root,
    )

    assert record.label == "normal"
    assert record.file_path == file_path
    assert isinstance(record.signal, np.ndarray)
    assert record.signal.ndim == 1
    assert record.signal.size == 4096
    assert record.signal_key == "X097_DE_time"
    assert record.rpm == pytest.approx(1772.0)


def test_load_dataset(
    synthetic_data_root: Path,
) -> None:
    records = load_dataset(
        synthetic_data_root
    )

    labels = {
        record.label
        for record in records
    }

    assert len(records) == 4
    assert labels == ALLOWED_LABELS

    for record in records:
        assert record.signal.ndim == 1
        assert record.signal.size == 4096
        assert np.isfinite(record.signal).all()


def test_find_mat_files_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing"

    with pytest.raises(
        FileNotFoundError,
        match="Data root does not exist",
    ):
        find_mat_files(missing_path)
