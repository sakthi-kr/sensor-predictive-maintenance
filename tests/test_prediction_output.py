from pathlib import Path

import pytest

from src.data_loader import ALLOWED_LABELS
from src.predict import MODEL_PATH, predict_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_FILE = PROJECT_ROOT / "data" / "raw" / "cwru" / "normal" / "97.mat"


def test_prediction_output_structure():
    if not MODEL_PATH.exists():
        pytest.skip("Model file does not exist. Run `python src/train.py` first.")

    if not DEFAULT_TEST_FILE.exists():
        pytest.skip(f"Test file does not exist: {DEFAULT_TEST_FILE}")

    result = predict_file(DEFAULT_TEST_FILE)

    required_keys = {
        "file_path",
        "file_name",
        "true_label_from_folder",
        "signal_key",
        "signal_length",
        "n_windows_used",
        "final_prediction",
        "aggregation_method",
        "window_prediction_counts",
    }

    assert required_keys.issubset(result.keys())
    assert result["final_prediction"] in ALLOWED_LABELS
    assert result["true_label_from_folder"] == "normal"
    assert result["n_windows_used"] > 0
    assert result["signal_length"] > 0


def test_prediction_counts_are_valid():
    if not MODEL_PATH.exists():
        pytest.skip("Model file does not exist. Run `python src/train.py` first.")

    if not DEFAULT_TEST_FILE.exists():
        pytest.skip(f"Test file does not exist: {DEFAULT_TEST_FILE}")

    result = predict_file(DEFAULT_TEST_FILE)

    total_window_predictions = sum(result["window_prediction_counts"].values())

    assert total_window_predictions == result["n_windows_used"]

    for label in result["window_prediction_counts"].keys():
        assert label in ALLOWED_LABELS
