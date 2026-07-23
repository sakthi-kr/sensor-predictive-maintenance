"""
Validate the sensor predictive-maintenance feature table and model outputs.

This script integrates the reusable ml-validation-toolkit with the existing
CWRU bearing-fault classification pipeline.

Validation scope:
- feature-table schema and data quality
- allowed class labels
- unique window identifiers
- numerical feature ranges
- class representation
- prediction lengths and labels
- probability matrix validity
- confidence-score validity
- metric regression thresholds
- confusion-matrix consistency

Important:
This validates the current window-level baseline pipeline. It does not remove
the known possibility of leakage between similar windows from the same source
signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from ml_validation_toolkit import (
    print_validation_summary,
    raise_for_validation_failures,
    run_data_checks,
    run_model_checks,
    save_validation_csv,
    save_validation_json,
)


try:
    # Works when running as: python -m src.validate_pipeline
    from src.data_loader import ALLOWED_LABELS, load_dataset
    from src.features import build_feature_table
    from src.train import RANDOM_STATE, get_feature_columns
except ModuleNotFoundError:
    # Works when running directly from PyCharm.
    from data_loader import ALLOWED_LABELS, load_dataset
    from features import build_feature_table
    from train import RANDOM_STATE, get_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "baseline_random_forest.joblib"
)

RESULTS_DIRECTORY = PROJECT_ROOT / "results"

VALIDATION_JSON_PATH = (
    RESULTS_DIRECTORY
    / "sensor_validation_report.json"
)

VALIDATION_CSV_PATH = (
    RESULTS_DIRECTORY
    / "sensor_validation_checks.csv"
)

VALIDATION_PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "sensor_validation_predictions.csv"
)

WINDOW_SIZE = 2048
STEP_SIZE = 1024
SAMPLE_RATE = 12000
MAX_WINDOWS_PER_FILE = 100
TEST_SIZE = 0.25


def load_model_bundle(
    model_path: Path = MODEL_PATH,
) -> dict[str, Any]:
    """
    Load and validate the saved Random Forest model bundle.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {model_path}\n"
            "Run `python src/train.py` first."
        )

    model_bundle = joblib.load(model_path)

    if not isinstance(model_bundle, dict):
        raise TypeError(
            "The saved model bundle must be a dictionary."
        )

    required_keys = {
        "model",
        "feature_columns",
    }

    missing_keys = required_keys - set(model_bundle)

    if missing_keys:
        raise KeyError(
            "Saved model bundle is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    model = model_bundle["model"]

    if not hasattr(model, "predict"):
        raise TypeError(
            "The saved model does not provide predict()."
        )

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The saved model does not provide predict_proba()."
        )

    if not hasattr(model, "classes_"):
        raise TypeError(
            "The saved model does not provide classes_."
        )

    return model_bundle


def build_current_feature_table() -> pd.DataFrame:
    """
    Rebuild the current feature table using the project configuration.
    """
    records = load_dataset()

    feature_table = build_feature_table(
        records=records,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        sample_rate=SAMPLE_RATE,
        max_windows_per_file=MAX_WINDOWS_PER_FILE,
    )

    if feature_table.empty:
        raise ValueError(
            "The generated feature table is empty."
        )

    return feature_table


def recreate_test_split(
    feature_table: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Recreate the same deterministic test split used in train.py.
    """
    X = feature_table[list(feature_columns)]
    y = feature_table["label"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_test, y_test


def validate_feature_column_compatibility(
    current_feature_columns: Sequence[str],
    saved_feature_columns: Sequence[str],
) -> None:
    """
    Ensure that current features match the model's saved feature schema.
    """
    current = list(current_feature_columns)
    saved = list(saved_feature_columns)

    if current != saved:
        missing_from_current = [
            column
            for column in saved
            if column not in current
        ]

        unexpected_current = [
            column
            for column in current
            if column not in saved
        ]

        raise ValueError(
            "Current feature columns do not match the saved model schema.\n"
            f"Missing from current table: {missing_from_current}\n"
            f"Unexpected current columns: {unexpected_current}\n"
            f"Current order: {current}\n"
            f"Saved order: {saved}"
        )


def build_prediction_table(
    *,
    y_true: pd.Series | Sequence[Any],
    y_pred: Sequence[Any],
    probabilities: np.ndarray,
    class_labels: Sequence[Any],
) -> pd.DataFrame:
    """
    Build a prediction-level table for inspection and validation.
    """
    true_array = np.asarray(y_true)
    prediction_array = np.asarray(y_pred)
    probability_array = np.asarray(
        probabilities,
        dtype=float,
    )

    if true_array.ndim != 1:
        raise ValueError(
            "y_true must be one-dimensional."
        )

    if prediction_array.ndim != 1:
        raise ValueError(
            "y_pred must be one-dimensional."
        )

    if probability_array.ndim != 2:
        raise ValueError(
            "probabilities must be a two-dimensional matrix."
        )

    sample_count = len(true_array)

    if len(prediction_array) != sample_count:
        raise ValueError(
            "y_true and y_pred lengths do not match."
        )

    if probability_array.shape[0] != sample_count:
        raise ValueError(
            "Probability rows do not match the number of samples."
        )

    if probability_array.shape[1] != len(class_labels):
        raise ValueError(
            "Probability columns do not match the class labels."
        )

    if isinstance(y_true, pd.Series):
        source_indices = y_true.index.to_numpy()
    else:
        source_indices = np.arange(sample_count)

    table = pd.DataFrame(
        {
            "source_row_index": source_indices,
            "true_label": true_array,
            "predicted_label": prediction_array,
            "prediction_confidence": probability_array.max(
                axis=1
            ),
            "correct": true_array == prediction_array,
        }
    )

    for class_index, class_label in enumerate(class_labels):
        probability_column = (
            f"probability_{str(class_label)}"
        )

        table[probability_column] = (
            probability_array[:, class_index]
        )

    return table


def create_numeric_range_configuration() -> dict[
    str,
    tuple[float | None, float | None],
]:
    """
    Define physically or mathematically meaningful feature ranges.
    """
    nyquist_frequency = SAMPLE_RATE / 2

    return {
        "std": (0.0, None),
        "rms": (0.0, None),
        "peak_to_peak": (0.0, None),
        "crest_factor": (0.0, None),
        "shape_factor": (0.0, None),
        "dominant_frequency": (
            0.0,
            nyquist_frequency,
        ),
        "spectral_centroid": (
            0.0,
            nyquist_frequency,
        ),
        "spectral_bandwidth": (
            0.0,
            nyquist_frequency,
        ),
        "spectral_energy": (0.0, None),
        "window_index": (
            0.0,
            float(MAX_WINDOWS_PER_FILE - 1),
        ),
    }


def main() -> None:
    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Sensor pipeline validation")
    print("==========================")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Model path   : {MODEL_PATH}")

    print("\nBuilding feature table...")

    feature_table = build_current_feature_table()
    feature_columns = get_feature_columns(
        feature_table
    )

    print(
        f"Feature table: "
        f"{feature_table.shape[0]} rows × "
        f"{feature_table.shape[1]} columns"
    )

    print("\nRunning feature-table validation...")

    required_columns = [
        "label",
        "file_name",
        "signal_key",
        "rpm",
        "window_index",
        *feature_columns,
    ]

    # RPM is intentionally excluded from strict missing-value validation
    # because some CWRU files may not contain an RPM field.
    strict_non_missing_columns = [
        "label",
        "file_name",
        "signal_key",
        "window_index",
        *feature_columns,
    ]

    data_results = run_data_checks(
        feature_table,
        required_columns=required_columns,
        missing_value_columns=strict_non_missing_columns,
        max_missing_fraction=0.0,
        infinite_value_columns=feature_columns,
        duplicate_subset=[
            "file_name",
            "window_index",
        ],
        max_duplicates=0,
        allowed_values={
            "label": sorted(ALLOWED_LABELS),
        },
        numeric_ranges=(
            create_numeric_range_configuration()
        ),
        target_column="label",
        min_classes=4,
        min_samples_per_class=20,
        min_fraction_per_class=0.20,
    )

    print("\nLoading saved model...")

    model_bundle = load_model_bundle()
    model = model_bundle["model"]

    saved_feature_columns = model_bundle[
        "feature_columns"
    ]

    validate_feature_column_compatibility(
        current_feature_columns=feature_columns,
        saved_feature_columns=saved_feature_columns,
    )

    print("Recreating evaluation split...")

    X_test, y_test = recreate_test_split(
        feature_table=feature_table,
        feature_columns=feature_columns,
    )

    print("Running model predictions...")

    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    class_labels = [
        str(label)
        for label in model.classes_
    ]

    prediction_confidence = probabilities.max(
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=class_labels,
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }

    print("\nRunning model-output validation...")

    model_results = run_model_checks(
        y_true=y_test,
        y_pred=y_pred,
        allowed_labels=sorted(ALLOWED_LABELS),
        probabilities=probabilities,
        expected_probability_classes=len(
            class_labels
        ),
        scores=prediction_confidence,
        score_minimum=0.0,
        score_maximum=1.0,
        metrics=metrics,
        metric_minimums={
            # These are pipeline-regression thresholds,
            # not production acceptance thresholds.
            "accuracy": 0.90,
            "macro_f1": 0.90,
            "weighted_f1": 0.90,
        },
        matrix=matrix,
        confusion_matrix_labels=class_labels,
    )

    all_results = [
        *data_results,
        *model_results,
    ]

    prediction_table = build_prediction_table(
        y_true=y_test,
        y_pred=y_pred,
        probabilities=probabilities,
        class_labels=class_labels,
    )

    prediction_table.to_csv(
        VALIDATION_PREDICTIONS_PATH,
        index=False,
    )

    save_validation_json(
        all_results,
        VALIDATION_JSON_PATH,
        report_name=(
            "Sensor Predictive Maintenance "
            "Validation Report"
        ),
        metadata={
            "project": "sensor-predictive-maintenance",
            "dataset": (
                "Case Western Reserve University "
                "bearing vibration data"
            ),
            "model_type": type(model).__name__,
            "class_labels": class_labels,
            "feature_columns": feature_columns,
            "feature_table_rows": int(
                len(feature_table)
            ),
            "test_samples": int(len(y_test)),
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "sample_rate_hz": SAMPLE_RATE,
            "max_windows_per_file": (
                MAX_WINDOWS_PER_FILE
            ),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "metrics": metrics,
            "known_validation_limitation": (
                "The current baseline uses a random "
                "window-level split. Similar windows from "
                "the same original signal can occur in both "
                "training and test sets, so performance may "
                "be overestimated."
            ),
            "metric_threshold_interpretation": (
                "Metric thresholds are regression checks for "
                "the current development pipeline, not "
                "production-readiness criteria."
            ),
        },
    )

    save_validation_csv(
        all_results,
        VALIDATION_CSV_PATH,
    )

    print()
    print_validation_summary(
        all_results,
        report_name=(
            "Sensor Predictive Maintenance "
            "Validation"
        ),
    )

    print("\nModel metrics")
    print("-------------")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nSaved outputs")
    print("-------------")
    print(
        f"JSON report : {VALIDATION_JSON_PATH}"
    )
    print(
        f"CSV checks  : {VALIDATION_CSV_PATH}"
    )
    print(
        f"Predictions : "
        f"{VALIDATION_PREDICTIONS_PATH}"
    )

    # Reports are saved before this call so failed checks
    # can still be investigated.
    raise_for_validation_failures(
        all_results,
        message_prefix=(
            "Sensor predictive-maintenance validation failed"
        ),
    )


if __name__ == "__main__":
    main()
