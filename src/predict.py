import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat


try:
    # Works when running as: python -m src.predict
    from src.data_loader import ALLOWED_LABELS, extract_rpm, select_vibration_key
    from src.features import extract_features_from_window, segment_signal
except ModuleNotFoundError:
    # Works when running directly from PyCharm: Run predict.py
    from data_loader import ALLOWED_LABELS, extract_rpm, select_vibration_key
    from features import extract_features_from_window, segment_signal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATH = MODELS_DIR / "baseline_random_forest.joblib"

DEFAULT_FILE = PROJECT_ROOT / "data" / "raw" / "cwru" / "normal" / "97.mat"


def load_model_bundle(model_path: Path = MODEL_PATH) -> dict:
    """
    Load the saved model bundle created by train.py.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Run `python src/train.py` first."
        )

    model_bundle = joblib.load(model_path)

    required_keys = {"model", "feature_columns"}
    missing_keys = required_keys - set(model_bundle.keys())

    if missing_keys:
        raise KeyError(
            f"Model bundle is missing keys: {missing_keys}\n"
            f"Available keys: {list(model_bundle.keys())}"
        )

    return model_bundle


def load_signal_from_mat_file(
    file_path: Path,
    preferred_sensor: str = "DE",
) -> tuple[np.ndarray, str, Optional[float]]:
    """
    Load vibration signal from one CWRU .mat file.

    Returns:
    - signal
    - selected signal key
    - RPM value, if available
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {file_path}")

    mat_data = loadmat(file_path)

    signal_key = select_vibration_key(
        mat_data=mat_data,
        preferred_sensor=preferred_sensor,
    )

    rpm = extract_rpm(mat_data)

    signal = np.asarray(mat_data[signal_key]).squeeze().astype(float)

    if signal.ndim != 1:
        raise ValueError(
            f"Expected 1D vibration signal, got shape {signal.shape} "
            f"from key {signal_key}"
        )

    if signal.size == 0:
        raise ValueError(f"Empty signal found in file: {file_path}")

    return signal, signal_key, rpm


def infer_true_label_from_path(file_path: Path) -> Optional[str]:
    """
    Infer true label from parent folder name if the file is inside one of our
    known class folders.

    Example:
    data/raw/cwru/ball_fault/118.mat -> ball_fault
    """
    parent_name = Path(file_path).parent.name

    if parent_name in ALLOWED_LABELS:
        return parent_name

    return None


def build_features_for_prediction(
    signal: np.ndarray,
    feature_columns: list[str],
    window_size: int = 2048,
    step_size: int = 1024,
    sample_rate: int = 12000,
    max_windows: Optional[int] = 100,
) -> pd.DataFrame:
    """
    Convert one vibration signal into a feature table for prediction.
    """
    windows = segment_signal(
        signal=signal,
        window_size=window_size,
        step_size=step_size,
        max_windows=max_windows,
    )

    rows = []

    for window in windows:
        row = extract_features_from_window(
            window=window,
            sample_rate=sample_rate,
        )
        rows.append(row)

    feature_table = pd.DataFrame(rows)

    missing_columns = set(feature_columns) - set(feature_table.columns)
    extra_columns = set(feature_table.columns) - set(feature_columns)

    if missing_columns:
        raise ValueError(
            f"Prediction feature table is missing columns: {sorted(missing_columns)}"
        )

    if extra_columns:
        # Not an error. We simply keep the columns the model was trained on.
        feature_table = feature_table[feature_columns]

    return feature_table


def aggregate_predictions(
    window_predictions: np.ndarray,
    prediction_probabilities: Optional[np.ndarray],
    class_names: np.ndarray,
) -> dict:
    """
    Combine window-level predictions into one file-level prediction.

    We report both:
    - majority-vote prediction
    - mean-probability prediction, if probabilities are available
    """
    prediction_counts = Counter(window_predictions)

    majority_label, majority_count = prediction_counts.most_common(1)[0]
    majority_fraction = majority_count / len(window_predictions)

    result = {
        "final_prediction": majority_label,
        "aggregation_method": "majority_vote",
        "majority_fraction": majority_fraction,
        "window_prediction_counts": dict(prediction_counts),
    }

    if prediction_probabilities is not None:
        mean_probabilities = prediction_probabilities.mean(axis=0)

        probability_by_class = {
            class_name: float(probability)
            for class_name, probability in zip(class_names, mean_probabilities)
        }

        probability_label = max(
            probability_by_class,
            key=probability_by_class.get,
        )

        result["final_prediction"] = probability_label
        result["aggregation_method"] = "mean_predicted_probability"
        result["mean_probability_by_class"] = probability_by_class
        result["confidence"] = probability_by_class[probability_label]

    return result


def predict_file(
    file_path: Path,
    model_path: Path = MODEL_PATH,
    preferred_sensor: str = "DE",
    window_size: int = 2048,
    step_size: int = 1024,
    sample_rate: int = 12000,
    max_windows: Optional[int] = 100,
) -> dict:
    """
    Predict bearing condition for a single .mat file.
    """
    file_path = Path(file_path)

    model_bundle = load_model_bundle(model_path)

    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    signal, signal_key, rpm = load_signal_from_mat_file(
        file_path=file_path,
        preferred_sensor=preferred_sensor,
    )

    prediction_features = build_features_for_prediction(
        signal=signal,
        feature_columns=feature_columns,
        window_size=window_size,
        step_size=step_size,
        sample_rate=sample_rate,
        max_windows=max_windows,
    )

    window_predictions = model.predict(prediction_features)

    prediction_probabilities = None
    class_names = getattr(model, "classes_", np.array([]))

    if hasattr(model, "predict_proba"):
        prediction_probabilities = model.predict_proba(prediction_features)

    aggregated = aggregate_predictions(
        window_predictions=window_predictions,
        prediction_probabilities=prediction_probabilities,
        class_names=class_names,
    )

    true_label = infer_true_label_from_path(file_path)

    result = {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "true_label_from_folder": true_label,
        "signal_key": signal_key,
        "rpm": rpm,
        "signal_length": int(signal.size),
        "n_windows_used": int(len(prediction_features)),
        "window_size": window_size,
        "step_size": step_size,
        "sample_rate": sample_rate,
        **aggregated,
    }

    return result


def save_prediction_result(result: dict) -> Path:
    """
    Save prediction output as JSON.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    file_stem = Path(result["file_name"]).stem
    output_path = RESULTS_DIR / f"prediction_{file_stem}.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    return output_path


def print_prediction_result(result: dict) -> None:
    """
    Print prediction result clearly.
    """
    print("\nPrediction result")
    print("-----------------")
    print(f"File              : {result['file_name']}")
    print(f"Signal key        : {result['signal_key']}")
    print(f"True label        : {result['true_label_from_folder']}")
    print(f"Predicted label   : {result['final_prediction']}")
    print(f"Aggregation method: {result['aggregation_method']}")
    print(f"Windows used      : {result['n_windows_used']}")

    if "confidence" in result:
        print(f"Confidence        : {result['confidence']:.4f}")

    print("\nWindow prediction counts:")
    for label, count in result["window_prediction_counts"].items():
        print(f"  {label}: {count}")

    if "mean_probability_by_class" in result:
        print("\nMean probability by class:")
        for label, probability in result["mean_probability_by_class"].items():
            print(f"  {label}: {probability:.4f}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict bearing condition from one CWRU .mat vibration file."
    )

    parser.add_argument(
        "--file",
        type=str,
        default=str(DEFAULT_FILE),
        help="Path to the .mat file to classify.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_PATH),
        help="Path to the trained .joblib model bundle.",
    )

    parser.add_argument(
        "--sensor",
        type=str,
        default="DE",
        choices=["DE", "FE"],
        help="Preferred vibration sensor channel: DE or FE.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    result = predict_file(
        file_path=Path(args.file),
        model_path=Path(args.model),
        preferred_sensor=args.sensor,
    )

    print_prediction_result(result)

    output_path = save_prediction_result(result)
    print(f"\nSaved prediction result to: {output_path}")


if __name__ == "__main__":
    main()