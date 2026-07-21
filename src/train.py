

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


try:
    # Works when running as: python -m src.train
    from src.data_loader import load_dataset
    from src.features import build_feature_table
except ModuleNotFoundError:
    # Works when running directly from PyCharm: Run train.py
    from data_loader import load_dataset
    from features import build_feature_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42


def get_feature_columns(feature_table: pd.DataFrame) -> list[str]:
    """
    Select only numerical ML feature columns.

    We exclude label and metadata columns because they are not input features.
    """
    excluded_columns = {
        "label",
        "file_name",
        "signal_key",
        "rpm",
        "window_index",
    }

    feature_columns = [
        column
        for column in feature_table.columns
        if column not in excluded_columns
    ]

    return feature_columns


def train_random_forest(
    feature_table: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Train a Random Forest baseline model.

    Returns a dictionary containing:
    - trained model
    - feature columns
    - train/test data
    - predictions
    - metrics
    """
    feature_columns = get_feature_columns(feature_table)

    X = feature_table[feature_columns]
    y = feature_table["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )

    return {
        "model": model,
        "feature_columns": feature_columns,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
    }


def save_training_outputs(training_output: dict) -> None:
    """
    Save model and evaluation outputs.

    The model is saved locally under models/.
    The results are saved under results/.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "baseline_random_forest.joblib"
    metrics_path = RESULTS_DIR / "metrics.json"
    report_path = RESULTS_DIR / "classification_report.txt"
    feature_columns_path = RESULTS_DIR / "feature_columns.json"

    model_bundle = {
        "model": training_output["model"],
        "feature_columns": training_output["feature_columns"],
        "random_state": RANDOM_STATE,
        "model_type": "RandomForestClassifier",
    }

    joblib.dump(model_bundle, model_path)

    metrics = {
        "model_type": "RandomForestClassifier",
        "accuracy": training_output["accuracy"],
        "n_train_samples": len(training_output["X_train"]),
        "n_test_samples": len(training_output["X_test"]),
        "feature_columns": training_output["feature_columns"],
        "classification_report": training_output["classification_report_dict"],
        "validation_note": (
            "This first baseline uses window-level random train/test splitting. "
            "This is useful for pipeline development but may overestimate performance "
            "because windows from the same original signal can be similar. "
            "A later version should use a more leakage-aware split."
        ),
    }

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(training_output["classification_report_text"])

    with open(feature_columns_path, "w", encoding="utf-8") as file:
        json.dump(training_output["feature_columns"], file, indent=4)

    print("\nSaved outputs:")
    print(f"  Model              : {model_path}")
    print(f"  Metrics            : {metrics_path}")
    print(f"  Classification report: {report_path}")
    print(f"  Feature columns    : {feature_columns_path}")


def main() -> None:
    print("Loading dataset...")
    records = load_dataset()

    print("Building feature table...")
    feature_table = build_feature_table(
        records=records,
        window_size=2048,
        step_size=1024,
        sample_rate=12000,
        max_windows_per_file=100,
    )

    print(f"Feature table shape: {feature_table.shape}")
    print("\nClass counts:")
    print(feature_table["label"].value_counts())

    print("\nTraining Random Forest baseline...")
    training_output = train_random_forest(feature_table)

    print("\nTraining complete.")
    print(f"Accuracy: {training_output['accuracy']:.4f}")

    print("\nClassification report:")
    print(training_output["classification_report_text"])

    save_training_outputs(training_output)


if __name__ == "__main__":
    main()