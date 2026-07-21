


import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


try:
    # Works when running as: python -m src.evaluate
    from src.data_loader import load_dataset
    from src.features import build_feature_table
    from src.train import RANDOM_STATE, get_feature_columns
except ModuleNotFoundError:
    # Works when running directly from PyCharm: Run evaluate.py
    from data_loader import load_dataset
    from features import build_feature_table
    from train import RANDOM_STATE, get_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "baseline_random_forest.joblib"


def load_model_bundle(model_path: Path = MODEL_PATH) -> dict:
    """
    Load the saved model bundle from train.py.
    """
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


def prepare_test_data(
    test_size: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Rebuild the feature table and recreate the same test split used in training.
    """
    records = load_dataset()

    feature_table = build_feature_table(
        records=records,
        window_size=2048,
        step_size=1024,
        sample_rate=12000,
        max_windows_per_file=100,
    )

    feature_columns = get_feature_columns(feature_table)

    X = feature_table[feature_columns]
    y = feature_table["label"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_test, y_test, feature_columns


def save_confusion_matrix_plot(
    y_test: pd.Series,
    y_pred,
    output_path: Path,
) -> None:
    """
    Save confusion matrix plot.
    """
    labels = sorted(y_test.unique())

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    display.plot(ax=ax, values_format="d")
    ax.set_title("Confusion Matrix - Bearing Fault Classification")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_feature_importance(
    model,
    feature_columns: list[str],
    output_csv_path: Path,
    output_plot_path: Path,
) -> None:
    """
    Save feature importance table and plot for tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        print("Model does not provide feature_importances_. Skipping feature importance.")
        return

    importance_table = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_table.to_csv(output_csv_path, index=False)

    top_features = importance_table.head(15).sort_values("importance")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_features["feature"], top_features["importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances")
    plt.tight_layout()

    fig.savefig(output_plot_path, dpi=300)
    plt.close(fig)


def save_evaluation_summary(
    y_test: pd.Series,
    y_pred,
    output_path: Path,
) -> None:
    """
    Save classification report as JSON and text.
    """
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

    summary = {
        "model_type": "RandomForestClassifier",
        "n_test_samples": len(y_test),
        "classification_report": report_dict,
        "validation_note": (
            "This evaluation uses the same window-level random split as the first "
            "baseline training run. It is useful for checking the pipeline, but it "
            "may overestimate performance because windows from the same original "
            "signal can be similar. A future version should evaluate across files, "
            "fault sizes, or operating/load conditions."
        ),
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    text_output_path = output_path.with_suffix(".txt")
    with open(text_output_path, "w", encoding="utf-8") as file:
        file.write(report_text)

    print("\nClassification report:")
    print(report_text)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading saved model...")
    model_bundle = load_model_bundle()

    model = model_bundle["model"]
    saved_feature_columns = model_bundle["feature_columns"]

    print("Preparing test data...")
    X_test, y_test, feature_columns = prepare_test_data()

    if feature_columns != saved_feature_columns:
        raise ValueError(
            "Feature columns from the current feature table do not match "
            "the feature columns saved with the model.\n"
            f"Current columns: {feature_columns}\n"
            f"Saved columns: {saved_feature_columns}"
        )

    print("Running predictions...")
    y_pred = model.predict(X_test)

    confusion_matrix_path = RESULTS_DIR / "confusion_matrix.png"
    evaluation_summary_path = RESULTS_DIR / "evaluation_summary.json"
    feature_importance_csv_path = RESULTS_DIR / "feature_importance.csv"
    feature_importance_plot_path = RESULTS_DIR / "feature_importance.png"

    save_confusion_matrix_plot(
        y_test=y_test,
        y_pred=y_pred,
        output_path=confusion_matrix_path,
    )

    save_evaluation_summary(
        y_test=y_test,
        y_pred=y_pred,
        output_path=evaluation_summary_path,
    )

    save_feature_importance(
        model=model,
        feature_columns=feature_columns,
        output_csv_path=feature_importance_csv_path,
        output_plot_path=feature_importance_plot_path,
    )

    print("\nSaved evaluation outputs:")
    print(f"  Confusion matrix       : {confusion_matrix_path}")
    print(f"  Evaluation summary JSON: {evaluation_summary_path}")
    print(f"  Evaluation summary TXT : {evaluation_summary_path.with_suffix('.txt')}")
    print(f"  Feature importance CSV : {feature_importance_csv_path}")
    print(f"  Feature importance plot: {feature_importance_plot_path}")


if __name__ == "__main__":
    main()