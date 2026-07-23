from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


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

from src.leakage_aware_evaluation import (  # noqa: E402
    CLASS_LABELS,
    N_ESTIMATORS,
    RANDOM_STATE,
    STRATEGY_DISPLAY_NAMES,
    run_all_evaluations,
    summarize_strategies,
)


FEATURE_TABLE_PATH = (
    PROJECT_ROOT
    / "results"
    / "cwru_load_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "leakage_aware_evaluation"
)


def relative_path(
    path: Path,
) -> str:
    return (
        path.relative_to(
            PROJECT_ROOT
        )
        .as_posix()
    )


def save_confusion_matrix(
    predictions: pd.DataFrame,
    strategy: str,
) -> Path:
    strategy_predictions = (
        predictions[
            predictions[
                "strategy"
            ]
            == strategy
        ]
    )

    matrix = confusion_matrix(
        strategy_predictions[
            "y_true"
        ],
        strategy_predictions[
            "y_pred"
        ],
        labels=CLASS_LABELS,
    )

    display = (
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=(
                CLASS_LABELS
            ),
        )
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    display.plot(
        ax=axis,
        values_format="d",
        colorbar=False,
    )

    axis.set_title(
        f"{STRATEGY_DISPLAY_NAMES[strategy]} "
        "— window predictions"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_DIR
        / (
            "confusion_matrix_"
            f"{strategy}.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


def save_strategy_comparison_plot(
    strategy_summary: pd.DataFrame,
) -> Path:
    positions = np.arange(
        len(strategy_summary)
    )

    width = 0.34

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.bar(
        positions - width / 2,
        strategy_summary[
            "window_accuracy_mean"
        ],
        width,
        label="Window accuracy",
    )

    axis.bar(
        positions + width / 2,
        strategy_summary[
            "window_macro_f1_mean"
        ],
        width,
        label="Window macro F1",
    )

    axis.set_ylabel(
        "Mean score"
    )

    axis.set_ylim(
        0.0,
        1.05,
    )

    axis.set_title(
        "Evaluation strategy comparison"
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        strategy_summary[
            "strategy_name"
        ],
        rotation=15,
        ha="right",
    )

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "strategy_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


def save_leave_one_load_out_plot(
    fold_metrics: pd.DataFrame,
) -> Path:
    load_metrics = (
        fold_metrics[
            fold_metrics[
                "strategy"
            ]
            == "leave_one_load_out"
        ]
        .sort_values(
            "held_out_load_hp"
        )
        .copy()
    )

    positions = np.arange(
        len(load_metrics)
    )

    width = 0.34

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.bar(
        positions - width / 2,
        load_metrics[
            "window_accuracy"
        ],
        width,
        label="Window accuracy",
    )

    axis.bar(
        positions + width / 2,
        load_metrics[
            "window_macro_f1"
        ],
        width,
        label="Window macro F1",
    )

    axis.set_ylabel(
        "Score"
    )

    axis.set_ylim(
        0.0,
        1.05,
    )

    axis.set_xlabel(
        "Held-out motor load"
    )

    axis.set_title(
        "Leave-one-load-out "
        "generalization"
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        [
            f"{int(load)} HP"
            for load
            in load_metrics[
                "held_out_load_hp"
            ]
        ]
    )

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_DIR
        / (
            "leave_one_load_out_"
            "by_load.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


def save_classification_reports(
    predictions: pd.DataFrame,
) -> dict[str, str]:
    report_paths = {}

    for strategy in (
        STRATEGY_DISPLAY_NAMES
    ):
        strategy_predictions = (
            predictions[
                predictions[
                    "strategy"
                ]
                == strategy
            ]
        )

        report_text = (
            classification_report(
                strategy_predictions[
                    "y_true"
                ],
                strategy_predictions[
                    "y_pred"
                ],
                labels=CLASS_LABELS,
                target_names=(
                    CLASS_LABELS
                ),
                zero_division=0,
            )
        )

        output_path = (
            OUTPUT_DIR
            / (
                "classification_report_"
                f"{strategy}.txt"
            )
        )

        output_path.write_text(
            report_text,
            encoding="utf-8",
        )

        report_paths[
            strategy
        ] = relative_path(
            output_path
        )

    return report_paths


def strategy_record(
    summary_table: pd.DataFrame,
    strategy: str,
) -> dict:
    row = (
        summary_table[
            summary_table[
                "strategy"
            ]
            == strategy
        ]
        .iloc[0]
    )

    return {
        "strategy_name": (
            row[
                "strategy_name"
            ]
        ),
        "folds": int(
            row["folds"]
        ),
        "window_accuracy_mean": float(
            row[
                "window_accuracy_mean"
            ]
        ),
        "window_accuracy_std": float(
            row[
                "window_accuracy_std"
            ]
        ),
        "window_balanced_accuracy_mean": float(
            row[
                "window_balanced_accuracy_mean"
            ]
        ),
        "window_macro_f1_mean": float(
            row[
                "window_macro_f1_mean"
            ]
        ),
        "window_macro_f1_std": float(
            row[
                "window_macro_f1_std"
            ]
        ),
        "recording_accuracy_mean": float(
            row[
                "recording_accuracy_mean"
            ]
        ),
        "recording_accuracy_std": float(
            row[
                "recording_accuracy_std"
            ]
        ),
        "recording_macro_f1_mean": float(
            row[
                "recording_macro_f1_mean"
            ]
        ),
        "maximum_recording_overlap": int(
            row[
                "maximum_recording_overlap"
            ]
        ),
    }


def build_summary(
    feature_table: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    output_files: dict,
) -> dict:
    random_result = (
        strategy_record(
            strategy_summary,
            "random_window",
        )
    )

    grouped_result = (
        strategy_record(
            strategy_summary,
            "grouped_recording",
        )
    )

    load_result = (
        strategy_record(
            strategy_summary,
            "leave_one_load_out",
        )
    )

    return {
        "dataset": {
            "feature_table": (
                relative_path(
                    FEATURE_TABLE_PATH
                )
            ),
            "n_windows": int(
                len(feature_table)
            ),
            "n_source_recordings": int(
                feature_table[
                    "source_recording"
                ].nunique()
            ),
            "n_classes": int(
                feature_table[
                    "label"
                ].nunique()
            ),
            "loads_hp": [
                int(value)
                for value in sorted(
                    feature_table[
                        "load_hp"
                    ].unique()
                )
            ],
            "windows_per_recording": int(
                feature_table.groupby(
                    "source_recording"
                )
                .size()
                .iloc[0]
            ),
        },
        "model": {
            "type": (
                "RandomForestClassifier"
            ),
            "n_estimators": (
                N_ESTIMATORS
            ),
            "class_weight": (
                "balanced"
            ),
            "random_state_base": (
                RANDOM_STATE
            ),
        },
        "strategies": {
            "random_window": (
                random_result
            ),
            "grouped_recording": (
                grouped_result
            ),
            "leave_one_load_out": (
                load_result
            ),
        },
        "generalization_gaps": {
            (
                "random_minus_grouped_"
                "window_accuracy"
            ): float(
                random_result[
                    "window_accuracy_mean"
                ]
                - grouped_result[
                    "window_accuracy_mean"
                ]
            ),
            (
                "random_minus_leave_one_"
                "load_out_window_accuracy"
            ): float(
                random_result[
                    "window_accuracy_mean"
                ]
                - load_result[
                    "window_accuracy_mean"
                ]
            ),
            (
                "random_minus_grouped_"
                "macro_f1"
            ): float(
                random_result[
                    "window_macro_f1_mean"
                ]
                - grouped_result[
                    "window_macro_f1_mean"
                ]
            ),
            (
                "random_minus_leave_one_"
                "load_out_macro_f1"
            ): float(
                random_result[
                    "window_macro_f1_mean"
                ]
                - load_result[
                    "window_macro_f1_mean"
                ]
            ),
        },
        "leakage_checks": {
            (
                "random_window_maximum_"
                "overlapping_recordings"
            ): int(
                fold_metrics.loc[
                    (
                        fold_metrics[
                            "strategy"
                        ]
                        == "random_window"
                    ),
                    (
                        "overlapping_"
                        "recording_count"
                    ),
                ].max()
            ),
            (
                "grouped_recording_maximum_"
                "overlapping_recordings"
            ): int(
                fold_metrics.loc[
                    (
                        fold_metrics[
                            "strategy"
                        ]
                        == "grouped_recording"
                    ),
                    (
                        "overlapping_"
                        "recording_count"
                    ),
                ].max()
            ),
            (
                "leave_one_load_out_maximum_"
                "overlapping_recordings"
            ): int(
                fold_metrics.loc[
                    (
                        fold_metrics[
                            "strategy"
                        ]
                        == "leave_one_load_out"
                    ),
                    (
                        "overlapping_"
                        "recording_count"
                    ),
                ].max()
            ),
        },
        "interpretation": {
            "random_window": (
                "Development baseline. "
                "Windows from the same source "
                "recording can occur in both "
                "training and test sets."
            ),
            "grouped_recording": (
                "Leakage-aware recording-level "
                "evaluation. A source recording "
                "never appears in both training "
                "and test sets within a fold."
            ),
            "leave_one_load_out": (
                "Operating-condition evaluation. "
                "The model is trained on three "
                "motor loads and tested on one "
                "unseen load."
            ),
            "important_dataset_limitation": (
                "Normal recordings were originally "
                "sampled at 48 kHz and resampled "
                "to 12 kHz, while the selected "
                "fault recordings were collected "
                "at 12 kHz. Results may still "
                "reflect acquisition-domain "
                "differences in addition to "
                "bearing condition."
            ),
        },
        "output_files": (
            output_files
        ),
    }


def main() -> None:
    if not FEATURE_TABLE_PATH.exists():
        raise FileNotFoundError(
            "Feature table not found: "
            f"{FEATURE_TABLE_PATH}\n"
            "Run "
            "`python scripts/"
            "build_cwru_load_features.py` "
            "first."
        )

    print(
        "Loading feature table: "
        f"{FEATURE_TABLE_PATH}"
    )

    feature_table = pd.read_csv(
        FEATURE_TABLE_PATH
    )

    print(
        f"Rows={len(feature_table)}, "
        "recordings="
        f"{feature_table['source_recording'].nunique()}, "
        "classes="
        f"{feature_table['label'].nunique()}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        window_predictions,
        recording_predictions,
        fold_metrics,
    ) = run_all_evaluations(
        feature_table
    )

    strategy_summary = (
        summarize_strategies(
            fold_metrics
        )
    )

    fold_metrics_path = (
        OUTPUT_DIR
        / "fold_metrics.csv"
    )

    strategy_summary_path = (
        OUTPUT_DIR
        / "strategy_comparison.csv"
    )

    window_predictions_path = (
        OUTPUT_DIR
        / "window_predictions.csv"
    )

    recording_predictions_path = (
        OUTPUT_DIR
        / "recording_predictions.csv"
    )

    fold_metrics.to_csv(
        fold_metrics_path,
        index=False,
    )

    strategy_summary.to_csv(
        strategy_summary_path,
        index=False,
    )

    window_predictions.to_csv(
        window_predictions_path,
        index=False,
    )

    recording_predictions.to_csv(
        recording_predictions_path,
        index=False,
    )

    report_paths = (
        save_classification_reports(
            window_predictions
        )
    )

    confusion_paths = {
        strategy: relative_path(
            save_confusion_matrix(
                window_predictions,
                strategy,
            )
        )
        for strategy
        in STRATEGY_DISPLAY_NAMES
    }

    comparison_plot_path = (
        save_strategy_comparison_plot(
            strategy_summary
        )
    )

    load_plot_path = (
        save_leave_one_load_out_plot(
            fold_metrics
        )
    )

    output_files = {
        "fold_metrics_csv": (
            relative_path(
                fold_metrics_path
            )
        ),
        "strategy_comparison_csv": (
            relative_path(
                strategy_summary_path
            )
        ),
        "window_predictions_csv": (
            relative_path(
                window_predictions_path
            )
        ),
        "recording_predictions_csv": (
            relative_path(
                recording_predictions_path
            )
        ),
        "strategy_comparison_plot": (
            relative_path(
                comparison_plot_path
            )
        ),
        "leave_one_load_out_plot": (
            relative_path(
                load_plot_path
            )
        ),
        "classification_reports": (
            report_paths
        ),
        "confusion_matrices": (
            confusion_paths
        ),
    }

    summary = build_summary(
        feature_table=(
            feature_table
        ),
        fold_metrics=(
            fold_metrics
        ),
        strategy_summary=(
            strategy_summary
        ),
        output_files=(
            output_files
        ),
    )

    summary_path = (
        OUTPUT_DIR
        / "summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("=" * 92)
    print(
        "LEAKAGE-AWARE "
        "EVALUATION SUMMARY"
    )
    print("=" * 92)

    display_columns = [
        "strategy_name",
        "folds",
        "window_accuracy_mean",
        "window_accuracy_std",
        "window_macro_f1_mean",
        "window_macro_f1_std",
        "recording_accuracy_mean",
        "maximum_recording_overlap",
    ]

    print(
        strategy_summary[
            display_columns
        ].to_string(
            index=False,
            float_format=(
                lambda value:
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(
        "Generalization gaps:"
    )

    for name, value in (
        summary[
            "generalization_gaps"
        ].items()
    ):
        print(
            f"  {name}: "
            f"{value:.4f}"
        )

    print()
    print(
        "Leakage checks:"
    )

    for name, value in (
        summary[
            "leakage_checks"
        ].items()
    ):
        print(
            f"  {name}: "
            f"{value}"
        )

    print()
    print(
        "Saved results to: "
        f"{OUTPUT_DIR}"
    )

    print()
    print(
        "PASS: Random-window, "
        "grouped-recording, and "
        "leave-one-load-out "
        "evaluations completed."
    )


if __name__ == "__main__":
    main()
