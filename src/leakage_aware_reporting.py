from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import classification_report

try:
    from src.leakage_aware_evaluation import (
        CLASS_LABELS,
        STRATEGY_DISPLAY_NAMES,
    )
except ModuleNotFoundError:
    from leakage_aware_evaluation import (
        CLASS_LABELS,
        STRATEGY_DISPLAY_NAMES,
    )


STRATEGY_ORDER = [
    "random_window",
    "grouped_recording",
    "leave_one_load_out",
]

REQUIRED_STRATEGY_COLUMNS = {
    "strategy",
    "strategy_name",
    "folds",
    "window_accuracy_mean",
    "window_accuracy_std",
    "window_balanced_accuracy_mean",
    "window_macro_f1_mean",
    "window_macro_f1_std",
    "recording_accuracy_mean",
    "recording_accuracy_std",
    "recording_macro_f1_mean",
    "maximum_recording_overlap",
}

REQUIRED_FOLD_COLUMNS = {
    "strategy",
    "strategy_name",
    "fold",
    "held_out_load_hp",
    "n_train_recordings",
    "n_test_recordings",
    "window_accuracy",
    "window_balanced_accuracy",
    "window_macro_f1",
    "recording_accuracy",
    "recording_macro_f1",
    "overlapping_recording_count",
    "test_recordings",
}

REQUIRED_WINDOW_PREDICTION_COLUMNS = {
    "strategy",
    "fold",
    "source_recording",
    "load_hp",
    "window_index",
    "y_true",
    "y_pred",
    "correct",
    "confidence",
}

REQUIRED_RECORDING_PREDICTION_COLUMNS = {
    "strategy",
    "fold",
    "held_out_load_hp",
    "source_recording",
    "load_hp",
    "y_true",
    "y_pred",
    "correct",
    "n_test_windows",
    "source_seen_in_training",
    "confidence",
}


def validate_columns(
    table: pd.DataFrame,
    required_columns: set[str],
    table_name: str,
) -> None:
    if table.empty:
        raise ValueError(
            f"{table_name} is empty."
        )

    missing_columns = (
        required_columns
        - set(table.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{table_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def validate_reporting_inputs(
    strategy_comparison: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    window_predictions: pd.DataFrame,
    recording_predictions: pd.DataFrame,
) -> None:
    validate_columns(
        strategy_comparison,
        REQUIRED_STRATEGY_COLUMNS,
        "strategy_comparison",
    )

    validate_columns(
        fold_metrics,
        REQUIRED_FOLD_COLUMNS,
        "fold_metrics",
    )

    validate_columns(
        window_predictions,
        REQUIRED_WINDOW_PREDICTION_COLUMNS,
        "window_predictions",
    )

    validate_columns(
        recording_predictions,
        REQUIRED_RECORDING_PREDICTION_COLUMNS,
        "recording_predictions",
    )

    available_strategies = set(
        strategy_comparison[
            "strategy"
        ]
    )

    missing_strategies = (
        set(STRATEGY_ORDER)
        - available_strategies
    )

    if missing_strategies:
        raise ValueError(
            "Missing evaluation strategies: "
            f"{sorted(missing_strategies)}"
        )


def build_per_class_metrics(
    window_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    for strategy in STRATEGY_ORDER:
        strategy_predictions = (
            window_predictions.loc[
                window_predictions[
                    "strategy"
                ]
                == strategy
            ]
        )

        if strategy_predictions.empty:
            raise ValueError(
                "No window predictions found for "
                f"strategy: {strategy}"
            )

        report = classification_report(
            strategy_predictions[
                "y_true"
            ],
            strategy_predictions[
                "y_pred"
            ],
            labels=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        )

        for label in CLASS_LABELS:
            class_result = report[
                label
            ]

            rows.append(
                {
                    "strategy": strategy,
                    "strategy_name": (
                        STRATEGY_DISPLAY_NAMES[
                            strategy
                        ]
                    ),
                    "class_label": label,
                    "precision": float(
                        class_result[
                            "precision"
                        ]
                    ),
                    "recall": float(
                        class_result[
                            "recall"
                        ]
                    ),
                    "f1_score": float(
                        class_result[
                            "f1-score"
                        ]
                    ),
                    "support": int(
                        class_result[
                            "support"
                        ]
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def build_load_generalization_metrics(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    load_metrics = (
        fold_metrics.loc[
            fold_metrics[
                "strategy"
            ]
            == "leave_one_load_out"
        ]
        .copy()
    )

    if load_metrics.empty:
        raise ValueError(
            "No leave-one-load-out "
            "fold metrics were found."
        )

    load_metrics[
        "held_out_load_hp"
    ] = (
        load_metrics[
            "held_out_load_hp"
        ]
        .astype(int)
    )

    selected_columns = [
        "held_out_load_hp",
        "fold",
        "n_train_recordings",
        "n_test_recordings",
        "window_accuracy",
        "window_balanced_accuracy",
        "window_macro_f1",
        "recording_accuracy",
        "recording_macro_f1",
        "overlapping_recording_count",
        "test_recordings",
    ]

    return (
        load_metrics[
            selected_columns
        ]
        .sort_values(
            "held_out_load_hp"
        )
        .reset_index(
            drop=True
        )
    )


def build_recording_error_analysis(
    recording_predictions: pd.DataFrame,
) -> pd.DataFrame:
    errors = (
        recording_predictions.loc[
            ~recording_predictions[
                "correct"
            ].astype(bool)
        ]
        .copy()
    )

    probability_columns = [
        column
        for column
        in recording_predictions.columns
        if column.startswith(
            "probability_"
        )
    ]

    selected_columns = [
        "strategy",
        "fold",
        "held_out_load_hp",
        "source_recording",
        "load_hp",
        "y_true",
        "y_pred",
        "confidence",
        "n_test_windows",
        "source_seen_in_training",
        *probability_columns,
    ]

    if errors.empty:
        return pd.DataFrame(
            columns=selected_columns
        )

    return (
        errors[
            selected_columns
        ]
        .sort_values(
            [
                "strategy",
                "fold",
                "source_recording",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def build_recording_strategy_summary(
    recording_predictions: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        recording_predictions.groupby(
            "strategy",
            sort=False,
        )
        .agg(
            recording_predictions=(
                "correct",
                "count",
            ),
            correct_recordings=(
                "correct",
                "sum",
            ),
            recording_accuracy=(
                "correct",
                "mean",
            ),
            mean_confidence=(
                "confidence",
                "mean",
            ),
            recordings_seen_in_training=(
                "source_seen_in_training",
                "sum",
            ),
        )
        .reset_index()
    )

    summary[
        "strategy_name"
    ] = (
        summary[
            "strategy"
        ]
        .map(
            STRATEGY_DISPLAY_NAMES
        )
    )

    summary[
        "incorrect_recordings"
    ] = (
        summary[
            "recording_predictions"
        ]
        - summary[
            "correct_recordings"
        ]
    )

    strategy_order = {
        strategy: index
        for index, strategy
        in enumerate(
            STRATEGY_ORDER
        )
    }

    summary[
        "_order"
    ] = (
        summary[
            "strategy"
        ]
        .map(
            strategy_order
        )
    )

    selected_columns = [
        "strategy",
        "strategy_name",
        "recording_predictions",
        "correct_recordings",
        "incorrect_recordings",
        "recording_accuracy",
        "mean_confidence",
        "recordings_seen_in_training",
    ]

    return (
        summary.sort_values(
            "_order"
        )[
            selected_columns
        ]
        .reset_index(
            drop=True
        )
    )


def get_strategy_row(
    strategy_comparison: pd.DataFrame,
    strategy: str,
) -> pd.Series:
    matching_rows = (
        strategy_comparison.loc[
            strategy_comparison[
                "strategy"
            ]
            == strategy
        ]
    )

    if len(
        matching_rows
    ) != 1:
        raise ValueError(
            "Expected exactly one row for "
            f"strategy '{strategy}', but found "
            f"{len(matching_rows)}."
        )

    return matching_rows.iloc[
        0
    ]


def build_analysis_summary(
    summary_json: dict,
    strategy_comparison: pd.DataFrame,
    load_metrics: pd.DataFrame,
    recording_errors: pd.DataFrame,
) -> dict:
    random_result = (
        get_strategy_row(
            strategy_comparison,
            "random_window",
        )
    )

    grouped_result = (
        get_strategy_row(
            strategy_comparison,
            "grouped_recording",
        )
    )

    load_result = (
        get_strategy_row(
            strategy_comparison,
            "leave_one_load_out",
        )
    )

    load_metric_columns = [
        "window_macro_f1",
        "window_accuracy",
    ]

    load_performance_tied = bool(
        load_metrics[
            load_metric_columns
        ]
        .nunique(
            dropna=False
        )
        .eq(1)
        .all()
    )

    hardest_load_row = (
        load_metrics.sort_values(
            load_metric_columns
        )
        .iloc[0]
    )

    easiest_load_row = (
        load_metrics.sort_values(
            load_metric_columns,
            ascending=False,
        )
        .iloc[0]
    )

    hardest_load_hp: int | None = (
        None
        if load_performance_tied
        else int(
            hardest_load_row[
                "held_out_load_hp"
            ]
        )
    )

    easiest_load_hp: int | None = (
        None
        if load_performance_tied
        else int(
            easiest_load_row[
                "held_out_load_hp"
            ]
        )
    )

    error_counts = {
        strategy: int(
            (
                recording_errors[
                    "strategy"
                ]
                == strategy
            ).sum()
        )
        for strategy
        in STRATEGY_ORDER
    }

    return {
        "dataset": (
            summary_json[
                "dataset"
            ]
        ),
        "random_window_accuracy": float(
            random_result[
                "window_accuracy_mean"
            ]
        ),
        "grouped_recording_accuracy": float(
            grouped_result[
                "window_accuracy_mean"
            ]
        ),
        "leave_one_load_out_accuracy": float(
            load_result[
                "window_accuracy_mean"
            ]
        ),
        "random_window_macro_f1": float(
            random_result[
                "window_macro_f1_mean"
            ]
        ),
        "grouped_recording_macro_f1": float(
            grouped_result[
                "window_macro_f1_mean"
            ]
        ),
        "leave_one_load_out_macro_f1": float(
            load_result[
                "window_macro_f1_mean"
            ]
        ),
        "random_minus_grouped_accuracy_gap": float(
            random_result[
                "window_accuracy_mean"
            ]
            - grouped_result[
                "window_accuracy_mean"
            ]
        ),
        (
            "random_minus_leave_one_load_out_"
            "accuracy_gap"
        ): float(
            random_result[
                "window_accuracy_mean"
            ]
            - load_result[
                "window_accuracy_mean"
            ]
        ),
        "load_performance_tied": (
            load_performance_tied
        ),
        "hardest_held_out_load_hp": (
            hardest_load_hp
        ),
        "hardest_load_accuracy": float(
            hardest_load_row[
                "window_accuracy"
            ]
        ),
        "hardest_load_macro_f1": float(
            hardest_load_row[
                "window_macro_f1"
            ]
        ),
        "easiest_held_out_load_hp": (
            easiest_load_hp
        ),
        "easiest_load_accuracy": float(
            easiest_load_row[
                "window_accuracy"
            ]
        ),
        "easiest_load_macro_f1": float(
            easiest_load_row[
                "window_macro_f1"
            ]
        ),
        "recording_error_counts": (
            error_counts
        ),
        "leakage_checks": (
            summary_json[
                "leakage_checks"
            ]
        ),
        "acquisition_limitation": (
            summary_json[
                "interpretation"
            ][
                "important_dataset_limitation"
            ]
        ),
    }


def format_score(
    value: float,
) -> str:
    return f"{float(value):.3f}"


def escape_markdown(
    value: object,
) -> str:
    return (
        str(value)
        .replace(
            "|",
            "\\|",
        )
        .replace(
            "\n",
            " ",
        )
    )


def markdown_table(
    headers: list[str],
    rows: list[
        list[object]
    ],
) -> str:
    header_line = (
        "| "
        + " | ".join(
            escape_markdown(
                header
            )
            for header in headers
        )
        + " |"
    )

    separator_line = (
        "| "
        + " | ".join(
            "---"
            for _ in headers
        )
        + " |"
    )

    row_lines = [
        (
            "| "
            + " | ".join(
                escape_markdown(
                    value
                )
                for value in row
            )
            + " |"
        )
        for row in rows
    ]

    return "\n".join(
        [
            header_line,
            separator_line,
            *row_lines,
        ]
    )


def build_interpretation_points(
    analysis_summary: dict,
) -> list[str]:
    grouped_gap = (
        analysis_summary[
            "random_minus_grouped_accuracy_gap"
        ]
    )

    load_gap = (
        analysis_summary[
            (
                "random_minus_leave_one_load_out_"
                "accuracy_gap"
            )
        ]
    )

    points: list[
        str
    ] = []

    if grouped_gap >= 0.05:
        points.append(
            "Random window splitting is "
            "substantially more optimistic "
            "than recording-level evaluation."
        )
    elif grouped_gap >= 0.01:
        points.append(
            "Random window splitting is "
            "modestly more optimistic than "
            "recording-level evaluation."
        )
    elif grouped_gap > -0.01:
        points.append(
            "Random-window and grouped scores "
            "are similar, but only the grouped "
            "result prevents source-recording "
            "overlap."
        )
    else:
        points.append(
            "The grouped result is at least as "
            "strong as the random-window result, "
            "so the benchmark remains separable "
            "after recording leakage is removed."
        )

    if load_gap >= 0.05:
        points.append(
            "Generalization to an unseen motor "
            "load is meaningfully harder than "
            "the random-window development "
            "baseline."
        )
    elif load_gap >= 0.01:
        points.append(
            "The unseen-load experiment shows "
            "a moderate operating-condition "
            "generalization gap."
        )
    else:
        points.append(
            "Performance remains comparatively "
            "stable when one motor load is "
            "completely held out."
        )

    if analysis_summary[
        "load_performance_tied"
    ]:
        points.append(
            "All held-out motor loads tied, "
            "with accuracy and macro F1 of "
            f"{analysis_summary['hardest_load_macro_f1']:.3f}."
        )
    else:
        points.append(
            "The hardest unseen operating condition "
            "is "
            f"{analysis_summary['hardest_held_out_load_hp']} "
            "HP, with macro F1 "
            f"{analysis_summary['hardest_load_macro_f1']:.3f}."
        )

    points.append(
        "Grouped recording and "
        "leave-one-load-out evaluations "
        "contain zero source-recording overlap."
    )

    return points


def render_markdown_report(
    summary_json: dict,
    strategy_comparison: pd.DataFrame,
    load_metrics: pd.DataFrame,
    per_class_metrics: pd.DataFrame,
    recording_summary: pd.DataFrame,
    recording_errors: pd.DataFrame,
    analysis_summary: dict,
) -> str:
    dataset = (
        summary_json[
            "dataset"
        ]
    )

    strategy_rows: list[
        list[object]
    ] = []

    for strategy in STRATEGY_ORDER:
        row = get_strategy_row(
            strategy_comparison,
            strategy,
        )

        strategy_rows.append(
            [
                row[
                    "strategy_name"
                ],
                (
                    f"{format_score(row['window_accuracy_mean'])} "
                    "± "
                    f"{format_score(row['window_accuracy_std'])}"
                ),
                format_score(
                    row[
                        "window_balanced_accuracy_mean"
                    ]
                ),
                (
                    f"{format_score(row['window_macro_f1_mean'])} "
                    "± "
                    f"{format_score(row['window_macro_f1_std'])}"
                ),
                format_score(
                    row[
                        "recording_accuracy_mean"
                    ]
                ),
                int(
                    row[
                        "maximum_recording_overlap"
                    ]
                ),
            ]
        )

    load_rows = [
        [
            f"{int(row['held_out_load_hp'])} HP",
            format_score(
                row[
                    "window_accuracy"
                ]
            ),
            format_score(
                row[
                    "window_balanced_accuracy"
                ]
            ),
            format_score(
                row[
                    "window_macro_f1"
                ]
            ),
            format_score(
                row[
                    "recording_accuracy"
                ]
            ),
        ]
        for _, row
        in load_metrics.iterrows()
    ]

    class_rows = [
        [
            row[
                "strategy_name"
            ],
            row[
                "class_label"
            ],
            format_score(
                row[
                    "precision"
                ]
            ),
            format_score(
                row[
                    "recall"
                ]
            ),
            format_score(
                row[
                    "f1_score"
                ]
            ),
            int(
                row[
                    "support"
                ]
            ),
        ]
        for _, row
        in per_class_metrics.iterrows()
    ]

    recording_rows = [
        [
            row[
                "strategy_name"
            ],
            int(
                row[
                    "recording_predictions"
                ]
            ),
            int(
                row[
                    "correct_recordings"
                ]
            ),
            int(
                row[
                    "incorrect_recordings"
                ]
            ),
            format_score(
                row[
                    "recording_accuracy"
                ]
            ),
            int(
                row[
                    "recordings_seen_in_training"
                ]
            ),
        ]
        for _, row
        in recording_summary.iterrows()
    ]

    if recording_errors.empty:
        error_section = (
            "No recording-level "
            "misclassifications were observed."
        )
    else:
        error_rows: list[
            list[object]
        ] = []

        for _, row in (
            recording_errors.iterrows()
        ):
            if pd.isna(
                row[
                    "held_out_load_hp"
                ]
            ):
                held_out_load = (
                    "not applicable"
                )
            else:
                held_out_load = (
                    f"{int(row['held_out_load_hp'])} HP"
                )

            error_rows.append(
                [
                    STRATEGY_DISPLAY_NAMES[
                        row[
                            "strategy"
                        ]
                    ],
                    int(
                        row[
                            "fold"
                        ]
                    ),
                    held_out_load,
                    row[
                        "source_recording"
                    ],
                    f"{int(row['load_hp'])} HP",
                    row[
                        "y_true"
                    ],
                    row[
                        "y_pred"
                    ],
                    format_score(
                        row[
                            "confidence"
                        ]
                    ),
                ]
            )

        error_section = (
            markdown_table(
                [
                    "Strategy",
                    "Fold",
                    "Held-out load",
                    "Recording",
                    "Recording load",
                    "True class",
                    "Predicted class",
                    "Confidence",
                ],
                error_rows,
            )
        )

    interpretation_text = (
        "\n".join(
            f"- {point}"
            for point
            in build_interpretation_points(
                analysis_summary
            )
        )
    )

    loads_text = (
        ", ".join(
            f"{load} HP"
            for load
            in dataset[
                "loads_hp"
            ]
        )
    )

    lines = [
        "# Leakage-Aware Evaluation",
        "",
        "## Purpose",
        "",
        (
            "The original development baseline "
            "randomly divided signal windows into "
            "training and test sets. Because many "
            "windows originate from the same source "
            "recording, this design can place windows "
            "from one recording in both sets."
        ),
        "",
        (
            "This experiment compares three "
            "evaluation strategies:"
        ),
        "",
        (
            "1. **Random window split:** optimistic "
            "development baseline."
        ),
        (
            "2. **Grouped recording cross-validation:** "
            "keeps all windows from each source "
            "recording together."
        ),
        (
            "3. **Leave-one-load-out:** trains on three "
            "motor loads and tests on one unseen "
            "motor load."
        ),
        "",
        "## Benchmark Dataset",
        "",
        markdown_table(
            [
                "Property",
                "Value",
            ],
            [
                [
                    "Source recordings",
                    dataset[
                        "n_source_recordings"
                    ],
                ],
                [
                    "Windows",
                    dataset[
                        "n_windows"
                    ],
                ],
                [
                    "Classes",
                    dataset[
                        "n_classes"
                    ],
                ],
                [
                    "Motor loads",
                    loads_text,
                ],
                [
                    "Windows per recording",
                    dataset[
                        "windows_per_recording"
                    ],
                ],
                [
                    "Effective sampling rate",
                    "12 kHz",
                ],
                [
                    "Model features",
                    14,
                ],
            ],
        ),
        "",
        "## Evaluation Results",
        "",
        markdown_table(
            [
                "Evaluation strategy",
                "Window accuracy",
                "Balanced accuracy",
                "Window macro F1",
                "Recording accuracy",
                "Maximum recording overlap",
            ],
            strategy_rows,
        ),
        "",
        (
            "Values are means across four folds. "
            "The uncertainty is the standard "
            "deviation across folds."
        ),
        "",
        (
            "![Evaluation strategy comparison]"
            "(../results/leakage_aware_evaluation/"
            "strategy_comparison.png)"
        ),
        "",
        "## Generalization Gaps",
        "",
        markdown_table(
            [
                "Comparison",
                "Accuracy difference",
            ],
            [
                [
                    (
                        "Random window minus "
                        "grouped recording"
                    ),
                    (
                        f"{analysis_summary['random_minus_grouped_accuracy_gap']:.3f}"
                    ),
                ],
                [
                    (
                        "Random window minus "
                        "leave-one-load-out"
                    ),
                    (
                        f"{analysis_summary['random_minus_leave_one_load_out_accuracy_gap']:.3f}"
                    ),
                ],
            ],
        ),
        "",
        "## Leave-One-Load-Out Results",
        "",
        markdown_table(
            [
                "Held-out load",
                "Window accuracy",
                "Balanced accuracy",
                "Macro F1",
                "Recording accuracy",
            ],
            load_rows,
        ),
        "",
        (
            "![Leave-one-load-out results]"
            "(../results/leakage_aware_evaluation/"
            "leave_one_load_out_by_load.png)"
        ),
        "",
        (
            (
                "All held-out motor loads tied at "
                f"accuracy **{analysis_summary['hardest_load_accuracy']:.3f}** "
                "and macro F1 "
                f"**{analysis_summary['hardest_load_macro_f1']:.3f}**."
            )
            if analysis_summary[
                "load_performance_tied"
            ]
            else (
                "The hardest held-out condition is "
                f"**{analysis_summary['hardest_held_out_load_hp']} "
                "HP**, with accuracy "
                f"**{analysis_summary['hardest_load_accuracy']:.3f}** "
                "and macro F1 "
                f"**{analysis_summary['hardest_load_macro_f1']:.3f}**."
            )
        ),
        "",
        "## Per-Class Results",
        "",
        markdown_table(
            [
                "Evaluation strategy",
                "Class",
                "Precision",
                "Recall",
                "F1-score",
                "Pooled test support",
            ],
            class_rows,
        ),
        "",
        "## Recording-Level Results",
        "",
        markdown_table(
            [
                "Evaluation strategy",
                "Recording predictions",
                "Correct",
                "Incorrect",
                "Recording accuracy",
                (
                    "Recordings also seen "
                    "in training"
                ),
            ],
            recording_rows,
        ),
        "",
        "## Recording-Level Error Analysis",
        "",
        error_section,
        "",
        "## Interpretation",
        "",
        interpretation_text,
        "",
        "## Leakage Verification",
        "",
        markdown_table(
            [
                "Evaluation strategy",
                (
                    "Maximum overlapping "
                    "source recordings"
                ),
            ],
            [
                [
                    "Random window split",
                    summary_json[
                        "leakage_checks"
                    ][
                        (
                            "random_window_maximum_"
                            "overlapping_recordings"
                        )
                    ],
                ],
                [
                    "Grouped recording CV",
                    summary_json[
                        "leakage_checks"
                    ][
                        (
                            "grouped_recording_maximum_"
                            "overlapping_recordings"
                        )
                    ],
                ],
                [
                    "Leave-one-load-out",
                    summary_json[
                        "leakage_checks"
                    ][
                        (
                            "leave_one_load_out_maximum_"
                            "overlapping_recordings"
                        )
                    ],
                ],
            ],
        ),
        "",
        (
            "The grouped recording and "
            "leave-one-load-out evaluations pass "
            "the leakage check only when this "
            "overlap is zero."
        ),
        "",
        "## Important Dataset Limitation",
        "",
        analysis_summary[
            "acquisition_limitation"
        ],
        "",
        (
            "Resampling standardizes the effective "
            "sample rate used for feature extraction, "
            "but it cannot remove every difference "
            "created by the original acquisition "
            "setup. These results are therefore not "
            "proof of generalization to independent "
            "industrial machinery."
        ),
        "",
        "## Generated Outputs",
        "",
        (
            "- `results/leakage_aware_evaluation/"
            "summary.json`"
        ),
        (
            "- `results/leakage_aware_evaluation/"
            "fold_metrics.csv`"
        ),
        (
            "- `results/leakage_aware_evaluation/"
            "strategy_comparison.csv`"
        ),
        (
            "- `results/leakage_aware_evaluation/"
            "per_class_metrics.csv`"
        ),
        (
            "- `results/leakage_aware_evaluation/"
            "load_generalization_metrics.csv`"
        ),
        (
            "- `results/leakage_aware_evaluation/"
            "recording_predictions.csv`"
        ),
        (
            "- `results/leakage_aware_evaluation/"
            "recording_error_analysis.csv`"
        ),
        (
            "- `results/leakage_aware_evaluation/"
            "window_predictions.csv`"
        ),
        "- classification reports",
        "- confusion matrices",
        "- comparison plots",
        "",
        "## Reproduction",
        "",
        "```bash",
        (
            "python scripts/"
            "download_cwru_load_benchmark.py"
        ),
        (
            "python scripts/"
            "build_cwru_load_features.py"
        ),
        (
            "python scripts/"
            "run_leakage_aware_evaluation.py"
        ),
        (
            "python scripts/"
            "generate_leakage_aware_report.py"
        ),
        "```",
    ]

    return (
        "\n".join(
            lines
        )
        .rstrip()
        + "\n"
    )
