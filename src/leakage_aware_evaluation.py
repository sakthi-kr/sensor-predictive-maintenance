from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedShuffleSplit,
)

try:
    from src.features import (
        MODEL_FEATURE_COLUMNS,
    )
except ModuleNotFoundError:
    from features import (
        MODEL_FEATURE_COLUMNS,
    )


RANDOM_STATE = 42
N_ESTIMATORS = 200

N_RANDOM_SPLITS = 4
RANDOM_TEST_SIZE = 0.25

N_GROUPED_SPLITS = 4

CLASS_LABELS = [
    "ball_fault",
    "inner_race_fault",
    "normal",
    "outer_race_fault",
]

STRATEGY_DISPLAY_NAMES = {
    "random_window": (
        "Random window split"
    ),
    "grouped_recording": (
        "Grouped recording CV"
    ),
    "leave_one_load_out": (
        "Leave-one-load-out"
    ),
}


@dataclass(frozen=True)
class EvaluationSplit:
    strategy: str
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    held_out_load_hp: int | None = None


def validate_evaluation_table(
    feature_table: pd.DataFrame,
) -> None:
    required_columns = (
        set(MODEL_FEATURE_COLUMNS)
        | {
            "label",
            "source_recording",
            "load_hp",
            "window_index",
        }
    )

    missing_columns = (
        required_columns
        - set(feature_table.columns)
    )

    if missing_columns:
        raise ValueError(
            "Feature table is missing required "
            "evaluation columns: "
            f"{sorted(missing_columns)}"
        )

    if feature_table.empty:
        raise ValueError(
            "Feature table is empty."
        )

    if (
        feature_table[
            "source_recording"
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            "source_recording contains "
            "missing values."
        )

    if (
        feature_table[
            "load_hp"
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            "load_hp contains missing values."
        )

    unknown_labels = (
        set(
            feature_table[
                "label"
            ].unique()
        )
        - set(CLASS_LABELS)
    )

    if unknown_labels:
        raise ValueError(
            "Unexpected class labels: "
            f"{sorted(unknown_labels)}"
        )

    feature_values = (
        feature_table[
            MODEL_FEATURE_COLUMNS
        ]
        .to_numpy(dtype=float)
    )

    if not np.isfinite(
        feature_values
    ).all():
        raise ValueError(
            "Model feature columns contain "
            "NaN or infinite values."
        )

    recording_label_counts = (
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
        recording_label_counts == 1
    ).all():
        raise ValueError(
            "Each source_recording must map "
            "to exactly one class label."
        )

    recording_load_counts = (
        feature_table[
            [
                "source_recording",
                "load_hp",
            ]
        ]
        .drop_duplicates()
        .groupby(
            "source_recording"
        )["load_hp"]
        .nunique()
    )

    if not (
        recording_load_counts == 1
    ).all():
        raise ValueError(
            "Each source_recording must map "
            "to exactly one motor load."
        )


def build_random_forest(
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def create_random_window_splits(
    feature_table: pd.DataFrame,
) -> list[EvaluationSplit]:
    y = (
        feature_table[
            "label"
        ]
        .to_numpy()
    )

    splitter = (
        StratifiedShuffleSplit(
            n_splits=N_RANDOM_SPLITS,
            test_size=RANDOM_TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    splits = []

    for fold, (
        train_indices,
        test_indices,
    ) in enumerate(
        splitter.split(
            np.zeros(
                (
                    len(feature_table),
                    1,
                )
            ),
            y,
        ),
        start=1,
    ):
        splits.append(
            EvaluationSplit(
                strategy="random_window",
                fold=fold,
                train_indices=(
                    train_indices
                ),
                test_indices=(
                    test_indices
                ),
            )
        )

    return splits


def create_grouped_recording_splits(
    feature_table: pd.DataFrame,
) -> list[EvaluationSplit]:
    """Create deterministic recording-level folds.

    The benchmark contains four classes, four motor loads,
    and one source recording for every class-load pair.

    Each grouped test fold therefore contains:
    - one recording from every class;
    - one recording from every motor load;
    - no recording used in the corresponding training set.

    A rotating class offset keeps this grouped evaluation
    different from leave-one-load-out evaluation.
    """
    groups = (
        feature_table[
            "source_recording"
        ]
        .astype(str)
        .to_numpy()
    )

    labels = (
        feature_table[
            "label"
        ]
        .astype(str)
        .to_numpy()
    )

    loads = (
        feature_table[
            "load_hp"
        ]
        .astype(int)
        .to_numpy()
    )

    recording_table = (
        feature_table[
            [
                "source_recording",
                "label",
                "load_hp",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    recording_counts = (
        recording_table[
            "source_recording"
        ]
        .value_counts()
    )

    if not (
        recording_counts == 1
    ).all():
        raise ValueError(
            "Every source recording must map to "
            "exactly one class and one load."
        )

    expected_loads = set(
        int(value)
        for value in sorted(
            recording_table[
                "load_hp"
            ].unique()
        )
    )

    if (
        len(expected_loads)
        != N_GROUPED_SPLITS
    ):
        raise ValueError(
            "The deterministic grouped design requires "
            f"{N_GROUPED_SPLITS} distinct loads, but found "
            f"{sorted(expected_loads)}."
        )

    recordings_by_label: dict[
        str,
        pd.DataFrame,
    ] = {}

    for label in CLASS_LABELS:
        class_recordings = (
            recording_table[
                recording_table[
                    "label"
                ]
                == label
            ]
            .sort_values(
                [
                    "load_hp",
                    "source_recording",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        if (
            len(class_recordings)
            != N_GROUPED_SPLITS
        ):
            raise ValueError(
                f"Class '{label}' must have exactly "
                f"{N_GROUPED_SPLITS} source recordings, "
                f"but found {len(class_recordings)}."
            )

        class_loads = set(
            int(value)
            for value in class_recordings[
                "load_hp"
            ]
        )

        if class_loads != expected_loads:
            raise ValueError(
                f"Class '{label}' does not contain "
                "one recording for every load. "
                f"Observed loads: {sorted(class_loads)}"
            )

        recordings_by_label[
            label
        ] = class_recordings

    splits: list[
        EvaluationSplit
    ] = []

    all_tested_recordings: list[
        str
    ] = []

    for fold_index in range(
        N_GROUPED_SPLITS
    ):
        test_recordings: set[
            str
        ] = set()

        # The rotating offset creates a Latin-square-like
        # assignment. Each fold receives every class and
        # every load, while each recording is tested once.
        for (
            class_offset,
            label,
        ) in enumerate(
            CLASS_LABELS
        ):
            class_recordings = (
                recordings_by_label[
                    label
                ]
            )

            recording_position = (
                fold_index
                + class_offset
            ) % N_GROUPED_SPLITS

            selected_recording = str(
                class_recordings.iloc[
                    recording_position
                ][
                    "source_recording"
                ]
            )

            test_recordings.add(
                selected_recording
            )

        test_mask = np.isin(
            groups,
            sorted(
                test_recordings
            ),
        )

        test_indices = np.flatnonzero(
            test_mask
        )

        train_indices = np.flatnonzero(
            ~test_mask
        )

        train_recordings = set(
            groups[
                train_indices
            ]
        )

        observed_test_recordings = set(
            groups[
                test_indices
            ]
        )

        recording_overlap = (
            train_recordings
            & observed_test_recordings
        )

        if recording_overlap:
            raise RuntimeError(
                "Grouped split contains recording "
                "leakage: "
                f"{sorted(recording_overlap)}"
            )

        test_labels = set(
            labels[
                test_indices
            ]
        )

        if test_labels != set(
            CLASS_LABELS
        ):
            raise RuntimeError(
                "Grouped test fold does not contain "
                "all classes. "
                f"Fold={fold_index + 1}, "
                f"labels={sorted(test_labels)}"
            )

        test_loads = set(
            int(value)
            for value in loads[
                test_indices
            ]
        )

        if test_loads != expected_loads:
            raise RuntimeError(
                "Grouped test fold does not contain "
                "all motor loads. "
                f"Fold={fold_index + 1}, "
                f"loads={sorted(test_loads)}"
            )

        all_tested_recordings.extend(
            sorted(
                observed_test_recordings
            )
        )

        splits.append(
            EvaluationSplit(
                strategy=(
                    "grouped_recording"
                ),
                fold=(
                    fold_index + 1
                ),
                train_indices=(
                    train_indices
                ),
                test_indices=(
                    test_indices
                ),
            )
        )

    expected_recordings = set(
        recording_table[
            "source_recording"
        ].astype(str)
    )

    if (
        set(all_tested_recordings)
        != expected_recordings
    ):
        raise RuntimeError(
            "Grouped folds did not test every "
            "source recording."
        )

    if (
        len(all_tested_recordings)
        != len(
            set(
                all_tested_recordings
            )
        )
    ):
        raise RuntimeError(
            "At least one source recording appears "
            "in more than one grouped test fold."
        )

    return splits


def create_leave_one_load_out_splits(
    feature_table: pd.DataFrame,
) -> list[EvaluationSplit]:
    loads = (
        feature_table[
            "load_hp"
        ]
        .to_numpy()
    )

    y = (
        feature_table[
            "label"
        ]
        .to_numpy()
    )

    unique_loads = sorted(
        int(value)
        for value in np.unique(
            loads
        )
    )

    splits = []

    for fold, held_out_load in enumerate(
        unique_loads,
        start=1,
    ):
        train_indices = np.flatnonzero(
            loads != held_out_load
        )

        test_indices = np.flatnonzero(
            loads == held_out_load
        )

        if (
            train_indices.size == 0
            or test_indices.size == 0
        ):
            raise RuntimeError(
                "Invalid leave-one-load-out "
                f"split for load "
                f"{held_out_load}."
            )

        test_labels = set(
            y[test_indices]
        )

        if test_labels != set(
            CLASS_LABELS
        ):
            raise RuntimeError(
                "Held-out load does not "
                "contain all classes. "
                f"Load={held_out_load}, "
                f"labels={sorted(test_labels)}"
            )

        splits.append(
            EvaluationSplit(
                strategy=(
                    "leave_one_load_out"
                ),
                fold=fold,
                train_indices=(
                    train_indices
                ),
                test_indices=(
                    test_indices
                ),
                held_out_load_hp=(
                    held_out_load
                ),
            )
        )

    return splits


def align_probabilities(
    model: RandomForestClassifier,
    probabilities: np.ndarray,
) -> np.ndarray:
    aligned = np.zeros(
        (
            probabilities.shape[0],
            len(CLASS_LABELS),
        ),
        dtype=float,
    )

    class_to_column = {
        label: column
        for column, label
        in enumerate(
            model.classes_
        )
    }

    for (
        target_column,
        label,
    ) in enumerate(
        CLASS_LABELS
    ):
        if label in class_to_column:
            source_column = (
                class_to_column[
                    label
                ]
            )

            aligned[
                :,
                target_column,
            ] = probabilities[
                :,
                source_column,
            ]

    return aligned


def aggregate_recording_predictions(
    window_predictions: pd.DataFrame,
    train_recordings: set[str],
) -> pd.DataFrame:
    probability_columns = [
        f"probability_{label}"
        for label in CLASS_LABELS
    ]

    rows = []

    grouped_predictions = (
        window_predictions.groupby(
            "source_recording",
            sort=True,
        )
    )

    for (
        source_recording,
        group,
    ) in grouped_predictions:
        true_labels = (
            group[
                "y_true"
            ]
            .unique()
        )

        if len(true_labels) != 1:
            raise ValueError(
                f"Recording "
                f"{source_recording} "
                "has multiple true labels."
            )

        mean_probabilities = (
            group[
                probability_columns
            ]
            .mean()
        )

        predicted_label = (
            CLASS_LABELS[
                int(
                    np.argmax(
                        mean_probabilities
                        .to_numpy()
                    )
                )
            ]
        )

        row = {
            "strategy": (
                group[
                    "strategy"
                ].iloc[0]
            ),
            "fold": int(
                group[
                    "fold"
                ].iloc[0]
            ),
            "held_out_load_hp": (
                group[
                    "held_out_load_hp"
                ].iloc[0]
            ),
            "source_recording": (
                source_recording
            ),
            "load_hp": int(
                group[
                    "load_hp"
                ].iloc[0]
            ),
            "y_true": (
                true_labels[0]
            ),
            "y_pred": (
                predicted_label
            ),
            "correct": (
                predicted_label
                == true_labels[0]
            ),
            "n_test_windows": int(
                len(group)
            ),
            "source_seen_in_training": (
                source_recording
                in train_recordings
            ),
            "confidence": float(
                mean_probabilities.max()
            ),
        }

        for column in (
            probability_columns
        ):
            row[column] = float(
                mean_probabilities[
                    column
                ]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_split(
    feature_table: pd.DataFrame,
    split: EvaluationSplit,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    train_table = (
        feature_table.iloc[
            split.train_indices
        ]
    )

    test_table = (
        feature_table.iloc[
            split.test_indices
        ]
    )

    X_train = train_table[
        MODEL_FEATURE_COLUMNS
    ]

    X_test = test_table[
        MODEL_FEATURE_COLUMNS
    ]

    y_train = train_table[
        "label"
    ]

    y_test = test_table[
        "label"
    ]

    model = build_random_forest(
        random_state=(
            RANDOM_STATE
            + split.fold
        )
    )

    model.fit(
        X_train,
        y_train,
    )

    y_pred = model.predict(
        X_test
    )

    raw_probabilities = (
        model.predict_proba(
            X_test
        )
    )

    probabilities = (
        align_probabilities(
            model,
            raw_probabilities,
        )
    )

    train_recordings = set(
        train_table[
            "source_recording"
        ].astype(str)
    )

    test_recordings = set(
        test_table[
            "source_recording"
        ].astype(str)
    )

    overlapping_recordings = (
        train_recordings
        & test_recordings
    )

    if (
        split.strategy
        != "random_window"
        and overlapping_recordings
    ):
        raise RuntimeError(
            f"{split.strategy} "
            "unexpectedly contains "
            "recording leakage: "
            f"{sorted(overlapping_recordings)}"
        )

    held_out_load = (
        np.nan
        if split.held_out_load_hp
        is None
        else int(
            split.held_out_load_hp
        )
    )

    window_predictions = (
        pd.DataFrame(
            {
                "strategy": (
                    split.strategy
                ),
                "fold": (
                    split.fold
                ),
                "held_out_load_hp": (
                    held_out_load
                ),
                "row_index": (
                    test_table
                    .index
                    .to_numpy()
                ),
                "source_recording": (
                    test_table[
                        "source_recording"
                    ]
                    .astype(str)
                    .to_numpy()
                ),
                "load_hp": (
                    test_table[
                        "load_hp"
                    ]
                    .astype(int)
                    .to_numpy()
                ),
                "window_index": (
                    test_table[
                        "window_index"
                    ]
                    .astype(int)
                    .to_numpy()
                ),
                "y_true": (
                    y_test.to_numpy()
                ),
                "y_pred": (
                    y_pred
                ),
                "correct": (
                    y_pred
                    == y_test.to_numpy()
                ),
                "confidence": (
                    probabilities.max(
                        axis=1
                    )
                ),
            }
        )
    )

    for (
        column,
        label,
    ) in enumerate(
        CLASS_LABELS
    ):
        window_predictions[
            f"probability_{label}"
        ] = probabilities[
            :,
            column,
        ]

    recording_predictions = (
        aggregate_recording_predictions(
            window_predictions=(
                window_predictions
            ),
            train_recordings=(
                train_recordings
            ),
        )
    )

    window_accuracy = (
        accuracy_score(
            y_test,
            y_pred,
        )
    )

    window_balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            y_pred,
        )
    )

    window_macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    window_weighted_f1 = (
        f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    recording_accuracy = (
        accuracy_score(
            recording_predictions[
                "y_true"
            ],
            recording_predictions[
                "y_pred"
            ],
        )
    )

    recording_macro_f1 = (
        f1_score(
            recording_predictions[
                "y_true"
            ],
            recording_predictions[
                "y_pred"
            ],
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0,
        )
    )

    fold_metrics = {
        "strategy": (
            split.strategy
        ),
        "strategy_name": (
            STRATEGY_DISPLAY_NAMES[
                split.strategy
            ]
        ),
        "fold": (
            split.fold
        ),
        "held_out_load_hp": (
            held_out_load
        ),
        "n_train_windows": int(
            len(train_table)
        ),
        "n_test_windows": int(
            len(test_table)
        ),
        "n_train_recordings": int(
            len(train_recordings)
        ),
        "n_test_recordings": int(
            len(test_recordings)
        ),
        "overlapping_recording_count": int(
            len(
                overlapping_recordings
            )
        ),
        "recording_leakage_detected": bool(
            overlapping_recordings
        ),
        "train_recordings": (
            ";".join(
                sorted(
                    train_recordings
                )
            )
        ),
        "test_recordings": (
            ";".join(
                sorted(
                    test_recordings
                )
            )
        ),
        "train_loads": (
            ";".join(
                str(value)
                for value in sorted(
                    train_table[
                        "load_hp"
                    ].unique()
                )
            )
        ),
        "test_loads": (
            ";".join(
                str(value)
                for value in sorted(
                    test_table[
                        "load_hp"
                    ].unique()
                )
            )
        ),
        "window_accuracy": float(
            window_accuracy
        ),
        "window_balanced_accuracy": float(
            window_balanced_accuracy
        ),
        "window_macro_f1": float(
            window_macro_f1
        ),
        "window_weighted_f1": float(
            window_weighted_f1
        ),
        "recording_accuracy": float(
            recording_accuracy
        ),
        "recording_macro_f1": float(
            recording_macro_f1
        ),
        "classification_report": (
            classification_report(
                y_test,
                y_pred,
                labels=CLASS_LABELS,
                output_dict=True,
                zero_division=0,
            )
        ),
    }

    return (
        window_predictions,
        recording_predictions,
        fold_metrics,
    )


def run_all_evaluations(
    feature_table: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    validate_evaluation_table(
        feature_table
    )

    all_splits = (
        create_random_window_splits(
            feature_table
        )
        + create_grouped_recording_splits(
            feature_table
        )
        + create_leave_one_load_out_splits(
            feature_table
        )
    )

    window_tables = []
    recording_tables = []
    fold_metric_rows = []

    for split in all_splits:
        print(
            "Running "
            f"{STRATEGY_DISPLAY_NAMES[split.strategy]} "
            f"fold {split.fold}..."
        )

        (
            window_predictions,
            recording_predictions,
            fold_metrics,
        ) = evaluate_split(
            feature_table,
            split,
        )

        window_tables.append(
            window_predictions
        )

        recording_tables.append(
            recording_predictions
        )

        fold_metrics.pop(
            "classification_report"
        )

        fold_metric_rows.append(
            fold_metrics
        )

    return (
        pd.concat(
            window_tables,
            ignore_index=True,
        ),
        pd.concat(
            recording_tables,
            ignore_index=True,
        ),
        pd.DataFrame(
            fold_metric_rows
        ),
    )


def summarize_strategies(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        fold_metrics.groupby(
            [
                "strategy",
                "strategy_name",
            ],
            sort=False,
        )
        .agg(
            folds=(
                "fold",
                "count",
            ),
            window_accuracy_mean=(
                "window_accuracy",
                "mean",
            ),
            window_accuracy_std=(
                "window_accuracy",
                "std",
            ),
            window_balanced_accuracy_mean=(
                "window_balanced_accuracy",
                "mean",
            ),
            window_macro_f1_mean=(
                "window_macro_f1",
                "mean",
            ),
            window_macro_f1_std=(
                "window_macro_f1",
                "std",
            ),
            recording_accuracy_mean=(
                "recording_accuracy",
                "mean",
            ),
            recording_accuracy_std=(
                "recording_accuracy",
                "std",
            ),
            recording_macro_f1_mean=(
                "recording_macro_f1",
                "mean",
            ),
            maximum_recording_overlap=(
                "overlapping_recording_count",
                "max",
            ),
        )
        .reset_index()
    )

    standard_deviation_columns = [
        column
        for column
        in summary.columns
        if column.endswith(
            "_std"
        )
    ]

    summary[
        standard_deviation_columns
    ] = (
        summary[
            standard_deviation_columns
        ]
        .fillna(0.0)
    )

    strategy_order = {
        strategy: index
        for index, strategy
        in enumerate(
            [
                "random_window",
                "grouped_recording",
                "leave_one_load_out",
            ]
        )
    }

    summary["_order"] = (
        summary[
            "strategy"
        ]
        .map(
            strategy_order
        )
    )

    return (
        summary.sort_values(
            "_order"
        )
        .drop(
            columns="_order"
        )
        .reset_index(
            drop=True
        )
    )
