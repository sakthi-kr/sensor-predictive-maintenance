import pandas as pd

from src.leakage_aware_evaluation import (
    CLASS_LABELS,
)
from src.leakage_aware_reporting import (
    STRATEGY_ORDER,
    build_per_class_metrics,
    build_recording_error_analysis,
    build_recording_strategy_summary,
)


def make_window_predictions() -> pd.DataFrame:
    rows = []

    for strategy in STRATEGY_ORDER:
        for index, label in enumerate(
            CLASS_LABELS
        ):
            rows.append(
                {
                    "strategy": strategy,
                    "fold": 1,
                    "source_recording": (
                        f"{strategy}_{index}.mat"
                    ),
                    "load_hp": index,
                    "window_index": 0,
                    "y_true": label,
                    "y_pred": label,
                    "correct": True,
                    "confidence": 0.9,
                }
            )

    return pd.DataFrame(
        rows
    )


def make_recording_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": [
                "random_window",
                "grouped_recording",
                "leave_one_load_out",
            ],
            "fold": [
                1,
                1,
                1,
            ],
            "held_out_load_hp": [
                float("nan"),
                float("nan"),
                0,
            ],
            "source_recording": [
                "97.mat",
                "105.mat",
                "118.mat",
            ],
            "load_hp": [
                0,
                0,
                0,
            ],
            "y_true": [
                "normal",
                "inner_race_fault",
                "ball_fault",
            ],
            "y_pred": [
                "normal",
                "ball_fault",
                "ball_fault",
            ],
            "correct": [
                True,
                False,
                True,
            ],
            "n_test_windows": [
                10,
                10,
                10,
            ],
            "source_seen_in_training": [
                True,
                False,
                False,
            ],
            "confidence": [
                0.95,
                0.60,
                0.88,
            ],
            "probability_ball_fault": [
                0.01,
                0.60,
                0.88,
            ],
            "probability_inner_race_fault": [
                0.01,
                0.35,
                0.04,
            ],
            "probability_normal": [
                0.97,
                0.03,
                0.04,
            ],
            "probability_outer_race_fault": [
                0.01,
                0.02,
                0.04,
            ],
        }
    )


def test_per_class_metrics_contains_every_strategy_and_class(
) -> None:
    metrics = (
        build_per_class_metrics(
            make_window_predictions()
        )
    )

    assert len(metrics) == (
        len(STRATEGY_ORDER)
        * len(CLASS_LABELS)
    )

    assert set(
        metrics[
            "strategy"
        ]
    ) == set(
        STRATEGY_ORDER
    )

    assert set(
        metrics[
            "class_label"
        ]
    ) == set(
        CLASS_LABELS
    )

    assert (
        metrics[
            "f1_score"
        ]
        == 1.0
    ).all()


def test_recording_error_analysis_keeps_only_errors(
) -> None:
    errors = (
        build_recording_error_analysis(
            make_recording_predictions()
        )
    )

    assert len(errors) == 1

    error = errors.iloc[
        0
    ]

    assert (
        error[
            "source_recording"
        ]
        == "105.mat"
    )

    assert (
        error[
            "y_true"
        ]
        == "inner_race_fault"
    )

    assert (
        error[
            "y_pred"
        ]
        == "ball_fault"
    )


def test_recording_strategy_summary_counts_errors(
) -> None:
    summary = (
        build_recording_strategy_summary(
            make_recording_predictions()
        )
    )

    grouped = (
        summary.loc[
            summary[
                "strategy"
            ]
            == "grouped_recording"
        ]
        .iloc[0]
    )

    assert (
        grouped[
            "recording_predictions"
        ]
        == 1
    )

    assert (
        grouped[
            "incorrect_recordings"
        ]
        == 1
    )

    assert (
        grouped[
            "recording_accuracy"
        ]
        == 0.0
    )
