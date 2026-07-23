import pandas as pd

from src.leakage_aware_reporting import (
    build_analysis_summary,
    build_interpretation_points,
)


def make_strategy_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": [
                "random_window",
                "grouped_recording",
                "leave_one_load_out",
            ],
            "window_accuracy_mean": [
                1.0,
                1.0,
                1.0,
            ],
            "window_macro_f1_mean": [
                1.0,
                1.0,
                1.0,
            ],
        }
    )


def make_load_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "held_out_load_hp": [
                0,
                1,
                2,
                3,
            ],
            "window_accuracy": [
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            "window_macro_f1": [
                1.0,
                1.0,
                1.0,
                1.0,
            ],
        }
    )


def test_tied_load_results_do_not_invent_hardest_load(
) -> None:
    summary_json = {
        "dataset": {
            "n_windows": 800,
        },
        "leakage_checks": {
            (
                "random_window_maximum_"
                "overlapping_recordings"
            ): 16,
            (
                "grouped_recording_maximum_"
                "overlapping_recordings"
            ): 0,
            (
                "leave_one_load_out_maximum_"
                "overlapping_recordings"
            ): 0,
        },
        "interpretation": {
            "important_dataset_limitation": (
                "Acquisition limitation."
            ),
        },
    }

    recording_errors = pd.DataFrame(
        columns=[
            "strategy",
        ]
    )

    analysis = build_analysis_summary(
        summary_json=summary_json,
        strategy_comparison=(
            make_strategy_comparison()
        ),
        load_metrics=(
            make_load_metrics()
        ),
        recording_errors=(
            recording_errors
        ),
    )

    assert bool(
        analysis[
            "load_performance_tied"
        ]
    )

    assert (
        analysis[
            "hardest_held_out_load_hp"
        ]
        is None
    )

    assert (
        analysis[
            "easiest_held_out_load_hp"
        ]
        is None
    )

    points = build_interpretation_points(
        analysis
    )

    assert any(
        "All held-out motor loads tied"
        in point
        for point in points
    )
