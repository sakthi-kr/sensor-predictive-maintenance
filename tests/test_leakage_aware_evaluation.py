import numpy as np
import pandas as pd

from src.features import (
    MODEL_FEATURE_COLUMNS,
)
from src.leakage_aware_evaluation import (
    CLASS_LABELS,
    aggregate_recording_predictions,
    create_grouped_recording_splits,
    create_leave_one_load_out_splits,
    create_random_window_splits,
    validate_evaluation_table,
)


def make_synthetic_feature_table(
    windows_per_recording: int = 12,
) -> pd.DataFrame:
    rows = []

    random_generator = (
        np.random.default_rng(
            123
        )
    )

    for (
        label_index,
        label,
    ) in enumerate(
        CLASS_LABELS
    ):
        for load_hp in range(4):
            source_recording = (
                f"{label_index}_"
                f"{load_hp}.mat"
            )

            for window_index in range(
                windows_per_recording
            ):
                feature_values = (
                    random_generator.normal(
                        loc=float(
                            label_index
                        ),
                        scale=0.1,
                        size=len(
                            MODEL_FEATURE_COLUMNS
                        ),
                    )
                )

                row = {
                    feature: value
                    for feature, value
                    in zip(
                        MODEL_FEATURE_COLUMNS,
                        feature_values,
                        strict=True,
                    )
                }

                row.update(
                    {
                        "label": label,
                        "source_recording": (
                            source_recording
                        ),
                        "load_hp": (
                            load_hp
                        ),
                        "window_index": (
                            window_index
                        ),
                    }
                )

                rows.append(
                    row
                )

    return pd.DataFrame(
        rows
    )


def test_random_window_split_contains_recording_overlap(
) -> None:
    table = (
        make_synthetic_feature_table(
            windows_per_recording=20
        )
    )

    splits = (
        create_random_window_splits(
            table
        )
    )

    assert len(splits) == 4

    groups = (
        table[
            "source_recording"
        ]
        .to_numpy()
    )

    for split in splits:
        train_groups = set(
            groups[
                split.train_indices
            ]
        )

        test_groups = set(
            groups[
                split.test_indices
            ]
        )

        assert (
            train_groups
            & test_groups
        )


def test_grouped_recording_splits_have_no_overlap(
) -> None:
    table = (
        make_synthetic_feature_table()
    )

    splits = (
        create_grouped_recording_splits(
            table
        )
    )

    assert len(splits) == 4

    groups = (
        table[
            "source_recording"
        ]
        .to_numpy()
    )

    tested_groups = []

    for split in splits:
        train_groups = set(
            groups[
                split.train_indices
            ]
        )

        test_groups = set(
            groups[
                split.test_indices
            ]
        )

        assert not (
            train_groups
            & test_groups
        )

        tested_groups.extend(
            test_groups
        )

        test_labels = set(
            table.iloc[
                split.test_indices
            ]["label"]
        )

        assert (
            test_labels
            == set(CLASS_LABELS)
        )

        test_loads = set(
            table.iloc[
                split.test_indices
            ]["load_hp"]
        )

        assert test_loads == {
            0,
            1,
            2,
            3,
        }

    assert sorted(
        tested_groups
    ) == sorted(
        set(groups)
    )


def test_leave_one_load_out_holds_out_exact_load(
) -> None:
    table = (
        make_synthetic_feature_table()
    )

    splits = (
        create_leave_one_load_out_splits(
            table
        )
    )

    assert len(splits) == 4

    loads = (
        table[
            "load_hp"
        ]
        .to_numpy()
    )

    for split in splits:
        held_out_load = (
            split.held_out_load_hp
        )

        assert (
            held_out_load
            is not None
        )

        assert set(
            loads[
                split.test_indices
            ]
        ) == {
            held_out_load
        }

        assert (
            held_out_load
            not in set(
                loads[
                    split.train_indices
                ]
            )
        )


def test_recording_probability_aggregation(
) -> None:
    window_predictions = (
        pd.DataFrame(
            {
                "strategy": [
                    "grouped_recording"
                ]
                * 3,
                "fold": [
                    1,
                    1,
                    1,
                ],
                "held_out_load_hp": [
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                "source_recording": [
                    "105.mat"
                ]
                * 3,
                "load_hp": [
                    0,
                    0,
                    0,
                ],
                "y_true": [
                    "inner_race_fault"
                ]
                * 3,
                "probability_ball_fault": [
                    0.05,
                    0.05,
                    0.10,
                ],
                (
                    "probability_"
                    "inner_race_fault"
                ): [
                    0.80,
                    0.75,
                    0.70,
                ],
                "probability_normal": [
                    0.10,
                    0.10,
                    0.10,
                ],
                (
                    "probability_"
                    "outer_race_fault"
                ): [
                    0.05,
                    0.10,
                    0.10,
                ],
            }
        )
    )

    recording_predictions = (
        aggregate_recording_predictions(
            window_predictions=(
                window_predictions
            ),
            train_recordings={
                "97.mat",
                "118.mat",
            },
        )
    )

    assert (
        len(
            recording_predictions
        )
        == 1
    )

    result = (
        recording_predictions.iloc[0]
    )

    assert (
        result[
            "y_pred"
        ]
        == "inner_race_fault"
    )

    assert bool(
        result["correct"]
    )

    assert not bool(
        result[
            "source_seen_in_training"
        ]
    )

    assert (
        result[
            "n_test_windows"
        ]
        == 3
    )


def test_evaluation_table_validation_passes(
) -> None:
    table = (
        make_synthetic_feature_table()
    )

    validate_evaluation_table(
        table
    )
