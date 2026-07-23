import numpy as np
import pandas as pd
import pytest

from src.validate_pipeline import (
    build_prediction_table,
)


def test_build_prediction_table() -> None:
    y_true = pd.Series(
        ["normal", "fault", "normal"],
        index=[10, 11, 12],
    )

    y_pred = np.array(
        ["normal", "fault", "fault"]
    )

    probabilities = np.array(
        [
            [0.90, 0.10],
            [0.20, 0.80],
            [0.40, 0.60],
        ]
    )

    table = build_prediction_table(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        class_labels=["normal", "fault"],
    )

    assert len(table) == 3

    assert list(table["source_row_index"]) == [
        10,
        11,
        12,
    ]

    assert list(table["correct"]) == [
        True,
        True,
        False,
    ]

    assert np.allclose(
        table["prediction_confidence"],
        [0.90, 0.80, 0.60],
    )

    assert "probability_normal" in table.columns
    assert "probability_fault" in table.columns


def test_build_prediction_table_detects_shape_error() -> None:
    with pytest.raises(
        ValueError,
        match="Probability rows",
    ):
        build_prediction_table(
            y_true=["normal", "fault"],
            y_pred=["normal", "fault"],
            probabilities=np.array(
                [
                    [0.9, 0.1],
                ]
            ),
            class_labels=["normal", "fault"],
        )
