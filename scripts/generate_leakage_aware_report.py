from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.leakage_aware_reporting import (  # noqa: E402
    build_analysis_summary,
    build_load_generalization_metrics,
    build_per_class_metrics,
    build_recording_error_analysis,
    build_recording_strategy_summary,
    render_markdown_report,
    validate_reporting_inputs,
)


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "leakage_aware_evaluation"
)

DOCS_DIR = (
    PROJECT_ROOT
    / "docs"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "summary.json"
)

STRATEGY_COMPARISON_PATH = (
    RESULTS_DIR
    / "strategy_comparison.csv"
)

FOLD_METRICS_PATH = (
    RESULTS_DIR
    / "fold_metrics.csv"
)

WINDOW_PREDICTIONS_PATH = (
    RESULTS_DIR
    / "window_predictions.csv"
)

RECORDING_PREDICTIONS_PATH = (
    RESULTS_DIR
    / "recording_predictions.csv"
)

PER_CLASS_METRICS_PATH = (
    RESULTS_DIR
    / "per_class_metrics.csv"
)

LOAD_METRICS_PATH = (
    RESULTS_DIR
    / "load_generalization_metrics.csv"
)

RECORDING_ERRORS_PATH = (
    RESULTS_DIR
    / "recording_error_analysis.csv"
)

RECORDING_SUMMARY_PATH = (
    RESULTS_DIR
    / "recording_strategy_summary.csv"
)

ANALYSIS_SUMMARY_PATH = (
    RESULTS_DIR
    / "analysis_summary.json"
)

REPORT_PATH = (
    DOCS_DIR
    / "leakage_aware_evaluation.md"
)


def require_file(
    path: Path,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            "Required evaluation file "
            f"not found: {path}"
        )


def load_inputs() -> tuple[
    dict,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    required_paths = [
        SUMMARY_PATH,
        STRATEGY_COMPARISON_PATH,
        FOLD_METRICS_PATH,
        WINDOW_PREDICTIONS_PATH,
        RECORDING_PREDICTIONS_PATH,
    ]

    for path in required_paths:
        require_file(
            path
        )

    summary_json = json.loads(
        SUMMARY_PATH.read_text(
            encoding="utf-8",
        )
    )

    strategy_comparison = pd.read_csv(
        STRATEGY_COMPARISON_PATH
    )

    fold_metrics = pd.read_csv(
        FOLD_METRICS_PATH
    )

    window_predictions = pd.read_csv(
        WINDOW_PREDICTIONS_PATH
    )

    recording_predictions = pd.read_csv(
        RECORDING_PREDICTIONS_PATH
    )

    validate_reporting_inputs(
        strategy_comparison=(
            strategy_comparison
        ),
        fold_metrics=(
            fold_metrics
        ),
        window_predictions=(
            window_predictions
        ),
        recording_predictions=(
            recording_predictions
        ),
    )

    return (
        summary_json,
        strategy_comparison,
        fold_metrics,
        window_predictions,
        recording_predictions,
    )


def main() -> None:
    (
        summary_json,
        strategy_comparison,
        fold_metrics,
        window_predictions,
        recording_predictions,
    ) = load_inputs()

    per_class_metrics = (
        build_per_class_metrics(
            window_predictions
        )
    )

    load_metrics = (
        build_load_generalization_metrics(
            fold_metrics
        )
    )

    recording_errors = (
        build_recording_error_analysis(
            recording_predictions
        )
    )

    recording_summary = (
        build_recording_strategy_summary(
            recording_predictions
        )
    )

    analysis_summary = (
        build_analysis_summary(
            summary_json=(
                summary_json
            ),
            strategy_comparison=(
                strategy_comparison
            ),
            load_metrics=(
                load_metrics
            ),
            recording_errors=(
                recording_errors
            ),
        )
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_class_metrics.to_csv(
        PER_CLASS_METRICS_PATH,
        index=False,
    )

    load_metrics.to_csv(
        LOAD_METRICS_PATH,
        index=False,
    )

    recording_errors.to_csv(
        RECORDING_ERRORS_PATH,
        index=False,
    )

    recording_summary.to_csv(
        RECORDING_SUMMARY_PATH,
        index=False,
    )

    ANALYSIS_SUMMARY_PATH.write_text(
        json.dumps(
            analysis_summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = render_markdown_report(
        summary_json=(
            summary_json
        ),
        strategy_comparison=(
            strategy_comparison
        ),
        load_metrics=(
            load_metrics
        ),
        per_class_metrics=(
            per_class_metrics
        ),
        recording_summary=(
            recording_summary
        ),
        recording_errors=(
            recording_errors
        ),
        analysis_summary=(
            analysis_summary
        ),
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print("=" * 80)
    print(
        "LEAKAGE-AWARE RESULT ANALYSIS"
    )
    print("=" * 80)

    print(
        "Random-window accuracy     : "
        f"{analysis_summary['random_window_accuracy']:.4f}"
    )

    print(
        "Grouped-recording accuracy : "
        f"{analysis_summary['grouped_recording_accuracy']:.4f}"
    )

    print(
        "Unseen-load accuracy       : "
        f"{analysis_summary['leave_one_load_out_accuracy']:.4f}"
    )

    print(
        "Random minus grouped gap   : "
        f"{analysis_summary['random_minus_grouped_accuracy_gap']:.4f}"
    )

    print(
        "Random minus unseen-load   : "
        f"{analysis_summary['random_minus_leave_one_load_out_accuracy_gap']:.4f}"
    )

    if analysis_summary[
        "load_performance_tied"
    ]:
        print(
            "Held-out load comparison  : "
            "all loads tied"
        )

        print(
            "Held-out-load macro F1    : "
            f"{analysis_summary['hardest_load_macro_f1']:.4f}"
        )

    else:
        print(
            "Hardest held-out load      : "
            f"{analysis_summary['hardest_held_out_load_hp']} HP"
        )

        print(
            "Hardest-load macro F1      : "
            f"{analysis_summary['hardest_load_macro_f1']:.4f}"
        )

    print()
    print(
        "Recording-level errors:"
    )

    for strategy, count in (
        analysis_summary[
            "recording_error_counts"
        ].items()
    ):
        print(
            f"  {strategy:<22}: {count}"
        )

    print()
    print(
        "Saved per-class metrics   : "
        f"{PER_CLASS_METRICS_PATH}"
    )

    print(
        "Saved load metrics        : "
        f"{LOAD_METRICS_PATH}"
    )

    print(
        "Saved recording errors    : "
        f"{RECORDING_ERRORS_PATH}"
    )

    print(
        "Saved analysis summary    : "
        f"{ANALYSIS_SUMMARY_PATH}"
    )

    print(
        "Saved technical report    : "
        f"{REPORT_PATH}"
    )

    print()
    print(
        "PASS: Leakage-aware results "
        "were analyzed and documented."
    )


if __name__ == "__main__":
    main()
