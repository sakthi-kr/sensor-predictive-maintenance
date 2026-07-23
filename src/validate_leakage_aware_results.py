from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "leakage_aware_evaluation"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
ANALYSIS_PATH = RESULTS_DIR / "analysis_summary.json"
STRATEGY_PATH = RESULTS_DIR / "strategy_comparison.csv"
FOLD_PATH = RESULTS_DIR / "fold_metrics.csv"
REPORT_PATH = PROJECT_ROOT / "docs" / "leakage_aware_evaluation.md"
README_PATH = PROJECT_ROOT / "README.md"
DATA_README_PATH = PROJECT_ROOT / "data" / "README.md"
VALIDATION_JSON_PATH = RESULTS_DIR / "validation_report.json"
VALIDATION_CSV_PATH = RESULTS_DIR / "validation_checks.csv"


EXPECTED_STRATEGIES = {
    "random_window",
    "grouped_recording",
    "leave_one_load_out",
}

REQUIRED_PLOTS = [
    RESULTS_DIR / "strategy_comparison.png",
    RESULTS_DIR / "leave_one_load_out_by_load.png",
    RESULTS_DIR / "confusion_matrix_random_window.png",
    RESULTS_DIR / "confusion_matrix_grouped_recording.png",
    RESULTS_DIR / "confusion_matrix_leave_one_load_out.png",
]

ABSOLUTE_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
    re.compile(r"[A-Za-z]:/Users/", re.IGNORECASE),
    re.compile(r"/Users/"),
    re.compile(r"/home/"),
]


def check(name: str, passed: bool, details: str) -> dict[str, object]:
    return {
        "name": name,
        "passed": bool(passed),
        "details": details,
    }


def scores_are_valid(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce")
    return bool(
        values.notna().all()
        and values.between(0.0, 1.0, inclusive="both").all()
    )


def contains_absolute_path(text: str) -> bool:
    return any(pattern.search(text) for pattern in ABSOLUTE_PATH_PATTERNS)


def run_validation() -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []

    required_files = [
        SUMMARY_PATH,
        ANALYSIS_PATH,
        STRATEGY_PATH,
        FOLD_PATH,
        REPORT_PATH,
        README_PATH,
        DATA_README_PATH,
        *REQUIRED_PLOTS,
    ]

    missing = [path for path in required_files if not path.exists()]
    checks.append(
        check(
            "required_files",
            not missing,
            "All required files exist."
            if not missing
            else "Missing: " + ", ".join(str(path) for path in missing),
        )
    )

    if missing:
        return checks

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    strategies = pd.read_csv(STRATEGY_PATH)
    folds = pd.read_csv(FOLD_PATH)
    readme = README_PATH.read_text(encoding="utf-8")
    data_readme = DATA_README_PATH.read_text(encoding="utf-8")
    report = REPORT_PATH.read_text(encoding="utf-8")

    expected_dataset = {
        "n_windows": 800,
        "n_source_recordings": 16,
        "n_classes": 4,
        "windows_per_recording": 50,
    }
    observed_dataset = {
        key: summary.get("dataset", {}).get(key)
        for key in expected_dataset
    }
    checks.append(
        check(
            "dataset_dimensions",
            observed_dataset == expected_dataset,
            f"Observed={observed_dataset}; expected={expected_dataset}",
        )
    )

    observed_strategies = set(strategies["strategy"])
    checks.append(
        check(
            "strategy_set",
            observed_strategies == EXPECTED_STRATEGIES,
            f"Observed={sorted(observed_strategies)}",
        )
    )

    fold_counts = folds.groupby("strategy")["fold"].nunique().to_dict()
    four_folds = (
        set(fold_counts) == EXPECTED_STRATEGIES
        and all(count == 4 for count in fold_counts.values())
    )
    checks.append(
        check(
            "four_folds_per_strategy",
            four_folds,
            f"Fold counts={fold_counts}",
        )
    )

    metric_columns = [
        "window_accuracy",
        "window_balanced_accuracy",
        "window_macro_f1",
        "window_weighted_f1",
        "recording_accuracy",
        "recording_macro_f1",
    ]
    missing_metrics = [column for column in metric_columns if column not in folds]
    valid_metrics = not missing_metrics and all(
        scores_are_valid(folds[column]) for column in metric_columns
    )
    checks.append(
        check(
            "metric_ranges",
            valid_metrics,
            "All metrics are finite and within [0, 1]."
            if valid_metrics
            else f"Missing or invalid metrics: {missing_metrics}",
        )
    )

    overlap = (
        folds.groupby("strategy")["overlapping_recording_count"]
        .max()
        .astype(int)
        .to_dict()
    )
    expected_overlap = {
        "random_window": 16,
        "grouped_recording": 0,
        "leave_one_load_out": 0,
    }
    checks.append(
        check(
            "recording_overlap",
            overlap == expected_overlap,
            f"Observed={overlap}; expected={expected_overlap}",
        )
    )

    tied = bool(analysis.get("load_performance_tied", False))
    tie_fields_valid = (
        not tied
        or (
            analysis.get("hardest_held_out_load_hp") is None
            and analysis.get("easiest_held_out_load_hp") is None
        )
    )
    checks.append(
        check(
            "tied_load_reporting",
            tie_fields_valid,
            (
                f"tied={tied}, hardest={analysis.get('hardest_held_out_load_hp')}, "
                f"easiest={analysis.get('easiest_held_out_load_hp')}"
            ),
        )
    )

    required_readme_text = [
        "16 source recordings",
        "Grouped recording CV",
        "Leave-one-load-out",
        "zero overlapping source recordings",
        "Fault-diameter generalization",
    ]
    missing_readme_text = [item for item in required_readme_text if item not in readme]
    checks.append(
        check(
            "readme_current",
            not missing_readme_text,
            "README contains the current experiment."
            if not missing_readme_text
            else f"Missing text: {missing_readme_text}",
        )
    )

    stale_text = [
        "For the first baseline version, four `.mat` files are used",
        "It is not yet a fully leakage-aware industrial validation",
        "A future grouped evaluation should separate",
    ]
    stale_present = [item for item in stale_text if item in readme]
    checks.append(
        check(
            "readme_no_stale_claims",
            not stale_present,
            "No stale baseline-only claims remain."
            if not stale_present
            else f"Stale text: {stale_present}",
        )
    )

    required_data_text = [
        "Current 16-Recording Benchmark",
        "cwru_load_benchmark.csv",
        "--verify-only",
        "48 kHz",
        "12 kHz",
    ]
    missing_data_text = [item for item in required_data_text if item not in data_readme]
    checks.append(
        check(
            "data_documentation",
            not missing_data_text,
            "Dataset documentation is current."
            if not missing_data_text
            else f"Missing text: {missing_data_text}",
        )
    )

    placeholders = [
        "{summary_json",
        "{analysis_summary",
        "{dataset",
        "PYCODE_EOF",
    ]
    placeholder_present = [item for item in placeholders if item in report]
    checks.append(
        check(
            "report_rendered",
            not placeholder_present,
            "The technical report contains rendered values."
            if not placeholder_present
            else f"Unrendered placeholders: {placeholder_present}",
        )
    )

    documentation_text = "\n".join([readme, data_readme, report])
    checks.append(
        check(
            "portable_documentation_paths",
            not contains_absolute_path(documentation_text),
            "No machine-specific absolute paths were found.",
        )
    )

    empty_plots = [path.name for path in REQUIRED_PLOTS if path.stat().st_size == 0]
    checks.append(
        check(
            "plot_files",
            not empty_plots,
            "All required plots are non-empty."
            if not empty_plots
            else f"Empty plots: {empty_plots}",
        )
    )

    return checks


def write_reports(checks: list[dict[str, object]]) -> None:
    passed = sum(int(item["passed"]) for item in checks)
    report = {
        "overall_status": "passed" if passed == len(checks) else "failed",
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_JSON_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    with VALIDATION_CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["name", "passed", "details"])
        writer.writeheader()
        writer.writerows(checks)


def main() -> int:
    checks = run_validation()
    write_reports(checks)

    print("=" * 80)
    print("LEAKAGE-AWARE RESULT VALIDATION")
    print("=" * 80)

    for item in checks:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status:<4} | {item['name']:<32} | {item['details']}")

    passed = sum(int(item["passed"]) for item in checks)
    print()
    print(f"Checks passed: {passed}/{len(checks)}")

    if passed != len(checks):
        print("FAIL: Result validation found problems.")
        return 1

    print("PASS: All leakage-aware result checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
