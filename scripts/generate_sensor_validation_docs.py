from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "results" / "sensor_validation_report.json"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "sensor_validation.md"


def load_report() -> dict[str, Any]:
    """Load and validate the structured sensor-validation report."""
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Validation report not found: {REPORT_PATH}\n"
            "Run `python src/validate_pipeline.py` first."
        )

    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report, dict):
        raise TypeError(
            "Validation report must contain a JSON object."
        )

    return report


def format_metric(
    metrics: dict[str, Any],
    name: str,
) -> str:
    """Format one metric for the Markdown results table."""
    value = metrics.get(name)

    if isinstance(value, (int, float)):
        return f"{value:.4f}"

    return "Not available"


def markdown_cell(value: Any) -> str:
    """Escape content before inserting it into a Markdown table."""
    return (
        str(value)
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )


def build_check_table(
    checks: list[dict[str, Any]],
) -> str:
    """Create Markdown rows for all validation checks."""
    rows: list[str] = []

    for check in checks:
        name = markdown_cell(
            check.get("name", "unknown")
        )
        status = markdown_cell(
            check.get("status", "UNKNOWN")
        )
        message = markdown_cell(
            check.get("message", "")
        )

        rows.append(
            f"| `{name}` | {status} | {message} |"
        )

    if not rows:
        rows.append(
            "| _No checks found_ | UNKNOWN | "
            "No check records were present. |"
        )

    return "\n".join(rows)


def main() -> None:
    report = load_report()

    summary = report.get("summary", {})
    metadata = report.get("metadata", {})
    checks = report.get("checks", [])

    if not isinstance(summary, dict):
        summary = {}

    if not isinstance(metadata, dict):
        metadata = {}

    if not isinstance(checks, list):
        checks = []

    metrics = metadata.get("metrics", {})

    if not isinstance(metrics, dict):
        metrics = {}

    known_limitation = metadata.get(
        "known_validation_limitation",
        (
            "The current baseline uses a random "
            "window-level train/test split."
        ),
    )

    valid_checks = [
        check
        for check in checks
        if isinstance(check, dict)
    ]

    check_table = build_check_table(valid_checks)

    content = f"""# Sensor Pipeline Validation

## Purpose

This document records the reusable data-quality and model-output validation applied to the bearing fault-classification pipeline.

The validation is implemented using the separate `ml-testing-validation-toolkit` repository.

## Validation Scope

The workflow validates:

- required feature-table columns
- missing values
- infinite numerical values
- duplicate signal windows
- allowed bearing-fault labels
- physically meaningful feature ranges
- class representation
- prediction and target lengths
- predicted label values
- class-probability shape and row sums
- prediction-confidence ranges
- metric regression thresholds
- confusion-matrix consistency

## Current Validation Summary

| Item | Result |
|---|---:|
| Overall status | {summary.get("status", "UNKNOWN")} |
| Total checks | {summary.get("total_checks", "Not available")} |
| Passed checks | {summary.get("passed_checks", "Not available")} |
| Failed checks | {summary.get("failed_checks", "Not available")} |
| Feature-table rows | {metadata.get("feature_table_rows", "Not available")} |
| Test samples | {metadata.get("test_samples", "Not available")} |

## Current Model Metrics

| Metric | Result |
|---|---:|
| Accuracy | {format_metric(metrics, "accuracy")} |
| Macro F1-score | {format_metric(metrics, "macro_f1")} |
| Weighted F1-score | {format_metric(metrics, "weighted_f1")} |

These thresholds are pipeline-regression checks. They are not production-readiness criteria.

## Individual Checks

| Check | Status | Message |
|---|---|---|
{check_table}

## Generated Outputs

```text
results/sensor_validation_report.json
results/sensor_validation_checks.csv
results/sensor_validation_predictions.csv
```

The JSON report contains structured metadata and detailed check results. The CSV files support manual inspection and downstream analysis.

## Known Validation Limitation

{known_limitation}

Windows extracted from the same original vibration recording can therefore appear in both training and test sets. Because neighbouring windows overlap and share signal characteristics, the current performance can be overestimated.

A stronger future evaluation should use:

- file-level or recording-level separation
- grouped train/test splitting
- unseen operating conditions
- unseen fault severities
- cross-domain or cross-machine testing

## Interpretation

Passing the validation report means that the current pipeline outputs are internally consistent and satisfy the configured development checks.

It does not prove that the model generalizes to unseen machines or production environments.
"""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        content,
        encoding="utf-8",
    )

    print(f"Created: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
