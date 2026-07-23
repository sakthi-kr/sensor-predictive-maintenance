from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PROJECT_ROOT / "README.md"
REPORT_PATH = (
    PROJECT_ROOT
    / "results"
    / "sensor_validation_report.json"
)

START_MARKER = (
    "<!-- VALIDATION_TOOLKIT_SECTION_START -->"
)

END_MARKER = (
    "<!-- VALIDATION_TOOLKIT_SECTION_END -->"
)


def load_report(path: Path) -> dict[str, Any]:
    """Load and validate the sensor-validation report."""
    if not path.exists():
        raise FileNotFoundError(
            f"Validation report not found: {path}\n"
            "Run `python src/validate_pipeline.py` first."
        )

    with path.open("r", encoding="utf-8") as file:
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
    """Format one model metric for the README table."""
    value = metrics.get(name)

    if isinstance(value, (int, float)):
        return f"{value:.4f}"

    return "Not available"


def build_section(
    report: dict[str, Any],
) -> str:
    """Build the complete marked README section."""
    summary = report.get("summary", {})
    metadata = report.get("metadata", {})

    if not isinstance(summary, dict):
        summary = {}

    if not isinstance(metadata, dict):
        metadata = {}

    metrics = metadata.get("metrics", {})

    if not isinstance(metrics, dict):
        metrics = {}

    return f"""\
{START_MARKER}
## Reusable Validation Toolkit Integration

The project integrates the separate `ml-testing-validation-toolkit` package to validate both the generated feature table and the saved model outputs.

The validation workflow checks:

- required feature and metadata columns
- missing, infinite, and duplicate values
- allowed bearing-fault labels
- numerical feature ranges
- class representation
- prediction lengths and labels
- probability-matrix validity
- prediction-confidence ranges
- metric regression thresholds
- confusion-matrix consistency

### Current Validation Result

| Item | Result |
|---|---:|
| Overall status | {summary.get("status", "UNKNOWN")} |
| Checks passed | {summary.get("passed_checks", "Not available")} / {summary.get("total_checks", "Not available")} |
| Accuracy | {format_metric(metrics, "accuracy")} |
| Macro F1-score | {format_metric(metrics, "macro_f1")} |
| Weighted F1-score | {format_metric(metrics, "weighted_f1")} |

Run the validation workflow after installing the local toolkit:

```bash
python -m pip install -e ../ml-testing-validation-toolkit
python src/validate_pipeline.py
```

Generated outputs:

```text
results/sensor_validation_report.json
results/sensor_validation_checks.csv
results/sensor_validation_predictions.csv
```

Detailed documentation:

```text
docs/sensor_validation.md
```

### Important Limitation

The current model uses a random window-level split. Similar or overlapping windows extracted from the same source recording may occur in both training and test sets.

The strong current scores should therefore be treated as development-baseline results rather than evidence of production-level generalization. A future grouped evaluation should separate entire recordings or source files between training and testing.
{END_MARKER}"""


def replace_or_append_section(
    readme: str,
    section: str,
) -> str:
    """Replace an existing marked section or append a new one."""
    has_start = START_MARKER in readme
    has_end = END_MARKER in readme

    if has_start != has_end:
        raise ValueError(
            "README contains only one validation marker. "
            "Remove the incomplete marked section and "
            "rerun the script."
        )

    if has_start and has_end:
        before, remainder = readme.split(
            START_MARKER,
            1,
        )

        _, after = remainder.split(
            END_MARKER,
            1,
        )

        parts = [
            before.rstrip(),
            section.strip(),
        ]

        if after.strip():
            parts.append(after.strip())

        return "\n\n".join(parts) + "\n"

    return (
        readme.rstrip()
        + "\n\n"
        + section.strip()
        + "\n"
    )


def main() -> None:
    if not README_PATH.exists():
        raise FileNotFoundError(
            f"README not found: {README_PATH}"
        )

    report = load_report(REPORT_PATH)

    current_readme = README_PATH.read_text(
        encoding="utf-8"
    )

    section = build_section(report)

    updated_readme = replace_or_append_section(
        current_readme,
        section,
    )

    README_PATH.write_text(
        updated_readme,
        encoding="utf-8",
    )

    print(f"Updated: {README_PATH}")


if __name__ == "__main__":
    main()
