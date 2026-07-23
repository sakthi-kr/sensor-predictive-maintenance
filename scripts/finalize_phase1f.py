from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PROJECT_ROOT / "README.md"
DATA_README_PATH = PROJECT_ROOT / "data" / "README.md"
RESULTS_DIR = PROJECT_ROOT / "results" / "leakage_aware_evaluation"
ANALYSIS_PATH = RESULTS_DIR / "analysis_summary.json"
STRATEGY_PATH = RESULTS_DIR / "strategy_comparison.csv"
LOAD_PATH = RESULTS_DIR / "load_generalization_metrics.csv"


STRATEGY_ORDER = [
    "random_window",
    "grouped_recording",
    "leave_one_load_out",
]


FILE_MATRIX = [
    ("Normal", "97.mat", "98.mat", "99.mat", "100.mat"),
    ("Inner-race fault, 0.007 in", "105.mat", "106.mat", "107.mat", "108.mat"),
    ("Ball fault, 0.007 in", "118.mat", "119.mat", "120.mat", "121.mat"),
    ("Outer-race fault, 0.007 in, 6 o'clock", "130.mat", "131.mat", "132.mat", "133.mat"),
]


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def format_score(value: str | float) -> str:
    return f"{float(value):.3f}"


def detect_tied_loads(load_rows: list[dict[str, str]]) -> bool:
    if not load_rows:
        raise ValueError("No leave-one-load-out rows were found.")

    accuracies = [float(row["window_accuracy"]) for row in load_rows]
    macro_f1_values = [float(row["window_macro_f1"]) for row in load_rows]

    tolerance = 1e-12

    return (
        max(accuracies) - min(accuracies) <= tolerance
        and max(macro_f1_values) - min(macro_f1_values) <= tolerance
    )


def build_data_readme() -> str:
    lines = [
        "# CWRU Dataset Instructions",
        "",
        "This project uses the Case Western Reserve University bearing vibration dataset.",
        "The raw MATLAB files are external data and are not committed to this repository.",
        "The versioned manifest stores the source filename, class, load, approximate speed, original sample rate, local path, and official download URL.",
        "",
        "## Current 16-Recording Benchmark",
        "",
        "| Condition | 0 HP | 1 HP | 2 HP | 3 HP |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for row in FILE_MATRIX:
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "Manifest:",
            "",
            "    data/manifests/cwru_load_benchmark.csv",
            "",
            "## Download and Verify",
            "",
            "Run from the repository root:",
            "",
            "    python scripts/download_cwru_load_benchmark.py",
            "",
            "Verify already-downloaded files without contacting the server:",
            "",
            "    python scripts/download_cwru_load_benchmark.py --verify-only",
            "",
            "The downloader creates the class folders under `data/raw/cwru/`, checks that all 16 files are readable MATLAB files, finds the drive-end signal, and verifies balance across classes and loads.",
            "",
            "## Sampling-Rate Standardization",
            "",
            "The selected normal recordings were originally sampled at 48 kHz.",
            "The selected fault recordings were collected at 12 kHz.",
            "Normal signals are resampled to 12 kHz with polyphase anti-alias filtering before feature extraction.",
            "This standardizes the effective feature-extraction rate, but it cannot guarantee that every acquisition-domain difference has been removed.",
            "",
            "## Local Data Policy",
            "",
            "Do not commit raw MATLAB files.",
            "The repository ignores `data/raw/` and `*.mat`, while the small manifest remains tracked for reproducibility.",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


def build_project_readme(
    analysis: dict,
    strategy_rows: list[dict[str, str]],
    load_rows: list[dict[str, str]],
) -> str:
    by_strategy = {
        row["strategy"]: row
        for row in strategy_rows
    }

    missing = set(STRATEGY_ORDER) - set(by_strategy)
    if missing:
        raise ValueError(f"Missing strategy rows: {sorted(missing)}")

    tied_loads = detect_tied_loads(load_rows)

    if tied_loads:
        load_statement = (
            "All four held-out motor loads tied at "
            f"{format_score(load_rows[0]['window_accuracy'])} accuracy and "
            f"{format_score(load_rows[0]['window_macro_f1'])} macro F1."
        )
    else:
        hardest = min(
            load_rows,
            key=lambda row: (
                float(row["window_macro_f1"]),
                float(row["window_accuracy"]),
            ),
        )
        load_statement = (
            f"The hardest held-out condition was {int(float(hardest['held_out_load_hp']))} HP, "
            f"with {format_score(hardest['window_accuracy'])} accuracy and "
            f"{format_score(hardest['window_macro_f1'])} macro F1."
        )

    result_rows = []
    for strategy in STRATEGY_ORDER:
        row = by_strategy[strategy]
        result_rows.append(
            "| "
            + " | ".join(
                [
                    row["strategy_name"],
                    format_score(row["window_accuracy_mean"]),
                    format_score(row["window_macro_f1_mean"]),
                    str(int(float(row["maximum_recording_overlap"]))),
                ]
            )
            + " |"
        )

    lines = [
        "# Sensor Predictive Maintenance",
        "",
        "[![Sensor Pipeline Tests](https://github.com/sakthi-kr/sensor-predictive-maintenance/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/sakthi-kr/sensor-predictive-maintenance/actions/workflows/tests.yml)",
        "",
        "## Summary",
        "",
        "A reproducible bearing-fault classification pipeline using vibration data from the Case Western Reserve University dataset.",
        "The current benchmark uses 16 source recordings covering four bearing conditions and four motor loads.",
        "It standardizes signals to 12 kHz, extracts 14 engineered time- and frequency-domain features from balanced windows, trains a Random Forest classifier, and compares three evaluation designs.",
        "",
        "## Why This Upgrade Matters",
        "",
        "A random window split can place windows from one source recording in both training and testing.",
        "This project therefore reports the random-window baseline alongside recording-grouped cross-validation and leave-one-load-out evaluation.",
        "The grouped and unseen-load evaluations explicitly enforce zero source-recording overlap.",
        "",
        "## Benchmark",
        "",
        "| Property | Value |",
        "| --- | ---: |",
        "| Source recordings | 16 |",
        "| Classes | 4 |",
        "| Motor loads | 0, 1, 2, and 3 HP |",
        "| Windows | 800 |",
        "| Windows per recording | 50 |",
        "| Window size | 2,048 samples |",
        "| Window step | 1,024 samples |",
        "| Effective sample rate | 12 kHz |",
        "| Model features | 14 |",
        "",
        "Classes:",
        "",
        "- normal",
        "- inner-race fault",
        "- ball fault",
        "- outer-race fault",
        "",
        "The versioned dataset manifest is stored at `data/manifests/cwru_load_benchmark.csv`.",
        "Raw `.mat` files remain local and are excluded from Git.",
        "",
        "## Evaluation Results",
        "",
        "| Evaluation strategy | Accuracy | Macro F1 | Maximum overlapping source recordings |",
        "| --- | ---: | ---: | ---: |",
        *result_rows,
        "",
        load_statement,
        "",
        "The key methodological result is that grouped recording and leave-one-load-out evaluation both use zero overlapping source recordings, while the random-window baseline contains all 16 source recordings in both training and test data.",
        "",
        "![Evaluation strategy comparison](results/leakage_aware_evaluation/strategy_comparison.png)",
        "",
        "![Leave-one-load-out comparison](results/leakage_aware_evaluation/leave_one_load_out_by_load.png)",
        "",
        "Detailed metrics, per-class results, recording-level predictions, confusion matrices, and limitations are documented in [`docs/leakage_aware_evaluation.md`](docs/leakage_aware_evaluation.md).",
        "",
        "## Pipeline",
        "",
        "1. Read the versioned dataset manifest.",
        "2. Download and verify 16 MATLAB recordings.",
        "3. Resample normal recordings from 48 kHz to 12 kHz.",
        "4. Select 50 evenly distributed windows per recording.",
        "5. Extract 14 time- and frequency-domain features.",
        "6. Train a class-weighted Random Forest with 200 trees.",
        "7. Run random-window, grouped-recording, and leave-one-load-out evaluation.",
        "8. Aggregate window probabilities into recording-level predictions.",
        "9. Generate CSV, JSON, text, and image reports.",
        "10. Validate split integrity and committed result artifacts.",
        "",
        "## Reproduce the Leakage-Aware Experiment",
        "",
        "Run these commands from the repository root:",
        "",
        "    python scripts/download_cwru_load_benchmark.py",
        "    python scripts/build_cwru_load_features.py",
        "    python scripts/run_leakage_aware_evaluation.py",
        "    python scripts/generate_leakage_aware_report.py",
        "    python src/validate_leakage_aware_results.py",
        "    pytest",
        "",
        "The automated tests use synthetic signals and temporary MATLAB files, so CI does not need the external CWRU dataset.",
        "",
        "## Main Outputs",
        "",
        "- `results/leakage_aware_evaluation/strategy_comparison.csv`",
        "- `results/leakage_aware_evaluation/fold_metrics.csv`",
        "- `results/leakage_aware_evaluation/window_predictions.csv`",
        "- `results/leakage_aware_evaluation/recording_predictions.csv`",
        "- `results/leakage_aware_evaluation/per_class_metrics.csv`",
        "- `results/leakage_aware_evaluation/analysis_summary.json`",
        "- `docs/leakage_aware_evaluation.md`",
        "",
        "## Testing and Validation",
        "",
        "The test suite covers manifest parsing, MATLAB loading, sample-rate standardization, window selection, feature extraction, split invariants, recording-level aggregation, report generation, tied-result handling, and reusable validation-toolkit integration.",
        "",
        "Run:",
        "",
        "    pytest",
        "",
        "## Limitations",
        "",
        "The perfect benchmark scores do not establish production readiness.",
        "",
        "- All recordings come from one laboratory test rig.",
        "- The benchmark uses only the 0.007-inch fault diameter.",
        "- Only the 6 o'clock outer-race position is included.",
        "- Normal and fault recordings originated from different sampling configurations.",
        "- Resampling cannot remove every acquisition-domain difference.",
        "- No independent machine or external bearing dataset has been tested.",
        "- No sensor drift, calibration, maintenance-cost, or deployment monitoring study is included.",
        "",
        "## Next Experiments",
        "",
        "1. Fault-diameter generalization with an unseen severity.",
        "2. Time-domain versus frequency-domain feature ablation.",
        "3. Comparison with one additional classical model.",
        "4. Controlled noise, amplitude-scaling, and filtering tests.",
        "5. CPU latency, memory, and saved-model-size benchmarking.",
        "6. Comparison with a compact raw-signal model.",
        "",
        "## Reusable Validation Toolkit",
        "",
        "The project integrates [`ml-testing-validation-toolkit`](https://github.com/sakthi-kr/ml-testing-validation-toolkit) for reusable feature-table and model-output checks.",
        "Leakage-specific invariants are implemented directly in this repository because they depend on source-recording and motor-load metadata.",
    ]

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    for path in [ANALYSIS_PATH, STRATEGY_PATH, LOAD_PATH]:
        require(path)

    analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    strategy_rows = load_csv(STRATEGY_PATH)
    load_rows = load_csv(LOAD_PATH)

    expected_dataset = {
        "n_windows": 800,
        "n_source_recordings": 16,
        "n_classes": 4,
        "windows_per_recording": 50,
    }

    observed_dataset = {
        key: analysis.get("dataset", {}).get(key)
        for key in expected_dataset
    }

    if observed_dataset != expected_dataset:
        raise ValueError(
            f"Unexpected dataset summary: {observed_dataset}; expected {expected_dataset}"
        )

    README_PATH.write_text(
        build_project_readme(analysis, strategy_rows, load_rows),
        encoding="utf-8",
    )

    DATA_README_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_README_PATH.write_text(build_data_readme(), encoding="utf-8")

    print("Updated README.md and data/README.md for Phase 1F.")


if __name__ == "__main__":
    main()
