# Sensor Pipeline Validation

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
| Overall status | PASS |
| Total checks | 13 |
| Passed checks | 13 |
| Failed checks | 0 |
| Feature-table rows | 400 |
| Test samples | 100 |

## Current Model Metrics

| Metric | Result |
|---|---:|
| Accuracy | 1.0000 |
| Macro F1-score | 1.0000 |
| Weighted F1-score | 1.0000 |

These thresholds are pipeline-regression checks. They are not production-readiness criteria.

## Individual Checks

| Check | Status | Message |
|---|---|---|
| `required_columns` | PASS | All 19 required columns are present. |
| `missing_values` | PASS | Missing-value fractions are within the allowed limit of 0.000. |
| `infinite_values` | PASS | No infinite values were found. |
| `duplicate_rows` | PASS | Duplicate count 0 is within the allowed limit of 0. |
| `allowed_values:label` | PASS | Column 'label' contains only allowed values. |
| `numeric_ranges` | PASS | All 10 configured numeric range checks passed. |
| `class_balance:label` | PASS | Target 'label' satisfies the configured class-balance requirements. |
| `prediction_lengths` | PASS | All model outputs contain 100 sample(s). |
| `prediction_labels` | PASS | All labels are within the configured allowed set. |
| `probability_matrix` | PASS | Probability matrix with shape (100, 4) is valid. |
| `score_range` | PASS | All scores are within the configured range. |
| `metric_thresholds` | PASS | All 3 configured metric threshold checks passed. |
| `confusion_matrix_consistency` | PASS | The supplied confusion matrix matches the predictions. |

## Generated Outputs

```text
results/sensor_validation_report.json
results/sensor_validation_checks.csv
results/sensor_validation_predictions.csv
```

The JSON report contains structured metadata and detailed check results. The CSV files support manual inspection and downstream analysis.

## Known Validation Limitation

The current baseline uses a random window-level split. Similar windows from the same original signal can occur in both training and test sets, so performance may be overestimated.

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
