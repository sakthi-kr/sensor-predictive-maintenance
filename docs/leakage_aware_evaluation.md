# Leakage-Aware Evaluation

## Purpose

The original development baseline randomly divided signal windows into training and test sets. Because many windows originate from the same source recording, this design can place windows from one recording in both sets.

This experiment compares three evaluation strategies:

1. **Random window split:** optimistic development baseline.
2. **Grouped recording cross-validation:** keeps all windows from each source recording together.
3. **Leave-one-load-out:** trains on three motor loads and tests on one unseen motor load.

## Benchmark Dataset

| Property | Value |
| --- | --- |
| Source recordings | 16 |
| Windows | 800 |
| Classes | 4 |
| Motor loads | 0 HP, 1 HP, 2 HP, 3 HP |
| Windows per recording | 50 |
| Effective sampling rate | 12 kHz |
| Model features | 14 |

## Evaluation Results

| Evaluation strategy | Window accuracy | Balanced accuracy | Window macro F1 | Recording accuracy | Maximum recording overlap |
| --- | --- | --- | --- | --- | --- |
| Random window split | 1.000 ± 0.000 | 1.000 | 1.000 ± 0.000 | 1.000 | 16 |
| Grouped recording CV | 1.000 ± 0.000 | 1.000 | 1.000 ± 0.000 | 1.000 | 0 |
| Leave-one-load-out | 1.000 ± 0.000 | 1.000 | 1.000 ± 0.000 | 1.000 | 0 |

Values are means across four folds. The uncertainty is the standard deviation across folds.

![Evaluation strategy comparison](../results/leakage_aware_evaluation/strategy_comparison.png)

## Generalization Gaps

| Comparison | Accuracy difference |
| --- | --- |
| Random window minus grouped recording | 0.000 |
| Random window minus leave-one-load-out | 0.000 |

## Leave-One-Load-Out Results

| Held-out load | Window accuracy | Balanced accuracy | Macro F1 | Recording accuracy |
| --- | --- | --- | --- | --- |
| 0 HP | 1.000 | 1.000 | 1.000 | 1.000 |
| 1 HP | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 HP | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 HP | 1.000 | 1.000 | 1.000 | 1.000 |

![Leave-one-load-out results](../results/leakage_aware_evaluation/leave_one_load_out_by_load.png)

All held-out motor loads tied at accuracy **1.000** and macro F1 **1.000**.

## Per-Class Results

| Evaluation strategy | Class | Precision | Recall | F1-score | Pooled test support |
| --- | --- | --- | --- | --- | --- |
| Random window split | ball_fault | 1.000 | 1.000 | 1.000 | 200 |
| Random window split | inner_race_fault | 1.000 | 1.000 | 1.000 | 200 |
| Random window split | normal | 1.000 | 1.000 | 1.000 | 200 |
| Random window split | outer_race_fault | 1.000 | 1.000 | 1.000 | 200 |
| Grouped recording CV | ball_fault | 1.000 | 1.000 | 1.000 | 200 |
| Grouped recording CV | inner_race_fault | 1.000 | 1.000 | 1.000 | 200 |
| Grouped recording CV | normal | 1.000 | 1.000 | 1.000 | 200 |
| Grouped recording CV | outer_race_fault | 1.000 | 1.000 | 1.000 | 200 |
| Leave-one-load-out | ball_fault | 1.000 | 1.000 | 1.000 | 200 |
| Leave-one-load-out | inner_race_fault | 1.000 | 1.000 | 1.000 | 200 |
| Leave-one-load-out | normal | 1.000 | 1.000 | 1.000 | 200 |
| Leave-one-load-out | outer_race_fault | 1.000 | 1.000 | 1.000 | 200 |

## Recording-Level Results

| Evaluation strategy | Recording predictions | Correct | Incorrect | Recording accuracy | Recordings also seen in training |
| --- | --- | --- | --- | --- | --- |
| Random window split | 64 | 64 | 0 | 1.000 | 64 |
| Grouped recording CV | 16 | 16 | 0 | 1.000 | 0 |
| Leave-one-load-out | 16 | 16 | 0 | 1.000 | 0 |

## Recording-Level Error Analysis

No recording-level misclassifications were observed.

## Interpretation

- Random-window and grouped scores are similar, but only the grouped result prevents source-recording overlap.
- Performance remains comparatively stable when one motor load is completely held out.
- All held-out motor loads tied, with accuracy and macro F1 of 1.000.
- Grouped recording and leave-one-load-out evaluations contain zero source-recording overlap.

## Leakage Verification

| Evaluation strategy | Maximum overlapping source recordings |
| --- | --- |
| Random window split | 16 |
| Grouped recording CV | 0 |
| Leave-one-load-out | 0 |

The grouped recording and leave-one-load-out evaluations pass the leakage check only when this overlap is zero.

## Important Dataset Limitation

Normal recordings were originally sampled at 48 kHz and resampled to 12 kHz, while the selected fault recordings were collected at 12 kHz. Results may still reflect acquisition-domain differences in addition to bearing condition.

Resampling standardizes the effective sample rate used for feature extraction, but it cannot remove every difference created by the original acquisition setup. These results are therefore not proof of generalization to independent industrial machinery.

## Generated Outputs

- `results/leakage_aware_evaluation/summary.json`
- `results/leakage_aware_evaluation/fold_metrics.csv`
- `results/leakage_aware_evaluation/strategy_comparison.csv`
- `results/leakage_aware_evaluation/per_class_metrics.csv`
- `results/leakage_aware_evaluation/load_generalization_metrics.csv`
- `results/leakage_aware_evaluation/recording_predictions.csv`
- `results/leakage_aware_evaluation/recording_error_analysis.csv`
- `results/leakage_aware_evaluation/window_predictions.csv`
- classification reports
- confusion matrices
- comparison plots

## Reproduction

```bash
python scripts/download_cwru_load_benchmark.py
python scripts/build_cwru_load_features.py
python scripts/run_leakage_aware_evaluation.py
python scripts/generate_leakage_aware_report.py
```
