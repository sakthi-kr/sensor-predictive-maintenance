# Validation Strategy

## Purpose

This document describes the validation approach for the sensor predictive-maintenance project.

The first baseline model uses a simple window-level train/test split. This is useful for checking that the full machine-learning pipeline works, but it is not sufficient for a realistic industrial validation.

## Current Baseline Validation

Current setup:

- Dataset: CWRU bearing vibration data
- Files used: 97.mat, 105.mat, 118.mat, 130.mat
- Classes: normal, inner race fault, ball fault, outer race fault
- Signal channel: drive-end vibration signal
- Feature type: time-domain and frequency-domain features
- Model: Random Forest classifier
- Split: random window-level train/test split

## Main Limitation

The current split can cause data leakage.

Each `.mat` file is divided into many overlapping windows. If windows from the same original signal are randomly split into both training and testing sets, the test windows may be very similar to training windows.

This can produce overly optimistic results.

## Why This Still Matters

The first baseline is still useful because it confirms that the full pipeline works:

```text
load data -> segment signal -> extract features -> train model -> evaluate -> predict
```

But the result should not be interpreted as deployment-ready performance.

## Improved Validation Plan

Future validation should become more realistic in stages.

### Stage 1: Window-Level Split

Purpose:

- confirm that the pipeline works
- debug data loading, feature extraction, and model training
- establish a simple baseline

Risk:

- possible leakage between train and test windows

Status:

- completed as first baseline

### Stage 2: File-Level Split

Purpose:

- train on some files and test on different files
- reduce similarity between train and test samples

Requirement:

- download more CWRU files per class

Expected benefit:

- more honest estimate of generalization

### Stage 3: Load-Condition Split

Purpose:

- train on one or more motor load conditions
- test on unseen load conditions

Example:

```text
Train: 0 HP, 1 HP, 2 HP
Test: 3 HP
```

Expected benefit:

- tests robustness to operating-condition changes

### Stage 4: Fault-Size Split

Purpose:

- train on some fault diameters
- test on unseen fault diameters

Example:

```text
Train: 0.007 inch and 0.014 inch faults
Test: 0.021 inch faults
```

Expected benefit:

- tests whether the model learns general fault behaviour, not only one specific fault severity

### Stage 5: Model Comparison

Compare:

- Logistic Regression
- Random Forest
- Support Vector Machine
- XGBoost
- simple 1D CNN

Expected benefit:

- separates feature-engineering performance from model-choice performance

## Validation Metrics

Classification metrics:

- accuracy
- precision
- recall
- F1-score
- confusion matrix

Additional analysis:

- per-class error analysis
- false-positive inspection
- false-negative inspection
- feature importance
- robustness across load conditions
- robustness across fault sizes

## Deployment-Relevant Checks

Before any real industrial use, additional checks would be needed:

- sensor calibration
- unseen machine testing
- noisy-signal testing
- missing-data behaviour
- drift monitoring
- threshold stability
- expert review
- safety and maintenance-process integration

## Summary

The current baseline is a working proof of pipeline. The next goal is not simply higher accuracy, but more realistic validation under unseen files, loads, and fault severities.
