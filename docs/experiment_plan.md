# Experiment Plan

## Goal

The goal is to build a reproducible machine-learning workflow for classifying bearing condition from vibration sensor data and to evaluate how well the model generalizes beyond the first simple baseline.

## Current Experiment: Baseline Random Forest

### Dataset

Initial four-file dataset:

| Class | File |
|---|---|
| normal | 97.mat |
| inner race fault | 105.mat |
| ball fault | 118.mat |
| outer race fault | 130.mat |

### Features

Time-domain features:

- mean
- standard deviation
- RMS
- minimum
- maximum
- peak-to-peak value
- skewness
- kurtosis
- crest factor
- shape factor

Frequency-domain features:

- dominant frequency
- spectral centroid
- spectral bandwidth
- spectral energy

### Model

Random Forest classifier.

### Purpose

This experiment tests whether the initial pipeline works end-to-end.

### Current Result

The model gives perfect classification on the first window-level split.

### Interpretation

The result is promising for pipeline development, but it is not enough to claim robust industrial performance because the current split may contain leakage between similar windows from the same signal.

---

## Experiment 1: Baseline Model Comparison

### Question

How do simple classical machine-learning models compare on the same extracted features?

### Models

- Logistic Regression
- Random Forest
- Support Vector Machine
- k-Nearest Neighbors
- Gradient Boosting or XGBoost

### Metrics

- accuracy
- precision
- recall
- F1-score
- confusion matrix

### Expected Outcome

Random Forest or gradient boosting may perform strongly because the feature table is tabular and nonlinear.

---

## Experiment 2: Feature Ablation

### Question

Which feature groups matter most?

### Comparisons

- time-domain features only
- frequency-domain features only
- combined time + frequency features

### Purpose

This checks whether the model depends mainly on amplitude-based features, frequency-based features, or both.

### Expected Outcome

Bearing faults should affect both amplitude statistics and frequency content, so the combined feature set should perform best.

---

## Experiment 3: Leakage-Aware File-Level Split

### Question

Does the model still perform well when tested on files not seen during training?

### Requirement

Download additional CWRU files for each class.

### Method

Train on one set of files and test on different files.

### Purpose

This reduces the chance that highly similar windows from the same source signal appear in both train and test sets.

---

## Experiment 4: Load-Condition Generalization

### Question

Can the model generalize to unseen motor load conditions?

### Method

Train on selected load conditions and test on a different load condition.

Example:

```text
Train: 0 HP, 1 HP, 2 HP
Test: 3 HP
```

### Purpose

This is closer to real predictive-maintenance use, where machine operating conditions can change.

---

## Experiment 5: Raw-Signal 1D CNN

### Question

Can a simple neural network learn directly from raw vibration windows?

### Method

Use raw segmented vibration windows as input to a 1D CNN.

### Comparison

Compare the 1D CNN with the feature-based Random Forest baseline.

### Purpose

This tests whether learned features improve performance over manually engineered vibration features.

---

## Final Deliverables

The final project should include:

- clean data loading
- feature extraction
- baseline model comparison
- confusion matrices
- feature importance
- leakage-aware validation
- documented limitations
- tests for core pipeline components
- model card
- reproducible scripts
