# Model Card

## Model Name

Baseline Random Forest Bearing Fault Classifier

## Problem

Classify bearing condition from vibration sensor data.

## Dataset

Case Western Reserve University bearing vibration dataset.

First baseline files:

- 97.mat: normal
- 105.mat: inner race fault
- 118.mat: ball fault
- 130.mat: outer race fault

## Intended Use

This model is intended as an educational and portfolio project for sensor-based fault detection and predictive maintenance.

It demonstrates:

- vibration signal loading
- signal windowing
- feature extraction
- machine learning classification
- model evaluation
- basic testing and validation

## Not Intended For

This model is not intended for real industrial maintenance decisions.

It has not been validated on:

- unseen machines
- multiple operating environments
- different sensor placements
- noisy production data
- long-term degradation patterns

## Model Type

Random Forest classifier trained on manually extracted time-domain and frequency-domain vibration features.

## Input

A CWRU `.mat` vibration file.

The pipeline extracts the drive-end vibration signal, splits it into fixed-length windows, and converts each window into numerical features.

## Output

The model outputs:

- predicted bearing condition for each window
- aggregated file-level bearing condition
- prediction confidence based on mean class probability

Possible classes:

- normal
- inner_race_fault
- ball_fault
- outer_race_fault

## Evaluation Metrics

Current evaluation includes:

- accuracy
- precision
- recall
- F1-score
- confusion matrix
- feature importance

## Main Results

The first baseline produced perfect classification on the initial four-file dataset.

This result should not be overinterpreted because the current split is window-level random splitting.

## Known Limitations

The main limitation is possible data leakage.

Windows from the same original signal may appear in both training and testing. This can produce over-optimistic results.

Other limitations:

- small initial dataset
- only four `.mat` files used
- no testing across different load conditions yet
- no testing across unseen fault sizes yet
- no deep-learning model comparison yet
- no real deployment monitoring

## Possible Failure Cases

The model may fail when:

- data comes from a different machine
- sensor position changes
- operating speed/load changes
- fault type is not included in training
- vibration signal is noisy or incomplete
- new fault severity differs from the training data

## Future Improvements

- use more CWRU files
- create leakage-aware splits
- evaluate across motor loads
- evaluate across fault diameters
- compare Random Forest with SVM, XGBoost, and 1D CNN
- add FFT plots and signal visualizations
- add model monitoring and drift reports
- add a simple user-facing inference demo
