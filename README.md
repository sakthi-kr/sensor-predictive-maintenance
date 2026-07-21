# Sensor Predictive Maintenance

This project develops a machine learning workflow for classifying equipment condition from vibration sensor data.

## Motivation

Industrial equipment produces sensor signals that can be used to detect faults, monitor degradation, and support predictive maintenance. This project focuses on vibration-based bearing fault classification.

## Dataset

Planned dataset: Case Western Reserve University bearing vibration dataset.

The dataset will not be uploaded to this repository. Download instructions will be provided in `data/README.md`.

## Methods

Planned methods:

- Vibration signal loading
- Signal segmentation
- Time-domain feature extraction
- Frequency-domain feature extraction
- Baseline machine learning models
- Model comparison
- Validation across operating conditions
- Error analysis

## Project Structure

```text
data/
notebooks/
src/
tests/
results/
docs/
```

## Current Status

Project skeleton created. Data processing and model development will be added in later phases.

## Planned Results

The final project will include:

- bearing fault classification model
- feature importance analysis
- confusion matrix
- validation metrics
- leakage-aware train/test split discussion
- reproducible prediction script
- basic testing and validation
