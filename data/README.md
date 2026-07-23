# CWRU Dataset Instructions

This project uses the Case Western Reserve University bearing vibration dataset.
The raw MATLAB files are external data and are not committed to this repository.
The versioned manifest stores the source filename, class, load, approximate speed, original sample rate, local path, and official download URL.

## Current 16-Recording Benchmark

| Condition | 0 HP | 1 HP | 2 HP | 3 HP |
| --- | ---: | ---: | ---: | ---: |
| Normal | 97.mat | 98.mat | 99.mat | 100.mat |
| Inner-race fault, 0.007 in | 105.mat | 106.mat | 107.mat | 108.mat |
| Ball fault, 0.007 in | 118.mat | 119.mat | 120.mat | 121.mat |
| Outer-race fault, 0.007 in, 6 o'clock | 130.mat | 131.mat | 132.mat | 133.mat |

Manifest:

    data/manifests/cwru_load_benchmark.csv

## Download and Verify

Run from the repository root:

    python scripts/download_cwru_load_benchmark.py

Verify already-downloaded files without contacting the server:

    python scripts/download_cwru_load_benchmark.py --verify-only

The downloader creates the class folders under `data/raw/cwru/`, checks that all 16 files are readable MATLAB files, finds the drive-end signal, and verifies balance across classes and loads.

## Sampling-Rate Standardization

The selected normal recordings were originally sampled at 48 kHz.
The selected fault recordings were collected at 12 kHz.
Normal signals are resampled to 12 kHz with polyphase anti-alias filtering before feature extraction.
This standardizes the effective feature-extraction rate, but it cannot guarantee that every acquisition-domain difference has been removed.

## Local Data Policy

Do not commit raw MATLAB files.
The repository ignores `data/raw/` and `*.mat`, while the small manifest remains tracked for reproducibility.
