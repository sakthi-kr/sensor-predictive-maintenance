# Dataset Instructions

This project uses the Case Western Reserve University bearing vibration dataset.

The raw dataset is not included in this repository. Download the required `.mat` files from the official CWRU Bearing Data Center.

## Local Folder Structure

Create this local folder structure:

```text
data/
└── raw/
    └── cwru/
        ├── normal/
        ├── inner_race_fault/
        ├── ball_fault/
        └── outer_race_fault/
```

## First Baseline Files

For the first working baseline, use these files:

| Class | File | Local Path |
|---|---|---|
| normal | 97.mat | data/raw/cwru/normal/97.mat |
| inner race fault | 105.mat | data/raw/cwru/inner_race_fault/105.mat |
| ball fault | 118.mat | data/raw/cwru/ball_fault/118.mat |
| outer race fault | 130.mat | data/raw/cwru/outer_race_fault/130.mat |

## Download with Git Bash

Run these commands from the project root:

```bash
mkdir -p data/raw/cwru/normal
mkdir -p data/raw/cwru/inner_race_fault
mkdir -p data/raw/cwru/ball_fault
mkdir -p data/raw/cwru/outer_race_fault

curl -L --fail \
  "https://engineering.case.edu/sites/default/files/97.mat" \
  -o data/raw/cwru/normal/97.mat

curl -L --fail \
  "https://engineering.case.edu/sites/default/files/105.mat" \
  -o data/raw/cwru/inner_race_fault/105.mat

curl -L --fail \
  "https://engineering.case.edu/sites/default/files/118.mat" \
  -o data/raw/cwru/ball_fault/118.mat

curl -L --fail \
  "https://engineering.case.edu/sites/default/files/130.mat" \
  -o data/raw/cwru/outer_race_fault/130.mat
```

## Verify Downloads

```bash
find data/raw/cwru -name "*.mat" -type f
ls -lh data/raw/cwru/normal
ls -lh data/raw/cwru/inner_race_fault
ls -lh data/raw/cwru/ball_fault
ls -lh data/raw/cwru/outer_race_fault
```

## Important

Do not upload raw `.mat` files to GitHub.

The `.gitignore` file excludes `.mat` files so the raw dataset remains local.
