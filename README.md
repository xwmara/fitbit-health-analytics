# Fitbit Health Analytics

[![Tests](https://github.com/xwmara/fitbit-health-analytics/actions/workflows/tests.yml/badge.svg)](https://github.com/xwmara/fitbit-health-analytics/actions/workflows/tests.yml)

An interactive Streamlit app that explores daily activity and sleep data from Fitbit users and applies
classic data analysis and machine learning techniques to describe what a healthy lifestyle looks like.

## Key findings

- **Missing sleep data is not random.** 56% of the days have no sleep record. On those days users log about
  1,207 sedentary minutes (20 hours) against 712 on days with a record: the device was simply not worn at night.
  9 of the 33 users never recorded a single night.
- **Most days fall short of 10,000 steps.** Users average about 7,600 steps a day, and only 32% of the days
  reach 10,000. Saturday is the most active day and Sunday the least.
- **Three behaviour profiles (K-Means):** *Low activity* (~5,900 steps, 8.5 h of sleep), *Moderate activity*
  (~9,000 steps, 5.7 h of sleep) and *Very active* (~12,400 steps, 7.1 h of sleep). Sleep does not simply
  follow activity: the least active group sleeps the most.
- **Intense activity drives calories burned.** Activity and sleep explain 42% of the variation in daily
  calories (OLS, R² = 0.42); very active minutes have the largest standardized effect (0.53).
- **Healthy days can be flagged, roughly.** Using only calories and active minutes, a logistic regression finds
  83% of the healthy days of users it has never seen (ROC AUC 0.80).

## Methodology notes

Decisions made while reviewing the original university version of the analysis:

- **No data leakage.** The "healthy day" label is defined from steps, sedentary minutes and sleep, so those
  variables (and distance, which is almost identical to steps) are excluded from the classifier's features.
  The original version used them, which inflated accuracy.
- **Test on unseen users.** Train and test sets are split by user (`GroupShuffleSplit`), so the same person's
  days never appear on both sides.
- **Honest metrics.** Healthy days are the minority class, so accuracy is reported next to a majority-class
  baseline, together with recall, F1 and ROC AUC; the classifier uses balanced class weights.
- **No imputation for modelling.** Because missing sleep reflects not wearing the device, the models that use
  sleep are fitted only on the 410 days with a real sleep record (24 users). Median imputation is shown in the
  app as a technique only.
- **Comparable coefficients.** OLS effects are compared with standardized coefficients, since raw coefficients
  are in different units (per step vs. per minute).
- **Data-driven labels.** K-Means numbers its clusters arbitrarily, so cluster names are derived from each
  cluster's average steps, and the conclusions in the app are computed from the results instead of written by hand.

## What the app does

The app is organised in sections, selectable from the sidebar:

| Section | What it shows |
| --- | --- |
| Data exploration | Structure of the two datasets and the merged table |
| Missing values | Median imputation, and why missing sleep is not random (the device is often not worn at night) |
| Outliers | IQR-based outlier detection with a boxplot for a chosen variable |
| Statistics and aggregations | Descriptive statistics, per-user averages, average steps by weekday |
| Encoding and scaling | Lifestyle categories from step counts, label encoding, standardisation |
| K-Means clustering | Segmentation into 3 behaviour profiles (activity and sleep), steps vs. calories scatter plot |
| Logistic regression | Classifier for a "healthy day" label, without data leakage, tested on unseen users and compared with a majority-class baseline (recall, F1, ROC AUC) |
| Multiple regression (OLS) | How steps, activity and sleep relate to calories burned |
| Data coverage | Heatmap of which days each user wore the device, by day and by night |
| Conclusions | Key findings computed live from the data, limitations and recommendations |

## Tech stack

Python, Streamlit, pandas, NumPy, scikit-learn, statsmodels, Matplotlib, seaborn, pytest.

## Dataset

The data comes from the public **Crowd-sourced Fitbit datasets 03.12.2016–05.12.2016**
(Furberg, Brinton, Keating & Ortiz, 2016), published on Zenodo under the
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license:
<https://doi.org/10.5281/zenodo.53894>. It is also widely known as the
[FitBit Fitness Tracker Data](https://www.kaggle.com/datasets/arashnic/fitbit) dataset on Kaggle.

This project uses two of its files, stored in [`data/`](data/):

- `dailyActivity_merged.csv`: 940 daily records from 33 users (steps, distance, active and sedentary minutes, calories)
- `sleepDay_merged.csv`: 413 daily sleep records from 24 users (minutes asleep, time in bed);
  3 of them are exact duplicates and are removed when the data is loaded

The data covers one month (April–May 2016) from users who volunteered through an online survey, so it is
a small, self-selected sample and the results should not be generalised to the wider population. The
models describe associations, not causes, and Fitbit's calorie figures are themselves estimated from
activity data.

## Run locally

Requirements: Python 3.11 or newer.

```bash
git clone https://github.com/xwmara/fitbit-health-analytics.git
cd fitbit-health-analytics

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. It runs fully offline: the data files ship with the repository.

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover data loading and merging, the analysis functions, and a smoke test that renders every
section of the app. They also run automatically on GitHub Actions for every push to `main`.

## Project structure

```
fitbit-health-analytics/
├── app.py                 # Streamlit app (user interface only)
├── .streamlit/config.toml # Streamlit theme
├── .github/workflows/     # GitHub Actions: run the tests on every push
├── fitbit_analytics/
│   ├── data.py            # Loading, merging and cleaning the CSV files
│   └── analysis.py        # Outliers, encoding, scaling, clustering and regression models
├── data/                  # Fitbit CSV files
├── tests/                 # pytest test suite
├── requirements.txt       # Pinned Python dependencies
├── requirements-dev.txt   # Extra tools for development (pytest)
├── pyproject.toml         # pytest configuration
├── LICENSE                # MIT License (code only)
└── README.md
```

## License

The code is released under the [MIT License](LICENSE). The Fitbit data in [`data/`](data/) is not covered by
it: it belongs to its authors and is redistributed under its original
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license (see [Dataset](#dataset)).

## Background

Originally built as a university project for the Software Packages course at the Bucharest University
of Economic Studies (ASE), Faculty of Cybernetics, Statistics and Economic Informatics.
