# Fitbit Health Analytics

An interactive Streamlit app that explores daily activity and sleep data from Fitbit users and applies
classic data analysis and machine learning techniques to describe what a healthy lifestyle looks like.

> **Status:** work in progress. The app started as a university project and is being turned into a
> portfolio project. Next steps: a review of the analysis methods and a public deployment.

## What the app does

The app is organised in sections, selectable from the sidebar:

| Section | What it shows |
| --- | --- |
| Data exploration | Structure of the two datasets and the merged table |
| Missing values | Median imputation, and why missing sleep is not random (the device is often not worn at night) |
| Outliers | IQR-based outlier detection with a boxplot for a chosen variable |
| Statistics and aggregations | Descriptive statistics, per-user averages, average steps by weekday |
| Encoding and scaling | Lifestyle categories from step counts, label encoding, standardisation |
| K-Means clustering | Segmentation into 3 activity profiles, steps vs. calories scatter plot |
| Logistic regression | Classifier for a "healthy day" label, without data leakage, tested on unseen users and compared with a majority-class baseline (recall, F1, ROC AUC) |
| Multiple regression (OLS) | How steps, activity and sleep relate to calories burned |
| Geographic view | World map built with GeoPandas |
| Conclusions | Key findings computed live from the data, limitations and recommendations |

## Tech stack

Python, Streamlit, pandas, NumPy, scikit-learn, statsmodels, Matplotlib, seaborn, GeoPandas.

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
a small, self-selected sample and the results should not be generalised to the wider population.

56% of the days have no sleep record, and on those days the night shows up as sedentary time. The models
that use sleep (K-Means, logistic regression, OLS) are therefore fitted only on the 410 days with a real
sleep record (24 users), instead of imputing the missing values.

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

The app opens at <http://localhost:8501>. Run it from the repository root, because the data files are
loaded with relative paths. The geographic section downloads a GeoJSON file at runtime and needs an
internet connection.

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover data loading and merging, the analysis functions, and a smoke test that renders every
section of the app (except the map, which needs an internet connection).

## Project structure

```
fitbit-health-analytics/
├── app.py                 # Streamlit app (user interface only)
├── fitbit_analytics/
│   ├── data.py            # Loading, merging and cleaning the CSV files
│   └── analysis.py        # Outliers, encoding, scaling, clustering and regression models
├── data/                  # Fitbit CSV files
├── tests/                 # pytest test suite
├── requirements.txt       # Pinned Python dependencies
├── requirements-dev.txt   # Extra tools for development (pytest)
├── pyproject.toml         # pytest configuration
└── README.md
```

## Background

Originally built as a university project for the Software Packages course at the Bucharest University
of Economic Studies (ASE), Faculty of Cybernetics, Statistics and Economic Informatics.
