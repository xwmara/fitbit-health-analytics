"""Load the Fitbit CSV files and prepare them for analysis."""

from pathlib import Path

import numpy as np
import pandas as pd

# The data folder sits next to this package, so the app works from any working directory.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SLEEP_COLUMNS = ["TotalSleepRecords", "TotalMinutesAsleep", "TotalTimeInBed"]


def load_raw_data(data_dir=DATA_DIR):
    """Read the daily activity and sleep CSV files, drop exact duplicate rows and parse the dates.

    The sleep file contains 3 rows that are exact copies of other rows; keeping them would
    duplicate those days in the merged table.
    """
    data_dir = Path(data_dir)
    daily = pd.read_csv(data_dir / "dailyActivity_merged.csv").drop_duplicates(ignore_index=True)
    sleep = pd.read_csv(data_dir / "sleepDay_merged.csv").drop_duplicates(ignore_index=True)

    daily["ActivityDate"] = pd.to_datetime(daily["ActivityDate"])
    sleep["SleepDay"] = pd.to_datetime(sleep["SleepDay"])
    return daily, sleep


def merge_activity_sleep(daily, sleep):
    """Left-join sleep data onto daily activity by user Id and calendar date.

    Adds the date-only join keys to both inputs (`ActivityDateOnly` on daily,
    `ActivityDate` on sleep). Every daily row is kept; days without a sleep
    record get NaN in the sleep columns.
    """
    sleep["ActivityDate"] = sleep["SleepDay"].dt.date
    daily["ActivityDateOnly"] = daily["ActivityDate"].dt.date

    return pd.merge(
        daily,
        sleep[["Id", "ActivityDate", *SLEEP_COLUMNS]],
        left_on=["Id", "ActivityDateOnly"],
        right_on=["Id", "ActivityDate"],
        how="left",
    )


def load_data(data_dir=DATA_DIR):
    """Return the daily activity table, the sleep table and the merged table."""
    daily, sleep = load_raw_data(data_dir)
    merged = merge_activity_sleep(daily, sleep)
    return daily, sleep, merged


def fill_missing_with_median(df, columns=None):
    """Return a copy of `df` where NaNs in `columns` are replaced by each column's median.

    If `columns` is None, all numeric columns are filled.
    """
    filled = df.copy()
    if columns is None:
        columns = filled.select_dtypes(include=[np.number]).columns
    for col in columns:
        filled[col] = filled[col].fillna(filled[col].median())
    return filled
