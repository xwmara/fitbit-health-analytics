"""Data and settings shared by several pages."""

import streamlit as st

from fitbit_analytics import data

# Model features, shared by the model pages and the conclusions
CLUSTER_FEATURES = ["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes", "TotalMinutesAsleep"]
# Only variables that are NOT used to define the "healthy day" label (avoids data leakage)
LOGISTIC_FEATURES = ["Calories", "VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes"]
OLS_FEATURES = ["TotalSteps", "VeryActiveMinutes", "SedentaryMinutes", "TotalMinutesAsleep"]


@st.cache_data
def load_data():
    """Return (daily activity, sleep, merged table, merged days with a sleep record).

    Cached, so the CSV files are read only once; every call returns a fresh copy.
    Models that use sleep work only on days with a real sleep record (see the Missing values page).
    """
    daily, sleep, merged = data.load_data()
    return daily, sleep, merged, data.days_with_sleep(merged)


def sleep_subset_note(df, df_sleep):
    """Caption explaining that a model uses only the days with a sleep record."""
    return (
        f"This model uses only the **{len(df_sleep)} days with a sleep record** "
        f"({df_sleep['Id'].nunique()} of {df['Id'].nunique()} users). Sleep values are not imputed, "
        "because days without a record count the night as sedentary time."
    )
