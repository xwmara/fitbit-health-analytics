"""Data and settings shared by several pages."""

import streamlit as st

from fitbit_analytics import analysis, data

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


@st.cache_data
def key_results():
    """Compute the headline results shown on the Overview and Conclusions pages.

    Uses the same settings as the model pages, so the numbers always match.
    """
    daily, _, df, df_sleep = load_data()

    comparison = analysis.compare_days_by_sleep_record(df)
    steps_by_day = analysis.average_steps_by_weekday(daily).set_index("DayOfWeek")["TotalSteps"]

    clusters = df_sleep[CLUSTER_FEATURES].copy()
    clusters["Cluster"] = analysis.kmeans_clusters(clusters[CLUSTER_FEATURES], n_clusters=3)
    profiles = clusters.groupby("Cluster")[CLUSTER_FEATURES].mean()
    profiles.index = profiles.index.map(analysis.name_clusters(profiles))
    profiles = profiles.loc[analysis.CLUSTER_NAMES]

    healthy = analysis.healthy_lifestyle_label(df_sleep)
    classifier = analysis.train_logistic_regression(df_sleep[LOGISTIC_FEATURES], healthy, groups=df_sleep["Id"])

    ols_model, _ = analysis.fit_ols(df_sleep[OLS_FEATURES], df_sleep["Calories"])
    ols_top = analysis.standardized_coefficients(ols_model, df_sleep[OLS_FEATURES], df_sleep["Calories"]).iloc[0]

    return {
        "users": df["Id"].nunique(),
        "users_with_sleep": df_sleep["Id"].nunique(),
        "days": len(df),
        "days_with_sleep": len(df_sleep),
        "missing_sleep_share": df["TotalMinutesAsleep"].isna().mean(),
        "sedentary_without_sleep": comparison.loc["Without sleep record", "SedentaryMinutes"],
        "sedentary_with_sleep": comparison.loc["With sleep record", "SedentaryMinutes"],
        "avg_steps": daily["TotalSteps"].mean(),
        "share_10k": (daily["TotalSteps"] >= 10000).mean(),
        "steps_by_day": steps_by_day,
        "profiles": profiles,
        "most_sleep": profiles["TotalMinutesAsleep"].idxmax(),
        "least_sleep": profiles["TotalMinutesAsleep"].idxmin(),
        "healthy_share": healthy.mean(),
        "recall": classifier.recall,
        "roc_auc": classifier.roc_auc,
        "ols_r2": ols_model.rsquared,
        "ols_top_feature": ols_top.Feature,
        "ols_top_effect": ols_top.StdCoefficient,
    }


def sleep_subset_note(df, df_sleep):
    """Caption explaining that a model uses only the days with a sleep record."""
    return (
        f"This model uses only the **{len(df_sleep)} days with a sleep record** "
        f"({df_sleep['Id'].nunique()} of {df['Id'].nunique()} users). Sleep values are not imputed, "
        "because days without a record count the night as sedentary time."
    )
