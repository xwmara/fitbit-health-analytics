"""Conclusions computed from the results."""

import streamlit as st

from fitbit_analytics import analysis
from app_pages.common import load_data, CLUSTER_FEATURES, LOGISTIC_FEATURES, OLS_FEATURES

daily_activity, sleep_data, df, df_sleep = load_data()

st.title("Conclusions")
st.write("Every number below is computed from the data with the same settings as the sections above.")

# --- Recompute the key results ---
comparison = analysis.compare_days_by_sleep_record(df)
missing_share = df["TotalMinutesAsleep"].isna().mean()
steps_by_day = analysis.average_steps_by_weekday(daily_activity).set_index("DayOfWeek")["TotalSteps"]
share_10k = (daily_activity["TotalSteps"] >= 10000).mean()

clusters = df_sleep[CLUSTER_FEATURES].copy()
clusters["Cluster"] = analysis.kmeans_clusters(clusters[CLUSTER_FEATURES], n_clusters=3)
profiles = clusters.groupby("Cluster")[CLUSTER_FEATURES].mean()
profiles.index = profiles.index.map(analysis.name_clusters(profiles))
profiles = profiles.loc[analysis.CLUSTER_NAMES]
most_sleep, least_sleep = profiles["TotalMinutesAsleep"].idxmax(), profiles["TotalMinutesAsleep"].idxmin()

healthy = analysis.healthy_lifestyle_label(df_sleep)
log_result = analysis.train_logistic_regression(df_sleep[LOGISTIC_FEATURES], healthy, groups=df_sleep["Id"])

ols_model, _ = analysis.fit_ols(df_sleep[OLS_FEATURES], df_sleep["Calories"])
ols_top = analysis.standardized_coefficients(ols_model, df_sleep[OLS_FEATURES], df_sleep["Calories"]).iloc[0]

# --- Headline numbers ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Users", df["Id"].nunique())
k2.metric("Days analysed", len(df))
k3.metric("Days with sleep data", f"{1 - missing_share:.0%}")
k4.metric("Healthy days", f"{healthy.mean():.0%}", help="Share of days with sleep data that meet all three targets")

st.subheader("Key Findings")
profile_lines = "\n".join(
    f"   - **{name}:** ~{row.TotalSteps:,.0f} steps, {row.VeryActiveMinutes:.0f} very active minutes, "
    f"{row.TotalMinutesAsleep / 60:.1f} h of sleep per day"
    for name, row in profiles.iterrows()
)
st.markdown(f"""
1. **Data quality comes first.** {missing_share:.0%} of the days have no sleep record. On those days users log
   {comparison.loc["Without sleep record", "SedentaryMinutes"]:,.0f} sedentary minutes on average, against
   {comparison.loc["With sleep record", "SedentaryMinutes"]:,.0f} on days with a record: the device was most likely
   not worn at night. The models therefore use only the {len(df_sleep)} days with real sleep data.
2. **Most days fall short of 10,000 steps.** Users average {daily_activity["TotalSteps"].mean():,.0f} steps a day and
   only {share_10k:.0%} of the days reach 10,000 steps. {steps_by_day.idxmax()} is the most active day
   ({steps_by_day.max():,.0f} steps on average) and {steps_by_day.idxmin()} the least active ({steps_by_day.min():,.0f}).
3. **Three behaviour profiles.** K-Means finds:
{profile_lines}

   Sleep does not simply follow activity: the **{most_sleep}** group sleeps the most and the
   **{least_sleep}** group the least.
4. **Intense activity drives calories burned.** Activity and sleep explain {ols_model.rsquared:.0%} of the variation
   in daily calories (R²). The strongest effect comes from **{ols_top.Feature}** (standardized coefficient
   {ols_top.StdCoefficient:.2f}).
5. **Healthy days can be flagged, roughly.** Using only calories and active minutes, the logistic model finds
   {log_result.recall:.0%} of the healthy days of users it has never seen (ROC AUC {log_result.roc_auc:.2f}).
   It is a useful first filter, not a reliable diagnosis on its own.
""")

st.subheader("Limitations")
st.markdown(f"""
- **Small, self-selected sample:** {df["Id"].nunique()} users ({df_sleep["Id"].nunique()} with sleep data) who volunteered
  through an online survey, over one month in 2016. The results should not be generalised.
- **Correlation, not causation:** the models describe associations in this data, not causes.
- **Device estimates:** Fitbit estimates calories partly from the same activity data, so the regression partly
  recovers the device's own formula.
- **Missing sleep data:** days without a sleep record are excluded from the models, which may favour users who
  wear the device consistently.
""")

st.subheader("Recommendations")
st.markdown(f"""
1. **Improve data collection:** encourage users to wear the device at night; {missing_share:.0%} of the days have
   no sleep data.
2. **Targeted activity goals:** the **Low activity** group averages
   {profiles.loc["Low activity", "TotalSteps"]:,.0f} steps a day; gradual step challenges fit this group better
   than a fixed 10,000-step goal.
3. **Sleep reminders:** the **{least_sleep}** group sleeps only {profiles.loc[least_sleep, "TotalMinutesAsleep"] / 60:.1f} h
   a night on average, below the 7 hours used in the "healthy day" definition.
""")
