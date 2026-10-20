"""Conclusions computed from the results."""

import streamlit as st

from app_pages.common import key_results

r = key_results()
profiles = r["profiles"]
steps_by_day = r["steps_by_day"]

st.title("Conclusions")
st.write("Every number below is computed from the data with the same settings as the model pages.")

# --- Headline numbers ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Users", r["users"])
k2.metric("Days analysed", r["days"])
k3.metric("Days with sleep data", f"{1 - r['missing_sleep_share']:.0%}")
k4.metric("Healthy days", f"{r['healthy_share']:.0%}", help="Share of days with sleep data that meet all three targets")

st.subheader("Key Findings")
profile_lines = "\n".join(
    f"   - **{name}:** ~{row.TotalSteps:,.0f} steps, {row.VeryActiveMinutes:.0f} very active minutes, "
    f"{row.TotalMinutesAsleep / 60:.1f} h of sleep per day"
    for name, row in profiles.iterrows()
)
st.markdown(f"""
1. **Data quality comes first.** {r["missing_sleep_share"]:.0%} of the days have no sleep record. On those days users log
   {r["sedentary_without_sleep"]:,.0f} sedentary minutes on average, against
   {r["sedentary_with_sleep"]:,.0f} on days with a record: the device was most likely
   not worn at night. The models therefore use only the {r["days_with_sleep"]} days with real sleep data.
2. **Most days fall short of 10,000 steps.** Users average {r["avg_steps"]:,.0f} steps a day and
   only {r["share_10k"]:.0%} of the days reach 10,000 steps. {steps_by_day.idxmax()} is the most active day
   ({steps_by_day.max():,.0f} steps on average) and {steps_by_day.idxmin()} the least active ({steps_by_day.min():,.0f}).
3. **Three behaviour profiles.** K-Means finds:
{profile_lines}

   Sleep does not simply follow activity: the **{r["most_sleep"]}** group sleeps the most and the
   **{r["least_sleep"]}** group the least.
4. **Intense activity drives calories burned.** Activity and sleep explain {r["ols_r2"]:.0%} of the variation
   in daily calories (R²). The strongest effect comes from **{r["ols_top_feature"]}** (standardized coefficient
   {r["ols_top_effect"]:.2f}).
5. **Healthy days can be flagged, roughly.** Using only calories and active minutes, the logistic model finds
   {r["recall"]:.0%} of the healthy days of users it has never seen (ROC AUC {r["roc_auc"]:.2f}).
   It is a useful first filter, not a reliable diagnosis on its own.
""")

st.subheader("Limitations")
st.markdown(f"""
- **Small, self-selected sample:** {r["users"]} users ({r["users_with_sleep"]} with sleep data) who volunteered
  through an online survey, over one month in 2016. The results should not be generalised.
- **Correlation, not causation:** the models describe associations in this data, not causes.
- **Device estimates:** Fitbit estimates calories partly from the same activity data, so the regression partly
  recovers the device's own formula.
- **Missing sleep data:** days without a sleep record are excluded from the models, which may favour users who
  wear the device consistently.
""")

st.subheader("Recommendations")
st.markdown(f"""
1. **Improve data collection:** encourage users to wear the device at night; {r["missing_sleep_share"]:.0%} of the
   days have no sleep data.
2. **Targeted activity goals:** the **Low activity** group averages
   {profiles.loc["Low activity", "TotalSteps"]:,.0f} steps a day; gradual step challenges fit this group better
   than a fixed 10,000-step goal.
3. **Sleep reminders:** the **{r["least_sleep"]}** group sleeps only {profiles.loc[r["least_sleep"], "TotalMinutesAsleep"] / 60:.1f} h
   a night on average, below the 7 hours used in the "healthy day" definition.
""")
