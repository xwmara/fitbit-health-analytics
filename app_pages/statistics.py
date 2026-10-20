"""Descriptive statistics and group aggregations."""

import streamlit as st

from fitbit_analytics import analysis, charts
from app_pages.common import load_data

daily_activity, sleep_data, df, df_sleep = load_data()

st.title("Statistics")
st.markdown(
    "Descriptive statistics for the main variables, averages per user, and how activity changes "
    "across the week."
)

# --- LEVEL 1: KEY METRICS ---
st.subheader("Key Indicators (Overall)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Max Steps", f"{df['TotalSteps'].max():,.0f}")
with col2:
    st.metric("Max Calories", f"{df['Calories'].max():,.0f} kcal")
with col3:
    st.metric("Avg Sedentary Time", f"{df['SedentaryMinutes'].mean():,.0f} min")
with col4:
    st.metric("Max Sleep", f"{df['TotalMinutesAsleep'].max() / 60:.1f} h")

st.divider()

# --- LEVEL 2: TABLES (grouping and aggregation) ---
main_vars = ["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes"]
number = st.column_config.NumberColumn(format="localized")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Descriptive Statistics")
    st.caption("Numeric summary of the main variables, over all 940 days.")
    summary = charts.readable(df[main_vars].describe()).round(0)
    summary.index = ["Days", "Mean", "Std. dev.", "Min", "25%", "Median", "75%", "Max"]
    st.dataframe(summary, width="stretch", column_config={c: number for c in summary.columns})

with col_b:
    st.subheader("Average per User")
    st.caption("Daily averages for each of the 33 users (scroll for more).")
    grouped_user = charts.readable(df.groupby("Id")[main_vars].mean()).round(0)
    grouped_user.index = grouped_user.index.astype(str)
    st.dataframe(grouped_user, width="stretch", height=318, column_config={c: number for c in grouped_user.columns})

st.divider()

# --- LEVEL 3: STEPS BY WEEKDAY ---
st.subheader("Average Steps by Day of the Week")

# Mean steps per weekday, ordered Monday to Sunday
grouped_day = analysis.average_steps_by_weekday(daily_activity)

st.bar_chart(data=grouped_day, x="DayOfWeek", y="TotalSteps", color=charts.PRIMARY,
             x_label="", y_label="Average steps", sort=False)

with st.expander("Show the data behind the chart"):
    st.dataframe(grouped_day.rename(columns={"DayOfWeek": "Day", "TotalSteps": "Average steps"}).round(0),
                 hide_index=True, column_config={"Average steps": number})
