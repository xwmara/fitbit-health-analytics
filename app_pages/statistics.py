"""Descriptive statistics and group aggregations."""

import streamlit as st

from fitbit_analytics import analysis, charts
from app_pages.common import load_data

daily_activity, sleep_data, df, df_sleep = load_data()

st.title("Descriptive Statistics and Group Analysis")

# --- LEVEL 1: KEY METRICS ---
st.subheader("Key Indicators (Overall)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Max Steps", f"{int(df['TotalSteps'].max())}")
with col2:
    st.metric("Max Calories", f"{int(df['Calories'].max())}")
with col3:
    st.metric("Avg Sedentary Min.", f"{int(df['SedentaryMinutes'].mean())}")
with col4:
    st.metric("Max Sleep", f"{int(df['TotalMinutesAsleep'].max())} min")

st.divider()

# --- LEVEL 2: TABLES (grouping and aggregation) ---
col_a, col_b = st.columns(2)

with col_a:
    st.subheader(" Descriptive Statistics")
    st.write("Numeric summary of the main variables:")
    st.dataframe(df[["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes"]].describe(),
                 width="stretch")

with col_b:
    st.subheader("Average per User")
    st.write("Aggregated by user Id (first 10 rows):")
    grouped_user = df.groupby("Id")[["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes"]].mean()
    st.dataframe(grouped_user.head(10), width="stretch")

st.divider()

# --- LEVEL 3: STEPS BY WEEKDAY ---
st.subheader("Average Steps by Day of the Week")

# Mean steps per weekday, ordered Monday to Sunday
grouped_day = analysis.average_steps_by_weekday(daily_activity)

# st.bar_chart gives an interactive chart
st.bar_chart(data=grouped_day, x="DayOfWeek", y="TotalSteps", color=charts.PRIMARY)

with st.expander("Show the data behind the chart"):
    st.table(grouped_day)
