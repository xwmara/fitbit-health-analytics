"""Overview: the landing page of the app."""

import streamlit as st

from app_pages.common import load_data

daily_activity, sleep_data, df, df_sleep = load_data()

st.title("Fitbit Health Analytics")
st.caption("Modelling and evaluating a healthy lifestyle from Fitbit activity and sleep data")

# Headline metrics (daily averages; sleep only over nights with a sleep record)
k1, k2, k3 = st.columns(3)
k1.metric("Average Steps per Day", f"{df['TotalSteps'].mean():,.0f}")
k2.metric("Average Calories per Day", f"{df['Calories'].mean():,.0f} kcal")
k3.metric("Average Sleep per Night", f"{df_sleep['TotalMinutesAsleep'].mean() / 60:.1f} h")

st.write("""
This project is an exploratory (EDA) and predictive analysis of data collected by Fitbit devices.
The organisation aims to monitor health through indicators such as:
- **Physical activity:** steps and active minutes.
- **Sedentary behaviour:** minutes of inactivity.
- **Balance:** the relationship between sleep and calories burned.
""")

with st.expander("Dataset Details"):
    st.write("""
    - **dailyActivity_merged.csv**: daily totals for steps, distance and calories.
    - **sleepDay_merged.csv**: sleep tracking, including time asleep and time in bed.
    """)
