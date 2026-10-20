"""Outliers: IQR-based detection for a chosen variable."""

import streamlit as st
import matplotlib.pyplot as plt

from fitbit_analytics import analysis, charts
from app_pages.common import load_data

daily_activity, sleep_data, df, df_sleep = load_data()

st.title("Outlier Detection")

st.markdown("""
Detecting outliers with the **IQR (interquartile range)** method helps us isolate records that
may be sensor errors or unusual behaviour that could distort the averages.
""")

col = st.selectbox(
    "Select a variable to analyse:",
    ["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes", "TotalMinutesAsleep"]
)

series = df[col].dropna()

# Summary statistics
outliers, bounds = analysis.find_outliers(df, col)
q1, q3, iqr = bounds["q1"], bounds["q3"], bounds["iqr"]
lower, upper = bounds["lower"], bounds["upper"]

# --- COLUMN LAYOUT ---
ol_graph, ol_stats = st.columns([2, 1])

with ol_graph:
    fig, ax = plt.subplots(figsize=(8, 4))
    # Fitbit teal for the box
    ax.boxplot(series, orientation="horizontal", patch_artist=True,
               boxprops=dict(facecolor=charts.LIGHT_TEAL, color=charts.PRIMARY),
               whiskerprops=dict(color=charts.PRIMARY), capprops=dict(color=charts.PRIMARY),
               flierprops=dict(markeredgecolor=charts.ACCENT),
               medianprops=dict(color=charts.ACCENT, linewidth=2))
    ax.set_title(f"Boxplot: distribution of {col}")
    st.pyplot(fig)

with ol_stats:
    st.write("**Detailed statistics:**")
    st.write(f"🔹 **Q1 (25%):** {q1:.2f}")
    st.write(f"🔹 **Q3 (75%):** {q3:.2f}")
    st.write(f"🔹 **IQR:** {iqr:.2f}")
    st.divider()
    st.metric("Total Outliers", len(outliers))

# --- OUTLIER TABLE ---
if len(outliers) > 0:
    with st.expander(f"Show the {len(outliers)} outlier records"):
        st.write(f"These values fall outside the interval [{lower:.2f}, {upper:.2f}]")
        st.dataframe(outliers[["Id", col]].sort_values(by=col, ascending=False))
else:
    st.success("No outliers were detected for this variable.")

st.info(
    f"**Interpretation:** values above {upper:.2f} are upper outliers, and values below {lower:.2f} are lower outliers.")
