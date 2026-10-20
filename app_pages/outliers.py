"""Outliers: IQR-based detection for a chosen variable."""

import streamlit as st
import matplotlib.pyplot as plt

from fitbit_analytics import analysis, charts
from app_pages.common import load_data

daily_activity, sleep_data, df, df_sleep = load_data()

st.title("Outliers")
st.markdown(
    "Unusually high or low values can be sensor errors or genuinely unusual days, and both can distort "
    "averages. The **IQR rule** flags values more than 1.5 × IQR below the first quartile or above the third."
)

col = st.selectbox(
    "Variable",
    ["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes", "TotalMinutesAsleep"],
    format_func=charts.label,
)

series = df[col].dropna()

# Summary statistics
outliers, bounds = analysis.find_outliers(df, col)
q1, q3, iqr = bounds["q1"], bounds["q3"], bounds["iqr"]
lower, upper = bounds["lower"], bounds["upper"]

# --- COLUMN LAYOUT ---
ol_graph, ol_stats = st.columns([2, 1])

with ol_graph:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.boxplot(series, orientation="horizontal", patch_artist=True,
               boxprops=dict(facecolor=charts.LIGHT_TEAL, color=charts.PRIMARY),
               whiskerprops=dict(color=charts.PRIMARY), capprops=dict(color=charts.PRIMARY),
               flierprops=dict(markeredgecolor=charts.ACCENT),
               medianprops=dict(color=charts.ACCENT, linewidth=2))
    ax.set_yticks([])
    ax.set_xlabel(charts.label(col))
    ax.set_title(f"Distribution of {charts.label(col).split(' (')[0].lower()}, with outliers as dots")
    charts.thousands(ax, "x")
    st.pyplot(fig)

with ol_stats:
    st.write("**Detailed statistics:**")
    st.write(f"- **Q1 (25%):** {q1:,.2f}\n- **Q3 (75%):** {q3:,.2f}\n- **IQR:** {iqr:,.2f}")
    st.divider()
    st.metric("Total Outliers", len(outliers))

st.write(
    f"Values above **{upper:,.2f}** are upper outliers and values below **{lower:,.2f}** are lower outliers."
)

# --- OUTLIER TABLE ---
if len(outliers) > 0:
    with st.expander(f"Show the {len(outliers)} outlier records"):
        st.write(f"These values fall outside the interval [{lower:,.2f}, {upper:,.2f}]")
        table = outliers[["Id", col]].sort_values(by=col, ascending=False)
        table["Id"] = table["Id"].astype(str)
        st.dataframe(charts.readable(table), hide_index=True)
else:
    st.caption("No outliers were detected for this variable.")
