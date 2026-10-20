"""Data coverage: when each user wore the device."""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from fitbit_analytics import analysis, charts
from app_pages.common import load_data

daily_activity, sleep_data, df, df_sleep = load_data()

st.title("Data coverage")
st.markdown(
    "**Who wears the device, and when?** Each row is a user and each column a day; the colour shows what "
    "was recorded. This view explains why so many sleep values are missing, and why the models use only "
    "days with a sleep record."
)

matrix = analysis.coverage_matrix(df)
per_user = analysis.coverage_by_user(df)
status = analysis.day_coverage_status(df)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Users with no sleep record", int((per_user["NightsWithSleep"] == 0).sum()))
k2.metric("Users with sleep on ≥ 80% of nights", int((per_user["SleepCoverage"] >= 0.8).sum()))
k3.metric("Days not worn", int((status == 1).sum()), help="0 steps and 24 hours of sedentary time")
k4.metric("Days with sleep data", f"{(status == 3).mean():.0%}")

# Heatmap with one colour per coverage status
colors = [charts.LIGHT_GREY, charts.ACCENT, charts.LIGHT_TEAL, charts.PRIMARY]
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(
    matrix,
    cmap=ListedColormap(colors),
    vmin=-0.5, vmax=3.5,
    linewidths=0.5, linecolor="white",
    cbar=False,
    ax=ax,
)
ax.set_xticks(np.arange(len(matrix.columns))[::3] + 0.5)
ax.set_xticklabels([d.strftime("%b %d") for d in matrix.columns[::3]], rotation=45, ha="right")
ax.set_yticks(np.arange(len(matrix.index)) + 0.5)
ax.set_yticklabels(matrix.index.astype(str), fontsize=8)
ax.set_xlabel("Date")
ax.set_ylabel("User Id")
ax.legend(
    handles=[Patch(color=c, label=analysis.COVERAGE_LABELS[code]) for code, c in enumerate(colors)],
    loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False,
)
st.pyplot(fig)

st.write("""
**What this shows:** most users wear the device during the day, but only some of them wear it at night.
Sleep data depends on the user's habits rather than on chance, so filling in the missing nights with a
typical value would hide a real difference between users.
""")

with st.expander("Show coverage per user"):
    per_user_display = per_user.copy()
    per_user_display.index = per_user_display.index.astype(str)
    st.dataframe(
        per_user_display, width="stretch",
        column_config={
            "DaysRecorded": st.column_config.NumberColumn("Days recorded"),
            "DaysNotWorn": st.column_config.NumberColumn("Days not worn"),
            "NightsWithSleep": st.column_config.NumberColumn("Nights with sleep"),
            "SleepCoverage": st.column_config.ProgressColumn("Sleep coverage", format="percent",
                                                             min_value=0, max_value=1),
        },
    )
