"""K-Means clustering: behaviour profiles."""

import streamlit as st
import matplotlib.pyplot as plt

from fitbit_analytics import analysis, charts
from app_pages.common import load_data, CLUSTER_FEATURES, sleep_subset_note

daily_activity, sleep_data, df, df_sleep = load_data()

st.title("Clustering")
st.markdown(
    "**Which behaviour profiles exist?** K-Means groups the days (one user, one day) by steps, calories, "
    "sedentary time, very active minutes and sleep into **3 clusters**. Each cluster is named after its "
    "average step count."
)
st.caption(sleep_subset_note(df, df_sleep))

features = CLUSTER_FEATURES

# Data preparation
df_cluster = df_sleep.copy()

# K-Means model (on standardised data)
df_cluster["Cluster"] = analysis.kmeans_clusters(df_cluster[features], n_clusters=3)

# Name the clusters from their average steps (K-Means numbers them arbitrarily)
cluster_means = df_cluster.groupby("Cluster")[features].mean()
cluster_names = analysis.name_clusters(cluster_means)

# --- RESULTS: one row per profile, from least to most active ---
st.subheader("Cluster Profiles")
profile_table = charts.readable(cluster_means).assign(
    Profile=cluster_means.index.map(cluster_names),
    Days=df_cluster["Cluster"].value_counts(),
)
profile_table["Sleep (h)"] = profile_table.pop("Sleep (min)") / 60
profile_table = (
    profile_table.reset_index()
    .set_index("Profile")
    .loc[analysis.CLUSTER_NAMES, ["Cluster", "Days", "Steps", "Calories (kcal)", "Sedentary (min)",
                                  "Very active (min)", "Sleep (h)"]]
)
display_table = profile_table.round(0)
display_table["Sleep (h)"] = profile_table["Sleep (h)"].round(1)
number = st.column_config.NumberColumn(format="localized")
st.dataframe(
    display_table,
    width="stretch",
    column_config={
        **{c: number for c in ["Days", "Steps", "Calories (kcal)", "Sedentary (min)", "Very active (min)"]},
        "Sleep (h)": st.column_config.NumberColumn(format="%.1f"),
    },
)
most_sleep = profile_table["Sleep (h)"].idxmax()
least_sleep = profile_table["Sleep (h)"].idxmin()
st.write(
    f"Average values per day. Sleep does not simply follow activity: the **{most_sleep}** group sleeps the "
    f"most ({profile_table.loc[most_sleep, 'Sleep (h)']:.1f} h) and the **{least_sleep}** group the least "
    f"({profile_table.loc[least_sleep, 'Sleep (h)']:.1f} h)."
)

st.divider()

# --- SCATTER PLOT ---
st.subheader("Segments: Steps vs. Calories")

fig, ax = plt.subplots(figsize=(10, 5.5))
# One scatter call per profile, in a fixed order and colour, so the legend shows profile names
for c, name in sorted(cluster_names.items(), key=lambda item: analysis.CLUSTER_NAMES.index(item[1])):
    days = df_cluster[df_cluster["Cluster"] == c]
    ax.scatter(days["TotalSteps"], days["Calories"], s=36, alpha=0.75,
               color=charts.CLUSTER_COLORS[name], edgecolors="white", linewidths=0.5,
               label=f"{name} ({len(days)} days)")

ax.legend(title="Profile", loc="upper left")
ax.set_xlabel(charts.label("TotalSteps"))
ax.set_ylabel(charts.label("Calories"))
charts.thousands(ax)

st.pyplot(fig)

with st.expander("Business interpretation"):
    label_of = {name: c for c, name in cluster_names.items()}
    st.write(f"""
    The organisation can use these segments for:
    1. **Targeting:** personalised messages for the least active group (Cluster {label_of["Low activity"]}).
    2. **Retention:** loyalty programmes for the most active users (Cluster {label_of["Very active"]}).
    """)
