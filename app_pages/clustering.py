"""K-Means clustering: behaviour profiles."""

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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

# --- RESULTS ---
col_cl1, col_cl2 = st.columns([1, 2])

with col_cl1:
    st.subheader(" Cluster Sizes")
    st.write(df_cluster["Cluster"].value_counts())

    st.subheader(" Cluster Profiles")
    st.write("\n".join(
        f"- **Cluster {c}: {cluster_names[c]}** "
        f"(~{row.TotalSteps:,.0f} steps, {row.VeryActiveMinutes:.0f} very active minutes "
        f"and {row.TotalMinutesAsleep / 60:.1f} hours of sleep per day)"
        for c, row in cluster_means.iterrows()
    ))

with col_cl2:
    st.subheader(" Cluster Comparison")
    st.dataframe(cluster_means.assign(Profile=cluster_means.index.map(cluster_names)),
                 width="stretch")

st.divider()

# --- SCATTER PLOT ---
st.subheader("Segments: Steps vs. Calories")

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    df_cluster["TotalSteps"],
    df_cluster["Calories"],
    c=df_cluster["Cluster"],
    # Each cluster gets its profile's fixed colour (cluster ids are 0, 1, 2)
    cmap=ListedColormap([charts.CLUSTER_COLORS[cluster_names[c]] for c in sorted(cluster_names)]),
    s=40,
    alpha=0.75,
    edgecolors="white",
    linewidths=0.5,
)

# Legend
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

ax.set_xlabel("Total Steps")
ax.set_ylabel("Calories")
ax.grid(True, linestyle='--', alpha=0.6)

st.pyplot(fig)

with st.expander("Business interpretation"):
    label_of = {name: c for c, name in cluster_names.items()}
    st.write(f"""
    The organisation can use these segments for:
    1. **Targeting:** personalised messages for the least active group (Cluster {label_of["Low activity"]}).
    2. **Retention:** loyalty programmes for the most active users (Cluster {label_of["Very active"]}).
    """)
