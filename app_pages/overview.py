"""Overview: the landing page of the app."""

import streamlit as st

from app_pages.common import key_results, load_data

daily_activity, sleep_data, df, df_sleep = load_data()
r = key_results()
profiles = r["profiles"]

st.title("Fitbit Health Analytics")
st.markdown(
    f"What can one month of wearable data from **{r['users']} people** tell us about activity, sleep and "
    "a healthy lifestyle? This app walks through the analysis, from data quality to three models, "
    "and is explicit about what the data can and cannot show."
)

# --- Headline numbers ---
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Users", r["users"])
k2.metric("Days of data", r["days"])
k3.metric("Steps per day", f"{r['avg_steps']:,.0f}", help="Average over all days")
k4.metric("Sleep per night", f"{df_sleep['TotalMinutesAsleep'].mean() / 60:.1f} h",
          help="Average over the nights with a sleep record")
k5.metric("Days with sleep data", f"{1 - r['missing_sleep_share']:.0%}")

# --- Highlights, each linking to the page with the details ---
st.subheader("Highlights")
c1, c2, c3 = st.columns(3)

with c1.container(border=True, height="stretch"):
    st.markdown("**Missing sleep is not random**")
    st.write(
        f"{r['missing_sleep_share']:.0%} of the days have no sleep record. On those days users log "
        f"{r['sedentary_without_sleep'] / 60:.0f} hours of sedentary time, against "
        f"{r['sedentary_with_sleep'] / 60:.0f} hours on days with a record: the device was not worn at night."
    )
    st.page_link("app_pages/coverage.py", label="See who wears the device, and when", icon=":material/arrow_forward:")

with c2.container(border=True, height="stretch"):
    st.markdown("**Three behaviour profiles**")
    st.write(
        f"Clustering finds a {', '.join(p.lower() for p in profiles.index[:-1])} and {profiles.index[-1].lower()} group. "
        f"Sleep does not follow activity: the {r['most_sleep'].lower()} group sleeps the most "
        f"({profiles.loc[r['most_sleep'], 'TotalMinutesAsleep'] / 60:.1f} h), the {r['least_sleep'].lower()} "
        f"group the least ({profiles.loc[r['least_sleep'], 'TotalMinutesAsleep'] / 60:.1f} h)."
    )
    st.page_link("app_pages/clustering.py", label="Explore the profiles", icon=":material/arrow_forward:")

with c3.container(border=True, height="stretch"):
    st.markdown("**Healthy days can be flagged, roughly**")
    st.write(
        f"Without seeing steps or sleep, a logistic regression finds {r['recall']:.0%} of the healthy days "
        f"of users it was never trained on (ROC AUC {r['roc_auc']:.2f})."
    )
    st.page_link("app_pages/logistic.py", label="See how the classifier is built", icon=":material/arrow_forward:")

# --- Guide to the app ---
st.subheader("How this app is organised")
g1, g2, g3 = st.columns(3)
with g1:
    st.markdown("**Data**")
    st.caption("Structure, quality and preparation of the two Fitbit files.")
    st.page_link("app_pages/exploration.py", label="Data exploration")
    st.page_link("app_pages/missing_values.py", label="Missing values")
    st.page_link("app_pages/outliers.py", label="Outliers")
    st.page_link("app_pages/statistics.py", label="Statistics")
    st.page_link("app_pages/encoding.py", label="Encoding and scaling")
with g2:
    st.markdown("**Models**")
    st.caption("Three questions, each answered with a different model.")
    st.page_link("app_pages/clustering.py", label="Clustering: which behaviour profiles exist?")
    st.page_link("app_pages/logistic.py", label="Classifier: can we flag healthy days?")
    st.page_link("app_pages/regression.py", label="Regression: what drives calories burned?")
with g3:
    st.markdown("**Results**")
    st.caption("Findings, limitations and recommendations, computed from the models.")
    st.page_link("app_pages/conclusions.py", label="Conclusions")
    st.page_link("app_pages/coverage.py", label="Data coverage")

# --- Data source ---
st.subheader("About the data")
st.write(
    f"Public **Crowd-sourced Fitbit dataset** (Furberg et al., 2016, [Zenodo](https://doi.org/10.5281/zenodo.53894), "
    f"CC BY 4.0): daily activity for {r['users']} users and sleep records for {r['users_with_sleep']} of them, "
    "from 12 April to 12 May 2016. The users volunteered through an online survey, so this is a small, "
    "self-selected sample: the results describe this group, not the general population."
)
