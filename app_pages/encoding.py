"""Encoding and scaling: preparing variables for the models."""

import streamlit as st

from fitbit_analytics import analysis, data
from app_pages.common import load_data

daily_activity, sleep_data, df, df_sleep = load_data()

st.title("Encoding and Scaling the Data")

st.info("""
This step prepares the data for the machine learning models (K-Means, regression).
Distance-based algorithms such as K-Means need scaling so that variables with large values (steps)
do not dominate variables with smaller values (calories).
""")

# Quick clean-up to avoid errors during scaling
df_prep = data.fill_missing_with_median(df, ["TotalMinutesAsleep"])

# --- 1. LABEL ENCODING ---
st.subheader("1. Encoding Categorical Variables")

df_prep = analysis.add_lifestyle_encoding(df_prep)

col_cat1, col_cat2 = st.columns(2)
with col_cat1:
    st.write("**New variable:**")
    st.dataframe(df_prep[["TotalSteps", "LifestyleCategory"]].head(10), width="stretch")

with col_cat2:
    st.write("**LabelEncoder result:**")
    st.dataframe(df_prep[["LifestyleCategory", "LifestyleEncoded"]].drop_duplicates(), width="stretch")

st.divider()

# --- 2. SCALING (StandardScaler) ---
st.subheader("2. Scaling Numeric Data (Standardisation)")
st.write("We transform the data to have mean 0 and standard deviation 1.")

features = ["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes", "TotalMinutesAsleep"]
# Make sure every missing value is filled before scaling
X = data.fill_missing_with_median(df_prep[features], features)
X_scaled_df = analysis.standardize(X)

tab_raw, tab_scaled = st.tabs(["Original Data", "Scaled Data (StandardScaler)"])

with tab_raw:
    st.dataframe(X.head(10), width="stretch")

with tab_scaled:
    st.dataframe(X_scaled_df.head(10), width="stretch")
    st.caption("All values are now on a comparable scale (roughly between -3 and 3).")
