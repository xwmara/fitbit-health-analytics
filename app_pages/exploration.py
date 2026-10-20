"""Data exploration: structure of the raw and merged tables."""

import streamlit as st

from app_pages.common import load_data

daily_activity, sleep_data, df, df_sleep = load_data()

st.title("Exploratory Data Analysis (EDA)")

st.markdown("""
In this step we look at the structure of the datasets to understand their size,
the variable types and how the information is organised.
""")

tab1, tab2, tab3 = st.tabs(["📊 Daily Activity", "😴 Sleep Tracking", "🔗 Merged Data (Join)"])

with tab1:
    st.subheader("Daily Activity Structure")
    st.dataframe(daily_activity.head(10), width="stretch")

    # Column information
    col1_info, col2_info = st.columns(2)
    with col1_info:
        st.write(f"**Rows:** {daily_activity.shape[0]}")
        st.write(f"**Columns:** {daily_activity.shape[1]}")
    with col2_info:
        if st.checkbox("Show data types", key="chk1"):
            st.write(daily_activity.dtypes.astype(str))

with tab2:
    st.subheader("Sleep Data Structure")
    st.caption("Exact duplicate rows in the sleep file are removed when the data is loaded.")
    st.dataframe(sleep_data.head(10), width="stretch")

    col1_s, col2_s = st.columns(2)
    with col1_s:
        st.write(f"**Rows:** {sleep_data.shape[0]}")
        st.write(f"**Columns:** {sleep_data.shape[1]}")
    with col2_s:
        if st.checkbox("Show data types", key="chk2"):
            st.write(sleep_data.dtypes.astype(str))

with tab3:
    st.subheader("Merged Dataset (Master Table)")
    st.info("The tables were combined with a **left join** on the composite key `[Id, ActivityDate]`.")
    st.dataframe(df.head(10), width="stretch")
