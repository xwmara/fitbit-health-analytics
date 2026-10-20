"""Data exploration: structure of the raw and merged tables."""

import streamlit as st

from app_pages.common import load_data

daily_activity, sleep_data, df, df_sleep = load_data()

st.title("Data exploration")
st.markdown(
    f"Two files from the same users: one row per user and day for **activity** "
    f"({len(daily_activity)} rows) and one row per user and night for **sleep** ({len(sleep_data)} rows). "
    "They are joined into a single table with one row per user and day."
)

tab1, tab2, tab3 = st.tabs(["Daily activity", "Sleep", "Merged table"])

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
    st.caption("The tables are combined with a **left join** on `[Id, date]`, so every activity day is kept; "
               "days without a sleep record get empty sleep columns.")
    st.dataframe(df.head(10), width="stretch")
