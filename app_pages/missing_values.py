"""Missing values: median imputation and why missing sleep is not random."""

import streamlit as st

from fitbit_analytics import analysis, data
from app_pages.common import load_data

daily_activity, sleep_data, df, df_sleep = load_data()

st.title("Missing values")
st.markdown(
    f"Only the sleep columns have gaps, but they are large: **{df['TotalMinutesAsleep'].isna().mean():.0%} "
    "of the days have no sleep record**. This page shows the standard fix (median imputation) and then "
    "why it would be the wrong choice for this data."
)

# Two columns to compare "before" and "after"
col_before, col_after = st.columns(2)

with col_before:
    st.subheader("1. Before imputation")
    missing_counts = df.isnull().sum()
    # Show only the columns that actually have missing values
    st.dataframe(missing_counts[missing_counts > 0], width="stretch")

with col_after:
    st.subheader("2. After imputation")

    # Fill the missing values
    df_clean = data.fill_missing_with_median(df)

    # Check again
    after_counts = df_clean.isnull().sum()
    st.dataframe(after_counts[after_counts > 0], width="stretch")
    st.caption("Every numeric column is complete after filling the gaps with the column median.")

st.divider()

st.subheader("Missing Values per Column (Before Imputation)")
if missing_counts.sum() > 0:
    # Plot only the columns with missing values
    missing_plot_data = missing_counts[missing_counts > 0]
    st.bar_chart(missing_plot_data)
else:
    st.write("No missing values were found in the dataset.")

with st.expander("Why the median?"):
    st.write("""
    We chose **median imputation** over the mean because Fitbit data (such as steps or calories)
    tends to be skewed. In this context the median gives a more central and more stable estimate.
    """)

st.divider()

# --- Missing sleep is not random ---
st.subheader("Why the Models Do Not Use Imputed Sleep")
comparison = analysis.compare_days_by_sleep_record(df)
st.dataframe(comparison.round(0), width="stretch")

sed_without = comparison.loc["Without sleep record", "SedentaryMinutes"]
sed_with = comparison.loc["With sleep record", "SedentaryMinutes"]
st.write(f"""
Missing sleep values are **not random**. On days without a sleep record, users log on average
**{sed_without:,.0f} sedentary minutes** ({sed_without / 60:.1f} hours), compared with **{sed_with:,.0f}**
on days with a record. The most likely explanation is that the device was not worn at night,
so the night was counted as sedentary time.

Filling in a typical sleep value for these days would pair a full night of "sedentary" time with a
full night of sleep, which is impossible. In an earlier version, K-Means mostly separated days
without a sleep record from the rest, instead of finding lifestyle patterns.

For that reason, the models that use sleep (K-Means, logistic regression and OLS) work only on the
**{len(df_sleep)} days with a real sleep record**. Median imputation is shown above as a technique,
not used for modelling.
""")
