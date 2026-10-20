"""Missing values: median imputation and why missing sleep is not random."""

import pandas as pd
import streamlit as st

from fitbit_analytics import analysis, charts, data
from app_pages.common import load_data

daily_activity, sleep_data, df, df_sleep = load_data()

st.title("Missing values")
st.markdown(
    f"Only the sleep columns have gaps, but they are large: **{df['TotalMinutesAsleep'].isna().mean():.0%} "
    "of the days have no sleep record**. This page shows the standard fix (median imputation) and then "
    "why it would be the wrong choice for this data."
)

missing_counts = df.isnull().sum()
# Fill the missing values and check again
df_clean = data.fill_missing_with_median(df)
after_counts = df_clean.isnull().sum()

# Only the columns that actually have missing values
has_missing = missing_counts[missing_counts > 0].index
missing_table = pd.DataFrame({
    "Column": [charts.label(c) if c != "ActivityDate_y" else "Sleep date (join key)" for c in has_missing],
    "Missing before": missing_counts[has_missing].values,
    "Share of days": (missing_counts[has_missing] / len(df) * 100).values,
    "Missing after": after_counts[has_missing].values,
})

st.subheader("Before and after imputation")
st.dataframe(
    missing_table, hide_index=True, width="stretch",
    column_config={"Share of days": st.column_config.NumberColumn(format="%.0f%%")},
)
st.caption(f"The sleep columns are missing on the same {missing_counts.max()} days. The sleep date is a "
           "text column, so it is not filled; every numeric column is complete after filling the gaps with the column median.")

with st.expander("Why the median?"):
    st.write("""
    We chose **median imputation** over the mean because Fitbit data (such as steps or calories)
    tends to be skewed. In this context the median gives a more central and more stable estimate.
    """)

st.divider()

# --- Missing sleep is not random ---
st.subheader("Why the Models Do Not Use Imputed Sleep")
comparison = analysis.compare_days_by_sleep_record(df)
st.dataframe(charts.readable(comparison).round(0).astype(int), width="stretch",
             column_config={c: st.column_config.NumberColumn(format="localized")
                            for c in charts.readable(comparison).columns})

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
