"""Multiple regression (OLS): what drives calories burned."""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from fitbit_analytics import analysis, charts
from app_pages.common import load_data, OLS_FEATURES, sleep_subset_note

daily_activity, sleep_data, df, df_sleep = load_data()

st.title("Multiple Regression: What Drives Calories Burned")

st.info("""
**Goal:** understand how physical activity and sleep relate to calories burned.
We use an **OLS (ordinary least squares)** model to estimate the coefficients.
""")
st.caption(sleep_subset_note(df, df_sleep))

df_reg = df_sleep.copy()

# Model variables
features = OLS_FEATURES
X = df_reg[features]
y = df_reg["Calories"]

# Add the intercept and fit the model
model, X_with_const = analysis.fit_ols(X, y)

# --- LEVEL 1: FIT METRICS ---
col_r1, col_r2, col_r3 = st.columns(3)
with col_r1:
    st.metric("R-squared", f"{model.rsquared:.3f}")
with col_r2:
    st.metric("Adj. R-squared", f"{model.rsquared_adj:.3f}")
with col_r3:
    # Overall significance of the model (F-test)
    status = "Valid" if model.f_pvalue < 0.05 else "Invalid"
    st.metric("Model Status (p-value)", status)

st.divider()

# --- LEVEL 2: REPORT AND CHART ---
tab_stats, tab_viz = st.tabs(["Statistical Report", "Prediction Chart"])

with tab_stats:
    st.subheader("Full OLS Summary")
    st.text(model.summary())
    st.caption("A p-value below 0.05 marks a statistically significant variable.")

with tab_viz:
    st.subheader("Actual vs. Predicted Values")
    df_reg["Predictions"] = model.predict(X_with_const)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(data=df_reg, x="Calories", y="Predictions",
                scatter_kws={'alpha': 0.4, 'color': charts.GREY, 's': 18},
                line_kws={'color': charts.PRIMARY, 'linewidth': 2})
    ax.set_title("Regression model accuracy")
    st.pyplot(fig)

# --- LEVEL 3: INTERPRETATION ---
with st.expander("Interpreting the Coefficients", expanded=True):
    # Raw coefficients use different units (per step vs. per minute), so compare standardized ones
    coef_table = analysis.standardized_coefficients(model, X, y)
    st.dataframe(coef_table.set_index("Feature").round(4), width="stretch")

    significant = coef_table[coef_table["PValue"] < 0.05]
    top = significant.iloc[0]
    st.write(
        f"According to the model, the variable with the largest effect on calories is **{top.Feature}**: "
        f"one standard deviation more is associated with **{top.StdCoefficient:+.2f}** standard deviations "
        f"of calories burned, holding the other variables constant."
    )
    corr = X["TotalSteps"].corr(X["VeryActiveMinutes"])
    st.caption(
        f"TotalSteps and VeryActiveMinutes are correlated (r = {corr:.2f}), so their separate effects "
        "should be read with care: part of what one measures is also captured by the other."
    )
    st.write("""
    - **Coefficient:** extra calories per one-unit increase (one step, one minute).
    - **StdCoefficient:** effect in standard deviations, comparable across variables.
    - A p-value above 0.05 means the variable is not a reliable predictor.
    """)
