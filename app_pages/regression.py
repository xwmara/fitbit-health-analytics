"""Multiple regression (OLS): what drives calories burned."""

import streamlit as st
import matplotlib.pyplot as plt

from fitbit_analytics import analysis, charts
from app_pages.common import load_data, OLS_FEATURES, sleep_subset_note

daily_activity, sleep_data, df, df_sleep = load_data()

st.title("Calories regression")
st.markdown(
    "**What drives calories burned?** A multiple linear regression (OLS) relates daily calories to steps, "
    "very active minutes, sedentary time and sleep, and shows which of them matters most."
)
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
    p_text = "< 0.001" if model.f_pvalue < 0.001 else f"{model.f_pvalue:.3f}"
    st.metric("F-test p-value", p_text, help="Below 0.05: the variables jointly explain calories better than no model")

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

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(df_reg["Calories"], df_reg["Predictions"], s=18, alpha=0.5, color=charts.PRIMARY,
               edgecolors="white", linewidths=0.4)
    limits = [min(df_reg["Calories"].min(), df_reg["Predictions"].min()),
              max(df_reg["Calories"].max(), df_reg["Predictions"].max())]
    ax.plot(limits, limits, linestyle="--", color=charts.ACCENT, linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual calories (kcal)")
    ax.set_ylabel("Predicted calories (kcal)")
    ax.legend(loc="upper left")
    charts.thousands(ax)
    st.pyplot(fig)
    st.caption("Each dot is a day. The closer the dots are to the dashed line, the better the prediction.")

# --- LEVEL 3: INTERPRETATION ---
with st.expander("Interpreting the Coefficients", expanded=True):
    # Raw coefficients use different units (per step vs. per minute), so compare standardized ones
    coef_table = analysis.standardized_coefficients(model, X, y)
    st.dataframe(
        coef_table.assign(Feature=coef_table["Feature"].map(charts.label)).set_index("Feature"),
        width="stretch",
        column_config={
            "Coefficient": st.column_config.NumberColumn("Coefficient (kcal per unit)", format="%.3f"),
            "StdCoefficient": st.column_config.NumberColumn("Standardized effect", format="%.2f"),
            "PValue": st.column_config.NumberColumn("p-value", format="%.4f"),
        },
    )

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
