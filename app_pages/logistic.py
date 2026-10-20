"""Logistic regression: flagging healthy days."""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from fitbit_analytics import analysis, charts, data
from app_pages.common import load_data, LOGISTIC_FEATURES, sleep_subset_note

daily_activity, sleep_data, df, df_sleep = load_data()

st.title("Healthy-day classifier")
st.markdown(
    "**Can we flag healthy days?** A day counts as *healthy* with at least 8,000 steps, fewer than "
    "1,000 sedentary minutes and at least 7 hours of sleep. A logistic regression tries to recognise "
    "these days without being shown steps, sedentary time or sleep."
)
st.caption(sleep_subset_note(df, df_sleep))

# The label needs real sleep values, so only days with a sleep record are used
df_log = df_sleep.copy()

# Target variable
df_log["HealthyLifestyle"] = analysis.healthy_lifestyle_label(df_log)

# Only variables that are NOT used to define the label. Steps, distance (~steps),
# sedentary minutes and sleep would let the model rebuild the label instead of learning.
features = LOGISTIC_FEATURES
X = data.fill_missing_with_median(df_log[features], features)

y = df_log["HealthyLifestyle"]

with st.expander("How the model is set up", expanded=True):
    st.write(f"""
    - **Features:** {", ".join(features)}. The variables used to define the label (steps, sedentary
      minutes, sleep) and distance, which is almost the same as steps, are left out. Otherwise the
      model would just rebuild the rule instead of learning anything (*data leakage*).
    - **Train/test split by user:** 30% of the users are held out, so the model is tested on people
      it has never seen.
    - **Balanced class weights:** only {y.mean():.0%} of the days are healthy, so the rare class gets more weight.
    """)

# Train/test split by user, scaling and model training
result = analysis.train_logistic_regression(X, y, groups=df_log["Id"], test_size=0.3, random_state=42)
acc = result.accuracy

# --- RESULTS ---
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{acc:.1%}")
m2.metric("Baseline accuracy", f"{result.baseline_accuracy:.1%}",
          help="Accuracy of always predicting the most common class")
m3.metric("Recall (healthy)", f"{result.recall:.1%}", help="Share of healthy days the model finds")
m4.metric("F1 (healthy)", f"{result.f1:.2f}")
m5.metric("ROC AUC", f"{result.roc_auc:.2f}", help="0.5 = random guessing, 1.0 = perfect separation")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Classification Report")
    st.text(result.report)

with c2:
    st.subheader("Confusion Matrix")
    cm = result.confusion_matrix

    # Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap=sns.light_palette(charts.PRIMARY, as_cmap=True),
                ax=ax, cbar=False, linewidths=1, linecolor="white")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

st.divider()

# --- FEATURE IMPORTANCE ---
st.subheader("Factor Importance for a Healthy Lifestyle")
importance = result.coefficients
st.bar_chart(data=importance, x='Feature', y='Importance', color=charts.PRIMARY)

with st.expander("Interpreting the results"):
    st.write(f"""
    Healthy days are the minority class, so accuracy alone is misleading: always answering "not healthy"
    already scores **{result.baseline_accuracy:.1%}** on this test set, but finds none of the healthy days.
    The model, trained with balanced class weights, finds **{result.recall:.0%}** of them, at the cost of
    some false alarms (accuracy **{acc:.1%}**).
    Recall, F1 and ROC AUC show how well the model actually finds healthy days. An ROC AUC of
    **{result.roc_auc:.2f}** (0.5 = chance, 1.0 = perfect) means the model separates the two classes
    clearly better than chance, but not perfectly, with only {df_log["Id"].nunique()} users to learn from.

    The coefficients are computed on standardised features, so they can be compared: the largest
    positive one, **{importance.iloc[0]['Feature']}**, raises the predicted probability of a healthy day the most.
    """)
