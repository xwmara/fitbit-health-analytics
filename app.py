import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

from fitbit_analytics import analysis, data

AUTHOR_NAME = "Mara-Livia Popa"
GITHUB_URL = "https://github.com/xwmara"
LINKEDIN_URL = "https://www.linkedin.com/in/mara-livia-popa-99ba18337/"

# Page configuration
st.set_page_config(page_title="Fitbit Health Analytics", layout="wide")

# --- CUSTOM STYLING (CSS) ---
st.markdown(
    """
    <style>
    .main-title {
        color: #00B0B9;
        font-size: 42px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="main-title">Modelling and Evaluating a Healthy Lifestyle</h1>', unsafe_allow_html=True)
st.write("A Streamlit app built on Fitbit activity and sleep data")

# -----------------------------
# DATA LOADING
# -----------------------------
@st.cache_data
def load_data():
    # Cached so the CSV files are read only once per session
    return data.load_data()


daily_activity, sleep_data, df = load_data()

# Models that use sleep work only on days with a real sleep record (see the "Missing values" section)
df_sleep = data.days_with_sleep(df)
SLEEP_SUBSET_NOTE = (
    f"This model uses only the **{len(df_sleep)} days with a sleep record** "
    f"({df_sleep['Id'].nunique()} of {df['Id'].nunique()} users). Sleep values are not imputed, "
    "because days without a record count the night as sedentary time."
)

# Model features, shared by the model sections and the conclusions
CLUSTER_FEATURES = ["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes", "TotalMinutesAsleep"]
# Only variables that are NOT used to define the "healthy day" label (avoids data leakage)
LOGISTIC_FEATURES = ["Calories", "VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes"]
OLS_FEATURES = ["TotalSteps", "VeryActiveMinutes", "SedentaryMinutes", "TotalMinutesAsleep"]

# --- SESSION STATE ---
if 'visitor_name' not in st.session_state:
    st.session_state['visitor_name'] = ""

# Personal greeting at the top of the sidebar
with st.sidebar:
    name = st.text_input("Enter your name to personalise the app:", value=st.session_state['visitor_name'])
    st.session_state['visitor_name'] = name
    if st.session_state['visitor_name']:
        st.write(f"Hi, **{st.session_state['visitor_name']}**! 👋")

# --- SIDEBAR PROFILE ---
st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <img src="https://www.pngall.com/wp-content/uploads/5/Profile-Avatar-PNG.png" width="100" style="border-radius: 50%;">
        <h2 style="color: #00B0B9;">Fitbit User</h2>
        <p style="font-size: 0.9em; color: gray;">Health Analytics v1.0</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Stats")

# Headline metrics
st.sidebar.metric("Average Steps", f"{int(df['TotalSteps'].mean())}", "steps/day")
st.sidebar.metric("Average Calories", f"{int(df['Calories'].mean())} kcal")
st.sidebar.metric("Average Sleep", f"{int(df['TotalMinutesAsleep'].mean() if not np.isnan(df['TotalMinutesAsleep'].mean()) else 0)} min")

st.sidebar.markdown("---")
# Section menu
menu = st.sidebar.selectbox(
    "Choose a section",
    [
        "1. About the project",
        "2. Data exploration",
        "3. Missing values",
        "4. Outliers",
        "5. Statistics and aggregations",
        "6. Encoding and scaling",
        "7. K-Means clustering",
        "8. Logistic regression",
        "9. Multiple regression",
        "10. Geographic analysis",
        "11. Conclusions"
    ]
)

# Author credits
st.sidebar.markdown("---")
st.sidebar.info(
    f"""
    **Built by {AUTHOR_NAME}**

    [GitHub]({GITHUB_URL}) · [LinkedIn]({LINKEDIN_URL})
    """
)

# -----------------------------
# 1. ABOUT THE PROJECT
# -----------------------------
if menu == "1. About the project":
    st.header(" Project Context and Goals")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("""
        This project is an exploratory (EDA) and predictive analysis of data collected by Fitbit devices.
        The organisation aims to monitor health through indicators such as:
        - **Physical activity:** steps and active minutes.
        - **Sedentary behaviour:** minutes of inactivity.
        - **Balance:** the relationship between sleep and calories burned.
        """)

    with st.expander(" Dataset Details"):
        st.write("""
        - **dailyActivity_merged.csv**: daily totals for steps, distance and calories.
        - **sleepDay_merged.csv**: sleep tracking, including time asleep and time in bed.
        """)

# -----------------------------
# 2. DATA EXPLORATION
# -----------------------------
elif menu == "2. Data exploration":
    st.header("Exploratory Data Analysis (EDA)")

    st.markdown("""
    In this step we look at the structure of the datasets to understand their size,
    the variable types and how the information is organised.
    """)

    tab1, tab2, tab3 = st.tabs(["📊 Daily Activity", "😴 Sleep Tracking", "🔗 Merged Data (Join)"])

    with tab1:
        st.subheader("Daily Activity Structure")
        st.dataframe(daily_activity.head(10), use_container_width=True)

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
        st.dataframe(sleep_data.head(10), use_container_width=True)

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
        st.dataframe(df.head(10), use_container_width=True)

# -----------------------------
# 3. MISSING VALUES
# -----------------------------
elif menu == "3. Missing values":
    st.header("Handling Missing Values")

    st.markdown("""
    Finding missing values is an essential EDA step. Left untreated, they can cause errors in
    regression or clustering models. We used the **median** for imputation because it is robust to outliers.
    """)

    # Two columns to compare "before" and "after"
    col_before, col_after = st.columns(2)

    with col_before:
        st.subheader("1. Before imputation")
        missing_counts = df.isnull().sum()
        # Show only the columns that actually have missing values
        st.dataframe(missing_counts[missing_counts > 0], use_container_width=True)
        st.warning(f" {df['TotalMinutesAsleep'].isna().mean():.0%} of the days have no sleep record.")

    with col_after:
        st.subheader("2. After imputation")

        # Fill the missing values
        df_clean = data.fill_missing_with_median(df)

        # Check again
        after_counts = df_clean.isnull().sum()
        st.dataframe(after_counts[after_counts > 0], use_container_width=True)
        st.success(" All numeric values have been filled in.")

    st.divider()

    st.subheader("Missing Values per Column (Before Imputation)")
    if missing_counts.sum() > 0:
        # Plot only the columns with missing values
        missing_plot_data = missing_counts[missing_counts > 0]
        st.bar_chart(missing_plot_data)
    else:
        st.write("No missing values were found in the dataset.")

    with st.expander("📝 Why the median?"):
        st.write("""
        We chose **median imputation** over the mean because Fitbit data (such as steps or calories)
        tends to be skewed. In this context the median gives a more central and more stable estimate.
        """)

    st.divider()

    # --- Missing sleep is not random ---
    st.subheader("Why the Models Do Not Use Imputed Sleep")
    comparison = analysis.compare_days_by_sleep_record(df)
    st.dataframe(comparison.round(0), use_container_width=True)

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

# -----------------------------
# 4. OUTLIERS
# -----------------------------
elif menu == "4. Outliers":
    st.header("Outlier Detection")

    st.markdown("""
    Detecting outliers with the **IQR (interquartile range)** method helps us isolate records that
    may be sensor errors or unusual behaviour that could distort the averages.
    """)

    col = st.selectbox(
        "Select a variable to analyse:",
        ["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes", "TotalMinutesAsleep"]
    )

    series = df[col].dropna()

    # Summary statistics
    outliers, bounds = analysis.find_outliers(df, col)
    q1, q3, iqr = bounds["q1"], bounds["q3"], bounds["iqr"]
    lower, upper = bounds["lower"], bounds["upper"]

    # --- COLUMN LAYOUT ---
    ol_graph, ol_stats = st.columns([2, 1])

    with ol_graph:
        fig, ax = plt.subplots(figsize=(8, 4))
        # Fitbit teal for the box
        ax.boxplot(series, vert=False, patch_artist=True,
                   boxprops=dict(facecolor='#00B0B9', color='black'),
                   medianprops=dict(color='orange'))
        ax.set_title(f"Boxplot: distribution of {col}")
        st.pyplot(fig)

    with ol_stats:
        st.write("**Detailed statistics:**")
        st.write(f"🔹 **Q1 (25%):** {q1:.2f}")
        st.write(f"🔹 **Q3 (75%):** {q3:.2f}")
        st.write(f"🔹 **IQR:** {iqr:.2f}")
        st.divider()
        st.metric("Total Outliers", len(outliers))

    # --- OUTLIER TABLE ---
    if len(outliers) > 0:
        with st.expander(f"Show the {len(outliers)} outlier records"):
            st.write(f"These values fall outside the interval [{lower:.2f}, {upper:.2f}]")
            st.dataframe(outliers[["Id", col]].sort_values(by=col, ascending=False))
    else:
        st.success("No outliers were detected for this variable.")

    st.info(
        f"**Interpretation:** values above {upper:.2f} are upper outliers, and values below {lower:.2f} are lower outliers.")

# -----------------------------
# 5. STATISTICS AND AGGREGATIONS
# -----------------------------
elif menu == "5. Statistics and aggregations":
    st.header(" Descriptive Statistics and Group Analysis")

    # --- LEVEL 1: KEY METRICS ---
    st.subheader("Key Indicators (Overall)")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Max Steps", f"{int(df['TotalSteps'].max())}")
    with col2:
        st.metric("Max Calories", f"{int(df['Calories'].max())}")
    with col3:
        st.metric("Avg Sedentary Min.", f"{int(df['SedentaryMinutes'].mean())}")
    with col4:
        st.metric("Max Sleep", f"{int(df['TotalMinutesAsleep'].max())} min")

    st.divider()

    # --- LEVEL 2: TABLES (grouping and aggregation) ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader(" Descriptive Statistics")
        st.write("Numeric summary of the main variables:")
        st.dataframe(df[["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes"]].describe(),
                     use_container_width=True)

    with col_b:
        st.subheader("Average per User")
        st.write("Aggregated by user Id (first 10 rows):")
        grouped_user = df.groupby("Id")[["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes"]].mean()
        st.dataframe(grouped_user.head(10), use_container_width=True)

    st.divider()

    # --- LEVEL 3: STEPS BY WEEKDAY ---
    st.subheader("Average Steps by Day of the Week")

    # Mean steps per weekday, ordered Monday to Sunday
    grouped_day = analysis.average_steps_by_weekday(daily_activity)

    # st.bar_chart gives an interactive chart
    st.bar_chart(data=grouped_day, x="DayOfWeek", y="TotalSteps", color="#00B0B9")

    with st.expander("Show the data behind the chart"):
        st.table(grouped_day)

# -----------------------------
# 6. ENCODING AND SCALING
# -----------------------------
elif menu == "6. Encoding and scaling":
    st.header("Encoding and Scaling the Data")

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
        st.dataframe(df_prep[["TotalSteps", "LifestyleCategory"]].head(10), use_container_width=True)

    with col_cat2:
        st.write("**LabelEncoder result:**")
        st.dataframe(df_prep[["LifestyleCategory", "LifestyleEncoded"]].drop_duplicates(), use_container_width=True)

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
        st.dataframe(X.head(10), use_container_width=True)

    with tab_scaled:
        st.dataframe(X_scaled_df.head(10), use_container_width=True)
        st.caption("All values are now on a comparable scale (roughly between -3 and 3).")

# -----------------------------
# 7. CLUSTERING
# -----------------------------
elif menu == "7. K-Means clustering":
    st.header("User Segmentation (K-Means Clustering)")

    st.info("""
    **Goal:** group users by their behaviour (steps, sleep, calories)
    to identify different segments. We used **k = 3 clusters**.
    """)
    st.caption(SLEEP_SUBSET_NOTE)

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
                     use_container_width=True)

    st.divider()

    # --- SCATTER PLOT ---
    st.subheader("Segments: Steps vs. Calories")

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df_cluster["TotalSteps"],
        df_cluster["Calories"],
        c=df_cluster["Cluster"],
        cmap='viridis',
        s=50,
        alpha=0.7
    )

    # Legend
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    ax.set_xlabel("Total Steps")
    ax.set_ylabel("Calories")
    ax.grid(True, linestyle='--', alpha=0.6)

    st.pyplot(fig)

    with st.expander("📝 Business Interpretation"):
        label_of = {name: c for c, name in cluster_names.items()}
        st.write(f"""
        The organisation can use these segments for:
        1. **Targeting:** personalised messages for the least active group (Cluster {label_of["Low activity"]}).
        2. **Retention:** loyalty programmes for the most active users (Cluster {label_of["Very active"]}).
        """)

# -----------------------------
# 8. LOGISTIC REGRESSION
# -----------------------------
elif menu == "8. Logistic regression":
    st.header("Logistic Regression: Predicting a Healthy Lifestyle")

    st.info("""
    **Goal:** classify each day as 'Healthy lifestyle' (1) or 'Needs improvement' (0).
    A day counts as healthy with at least 8,000 steps, fewer than 1,000 sedentary minutes and at least 7 hours of sleep.
    """)
    st.caption(SLEEP_SUBSET_NOTE)

    # The label needs real sleep values, so only days with a sleep record are used
    df_log = df_sleep.copy()

    # Target variable
    df_log["HealthyLifestyle"] = analysis.healthy_lifestyle_label(df_log)

    # Only variables that are NOT used to define the label. Steps, distance (~steps),
    # sedentary minutes and sleep would let the model rebuild the label instead of learning.
    features = LOGISTIC_FEATURES
    X = data.fill_missing_with_median(df_log[features], features)

    y = df_log["HealthyLifestyle"]

    with st.expander("🔍 How the model is set up", expanded=True):
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

    st.divider()

    # --- FEATURE IMPORTANCE ---
    st.subheader("Factor Importance for a Healthy Lifestyle")
    importance = result.coefficients
    st.bar_chart(data=importance, x='Feature', y='Importance', color="#00B0B9")

    with st.expander("📝 Interpreting the Results"):
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

# -----------------------------
# 9. MULTIPLE REGRESSION
# -----------------------------
elif menu == "9. Multiple regression":
    st.header("Multiple Regression: What Drives Calories Burned")

    st.info("""
    **Goal:** understand how physical activity and sleep relate to calories burned.
    We use an **OLS (ordinary least squares)** model to estimate the coefficients.
    """)
    st.caption(SLEEP_SUBSET_NOTE)

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
                    scatter_kws={'alpha': 0.3, 'color': 'gray'},
                    line_kws={'color': '#00B0B9'})
        ax.set_title("Regression model accuracy")
        st.pyplot(fig)

    # --- LEVEL 3: INTERPRETATION ---
    with st.expander("Interpreting the Coefficients", expanded=True):
        # Raw coefficients use different units (per step vs. per minute), so compare standardized ones
        coef_table = analysis.standardized_coefficients(model, X, y)
        st.dataframe(coef_table.set_index("Feature").round(4), use_container_width=True)

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

# -----------------------------
# 10. GEOGRAPHIC ANALYSIS
# -----------------------------
elif menu == "10. Geographic analysis":
    st.header("Geographic Analysis")
    st.write("A view of the users' global coverage.")

    col_map1, col_map2 = st.columns([3, 1])

    with col_map1:
        st.markdown("""
            This view uses **GeoPandas** to show the global presence of
            Fitbit users. The organisation collects data from many regions to understand
            how geographic factors influence lifestyle.
            """)

    with col_map2:
        st.info("**Data source:** GeoJSON World Boundaries")
        st.metric("Regions Analysed", "Global")

    try:
        url = "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson"
        world = gpd.read_file(url)

        fig, ax = plt.subplots(figsize=(12, 7))
        world.plot(ax=ax, color='#00B0B9', edgecolor='white', alpha=0.8)
        ax.set_axis_off()
        ax.set_title("Global Coverage of Fitbit Users", fontsize=15)
        st.pyplot(fig)
        st.success("Map generated successfully!")

    except Exception as e:
        st.error(f"Error: {e}")
        st.info(
            "If the error persists, the link may be blocked by a local firewall. The map is optional.")

# -----------------------------
# 11. CONCLUSIONS
# -----------------------------
elif menu == "11. Conclusions":
    st.header("Conclusions")
    st.write("Every number below is computed from the data with the same settings as the sections above.")

    # --- Recompute the key results ---
    comparison = analysis.compare_days_by_sleep_record(df)
    missing_share = df["TotalMinutesAsleep"].isna().mean()
    steps_by_day = analysis.average_steps_by_weekday(daily_activity).set_index("DayOfWeek")["TotalSteps"]
    share_10k = (daily_activity["TotalSteps"] >= 10000).mean()

    clusters = df_sleep[CLUSTER_FEATURES].copy()
    clusters["Cluster"] = analysis.kmeans_clusters(clusters[CLUSTER_FEATURES], n_clusters=3)
    profiles = clusters.groupby("Cluster")[CLUSTER_FEATURES].mean()
    profiles.index = profiles.index.map(analysis.name_clusters(profiles))
    profiles = profiles.loc[analysis.CLUSTER_NAMES]
    most_sleep, least_sleep = profiles["TotalMinutesAsleep"].idxmax(), profiles["TotalMinutesAsleep"].idxmin()

    healthy = analysis.healthy_lifestyle_label(df_sleep)
    log_result = analysis.train_logistic_regression(df_sleep[LOGISTIC_FEATURES], healthy, groups=df_sleep["Id"])

    ols_model, _ = analysis.fit_ols(df_sleep[OLS_FEATURES], df_sleep["Calories"])
    ols_top = analysis.standardized_coefficients(ols_model, df_sleep[OLS_FEATURES], df_sleep["Calories"]).iloc[0]

    # --- Headline numbers ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Users", df["Id"].nunique())
    k2.metric("Days analysed", len(df))
    k3.metric("Days with sleep data", f"{1 - missing_share:.0%}")
    k4.metric("Healthy days", f"{healthy.mean():.0%}", help="Share of days with sleep data that meet all three targets")

    st.subheader("Key Findings")
    profile_lines = "\n".join(
        f"   - **{name}:** ~{row.TotalSteps:,.0f} steps, {row.VeryActiveMinutes:.0f} very active minutes, "
        f"{row.TotalMinutesAsleep / 60:.1f} h of sleep per day"
        for name, row in profiles.iterrows()
    )
    st.markdown(f"""
1. **Data quality comes first.** {missing_share:.0%} of the days have no sleep record. On those days users log
   {comparison.loc["Without sleep record", "SedentaryMinutes"]:,.0f} sedentary minutes on average, against
   {comparison.loc["With sleep record", "SedentaryMinutes"]:,.0f} on days with a record: the device was most likely
   not worn at night. The models therefore use only the {len(df_sleep)} days with real sleep data.
2. **Most days fall short of 10,000 steps.** Users average {daily_activity["TotalSteps"].mean():,.0f} steps a day and
   only {share_10k:.0%} of the days reach 10,000 steps. {steps_by_day.idxmax()} is the most active day
   ({steps_by_day.max():,.0f} steps on average) and {steps_by_day.idxmin()} the least active ({steps_by_day.min():,.0f}).
3. **Three behaviour profiles.** K-Means finds:
{profile_lines}

   Sleep does not simply follow activity: the **{most_sleep}** group sleeps the most and the
   **{least_sleep}** group the least.
4. **Intense activity drives calories burned.** Activity and sleep explain {ols_model.rsquared:.0%} of the variation
   in daily calories (R²). The strongest effect comes from **{ols_top.Feature}** (standardized coefficient
   {ols_top.StdCoefficient:.2f}).
5. **Healthy days can be flagged, roughly.** Using only calories and active minutes, the logistic model finds
   {log_result.recall:.0%} of the healthy days of users it has never seen (ROC AUC {log_result.roc_auc:.2f}).
   It is a useful first filter, not a reliable diagnosis on its own.
""")

    st.subheader("Limitations")
    st.markdown(f"""
- **Small, self-selected sample:** {df["Id"].nunique()} users ({df_sleep["Id"].nunique()} with sleep data) who volunteered
  through an online survey, over one month in 2016. The results should not be generalised.
- **Correlation, not causation:** the models describe associations in this data, not causes.
- **Device estimates:** Fitbit estimates calories partly from the same activity data, so the regression partly
  recovers the device's own formula.
- **Missing sleep data:** days without a sleep record are excluded from the models, which may favour users who
  wear the device consistently.
""")

    st.subheader("Recommendations")
    st.markdown(f"""
1. **Improve data collection:** encourage users to wear the device at night; {missing_share:.0%} of the days have
   no sleep data.
2. **Targeted activity goals:** the **Low activity** group averages
   {profiles.loc["Low activity", "TotalSteps"]:,.0f} steps a day; gradual step challenges fit this group better
   than a fixed 10,000-step goal.
3. **Sleep reminders:** the **{least_sleep}** group sleeps only {profiles.loc[least_sleep, "TotalMinutesAsleep"] / 60:.1f} h
   a night on average, below the 7 hours used in the "healthy day" definition.
""")
