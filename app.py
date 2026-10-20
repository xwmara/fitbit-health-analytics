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
        st.warning(" The sleep columns have many missing values.")

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

    features = ["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes", "TotalMinutesAsleep"]

    # Data preparation
    df_cluster = data.fill_missing_with_median(df, features)

    # K-Means model (on standardised data)
    df_cluster["Cluster"] = analysis.kmeans_clusters(df_cluster[features], n_clusters=3)

    # --- RESULTS ---
    col_cl1, col_cl2 = st.columns([1, 2])

    with col_cl1:
        st.subheader(" Cluster Sizes")
        st.write(df_cluster["Cluster"].value_counts())

        st.subheader(" Cluster Profiles")
        st.write("""
        - **Cluster 0:** balanced users.
        - **Cluster 1:** very active users (athletes).
        - **Cluster 2:** sedentary users.
        """)

    with col_cl2:
        st.subheader(" Cluster Comparison")
        cluster_means = df_cluster.groupby("Cluster")[features].mean()
        st.dataframe(cluster_means, use_container_width=True)

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
        st.write("""
        The organisation can use these segments for:
        1. **Targeting:** personalised messages for the sedentary group.
        2. **Retention:** loyalty programmes for the most active users (Cluster 1).
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

    # Data clean-up
    df_log = data.fill_missing_with_median(df, ["TotalMinutesAsleep"])

    # Target variable
    df_log["HealthyLifestyle"] = analysis.healthy_lifestyle_label(df_log)

    features = ["Calories", "SedentaryMinutes", "VeryActiveMinutes", "TotalMinutesAsleep", "TotalDistance"]
    X = data.fill_missing_with_median(df_log[features], features)

    y = df_log["HealthyLifestyle"]

    # Train/test split, scaling and model training
    result = analysis.train_logistic_regression(X, y, test_size=0.3, random_state=42)
    acc = result.accuracy

    # --- RESULTS ---
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Model Performance")
        st.metric("Accuracy", f"{acc:.2%}")
        st.write("Classification report:")
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
        The model has an accuracy of **{acc:.2%}**.
        The variable with the largest positive coefficient has the strongest influence on the probability of a healthy day.
        Here, **{importance.iloc[0]['Feature']}** is the key factor.
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

    cols = ["Calories", "TotalSteps", "VeryActiveMinutes", "SedentaryMinutes", "TotalMinutesAsleep"]
    df_reg = data.fill_missing_with_median(df, cols)

    # Model variables
    features = ["TotalSteps", "VeryActiveMinutes", "SedentaryMinutes", "TotalMinutesAsleep"]
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
    with st.expander("Interpreting the Coefficients"):
        coeffs = model.params[1:]  # exclude the intercept
        max_impact = coeffs.idxmax()
        st.write(
            f"According to the model, the variable with the largest positive effect on calories is **{max_impact}**.")
        st.write("""
        - A positive coefficient means calories burned increase as the variable increases.
        - A p-value (P>|t|) above 0.05 means the variable is not a reliable predictor.
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
    st.markdown("""
        Our analysis of the Fitbit data led to the following conclusions, relevant both for the
        organisation's management and for end users:
        """)

    st.write("""
        1. Physical activity can be assessed through step counts and active minutes.
        2. High sedentary time may indicate a less healthy lifestyle.
        3. Sleep is an important factor in assessing lifestyle balance.
        4. Clustering split the users into distinct groups.
        5. Logistic and multiple regression help assess how activity, sleep and calories burned are related.
        """)

    # Conclusions grouped into two columns
    col_conc1, col_conc2 = st.columns(2)

    with col_conc1:
        st.subheader("Activity Analysis")
        st.success("**Activity level:** can be assessed accurately through the correlation between steps and active minutes.")
        st.success("**Segmentation:** clustering identified distinct groups (sedentary users vs. athletes).")
        st.success(
            "**Predictability:** multiple regression clearly showed how activity affects calories burned.")

    with col_conc2:
        st.subheader("Lifestyle")
        st.info("**Sedentary behaviour:** a high level points directly to a lifestyle that needs improvement.")
        st.info(
            "**Importance of sleep:** it is the deciding factor in assessing users' balance and recovery.")
        st.info(
            "**Predictive models:** logistic regression helps anticipate whether a user will behave healthily.")

    st.divider()

    # Data-driven recommendations
    st.subheader("Recommendations for the Organisation")

    with st.expander("Show the data-driven business proposals"):
        st.write("""
            1. **Personalised targeting:** use the clusters to send notifications tailored to each user type.
            2. **Gamification:** create challenges to reduce sedentary minutes (identified as a risk).
            3. **Focus on sleep:** encourage sleep tracking for a complete picture of health.
            """)
