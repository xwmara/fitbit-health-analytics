"""Entry point of the Streamlit app: page setup, navigation and sidebar credits.

Run with:  streamlit run app.py
Each page lives in its own file in app_pages/.
"""

import streamlit as st

from fitbit_analytics import charts

AUTHOR_NAME = "Mara-Livia Popa"
GITHUB_URL = "https://github.com/xwmara"
LINKEDIN_URL = "https://www.linkedin.com/in/mara-livia-popa-99ba18337/"

# Page configuration (colours and fonts are set in .streamlit/config.toml)
st.set_page_config(page_title="Fitbit Health Analytics", page_icon="📊", layout="wide")
charts.apply_style()  # one Matplotlib style for every chart in the app

# Pages grouped in the sidebar menu; each page gets its own URL (e.g. /clustering)
PAGES = {
    "Overview": [
        st.Page("app_pages/overview.py", title="Overview", icon=":material/home:", default=True),
        st.Page("app_pages/coverage.py", title="Data coverage", icon=":material/calendar_month:"),
    ],
    "Data": [
        st.Page("app_pages/exploration.py", title="Data exploration", icon=":material/table_view:"),
        st.Page("app_pages/missing_values.py", title="Missing values", icon=":material/help:"),
        st.Page("app_pages/outliers.py", title="Outliers", icon=":material/scatter_plot:"),
        st.Page("app_pages/statistics.py", title="Statistics", icon=":material/bar_chart:"),
        st.Page("app_pages/encoding.py", title="Encoding and scaling", icon=":material/tune:"),
    ],
    "Models": [
        st.Page("app_pages/clustering.py", title="Clustering", icon=":material/bubble_chart:"),
        st.Page("app_pages/logistic.py", title="Healthy-day classifier", icon=":material/verified:"),
        st.Page("app_pages/regression.py", title="Calories regression", icon=":material/trending_up:"),
    ],
    "Results": [
        st.Page("app_pages/conclusions.py", title="Conclusions", icon=":material/lightbulb:"),
    ],
}

page = st.navigation(PAGES)

with st.sidebar:
    st.caption(f"Built by **{AUTHOR_NAME}**  \n[GitHub]({GITHUB_URL}) · [LinkedIn]({LINKEDIN_URL})")

page.run()
