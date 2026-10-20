"""Smoke test: every section of the Streamlit app renders without an exception."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

SECTIONS = [
    "1. About the project",
    "2. Data exploration",
    "3. Missing values",
    "4. Outliers",
    "5. Statistics and aggregations",
    "6. Encoding and scaling",
    "7. K-Means clustering",
    "8. Logistic regression",
    "9. Multiple regression",
    "10. Data coverage",
    "11. Conclusions",
]


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(APP_PATH, default_timeout=60).run()


@pytest.mark.parametrize("section", SECTIONS)
def test_section_renders_without_errors(app, section):
    app.sidebar.selectbox[0].select(section).run()
    assert not app.exception
    assert not app.error
