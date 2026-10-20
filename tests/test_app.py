"""Smoke test: every section of the Streamlit app renders without an exception."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

# The map section downloads data from the internet, so it is not run here.
SECTIONS = [
    "1. Despre proiect",
    "2. Explorarea datelor",
    "3. Valori lipsă",
    "4. Valori extreme",
    "5. Statistici și agregări",
    "6. Codificare și scalare",
    "7. Clusterizare KMeans",
    "8. Regresie logistică",
    "9. Regresie multiplă",
    "11. Concluzii",
]


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(APP_PATH, default_timeout=60).run()


@pytest.mark.parametrize("section", SECTIONS)
def test_section_renders_without_errors(app, section):
    app.sidebar.selectbox[0].select(section).run()
    assert not app.exception
    assert not app.error
