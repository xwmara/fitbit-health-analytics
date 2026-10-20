"""Smoke test: every page of the Streamlit app renders without an exception."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP_PATH = str(ROOT / "app.py")

# Every page file registered in app.py
PAGES = sorted(
    f"app_pages/{p.name}" for p in (ROOT / "app_pages").glob("*.py")
    if p.name not in ("__init__.py", "common.py")
)


def test_all_pages_are_registered_in_navigation():
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    for page in PAGES:
        assert f'"{page}"' in app_source, f"{page} is not in the navigation"


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors(page):
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    app.switch_page(page).run()
    assert not app.exception
    assert not app.error
