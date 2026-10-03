import os

import pytest
from streamlit.testing.v1 import AppTest

from src.predictor import CHEAP, EXPENSIVE, FAIR

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TABS = ["Predict", "Market insights", "Business findings", "Model performance"]


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    return at


def by_label(widgets, label):
    return next(w for w in widgets if w.label == label)


def test_app_runs_without_exception(app):
    assert not app.exception


def test_app_has_four_tabs(app):
    assert [tab.label for tab in app.tabs] == TABS


def test_quote_gets_a_label(app):
    by_label(app.selectbox, "From").set_value("Banglore")
    app.run()
    by_label(app.selectbox, "To").set_value("Delhi")
    by_label(app.number_input, "Offered price (INR)").set_value(5000)
    app.run()
    assert not app.exception

    verdicts = [box.value for box in [*app.success, *app.info, *app.warning]]
    labelled = [text for text in verdicts if text.startswith((f"**{CHEAP}**", f"**{FAIR}**", f"**{EXPENSIVE}**"))]
    assert len(labelled) == 1
