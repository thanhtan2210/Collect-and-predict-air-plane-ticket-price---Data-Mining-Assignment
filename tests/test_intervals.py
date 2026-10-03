import numpy as np
import pytest

from src.predictor import (
    CHEAP,
    EXPENSIVE,
    FAIR,
    assess_quote,
    known_categories,
    known_routes,
    load_interval_metrics,
    load_intervals,
    quote_label,
)
from src.train import load_split
from src.train_intervals import conformal_adjustment, predict_interval

INPUTS = dict(
    airline="IndiGo",
    source="Banglore",
    destination="Delhi",
    journey_date="24/03/2019",
    dep_time="22:20",
    duration_minutes=170,
    total_stops=0,
)


def test_interval_bounds_never_cross_on_test_set():
    _, _, X_test, _, _ = load_split()
    lower, upper = predict_interval(load_intervals(), X_test)
    assert (lower <= upper).all()


def test_assess_quote_without_offer():
    quote = assess_quote(INPUTS)
    assert 0 <= quote["low"] <= quote["high"]
    assert "label" not in quote
    assert quote["low"] <= quote["predicted"] <= quote["high"]


def test_displayed_price_is_always_inside_the_interval():
    """The raw point prediction may fall outside; the displayed price may not."""
    routes = known_routes()
    outside = 0
    for airline in known_categories()["Airline"]:
        for source, destinations in routes.items():
            for stops, duration in [(0, 150), (1, 600), (2, 1200), (3, 1800)]:
                inputs = dict(INPUTS, airline=airline, source=source,
                              destination=destinations[0], total_stops=stops,
                              duration_minutes=duration)
                quote = assess_quote(inputs)
                assert quote["low"] <= quote["predicted"] <= quote["high"]
                clamped = min(max(quote["point_prediction"], quote["low"]), quote["high"])
                assert quote["predicted"] == clamped
                outside += quote["point_prediction"] != quote["predicted"]
    # The grid includes unusual flights, so the clamp is actually exercised.
    assert outside > 0


def test_quote_label_boundaries():
    assert quote_label(99, 100, 200) == CHEAP
    assert quote_label(100, 100, 200) == FAIR
    assert quote_label(150, 100, 200) == FAIR
    assert quote_label(200, 100, 200) == FAIR
    assert quote_label(201, 100, 200) == EXPENSIVE


def test_assess_quote_labels_and_gap():
    base = assess_quote(INPUTS)
    cheap = assess_quote(INPUTS, base["low"] - 1)
    fair = assess_quote(INPUTS, base["predicted"])
    expensive = assess_quote(INPUTS, base["high"] + 1)
    assert (cheap["label"], fair["label"], expensive["label"]) == (CHEAP, FAIR, EXPENSIVE)
    assert fair["difference"] == pytest.approx(0)
    assert fair["difference_pct"] == pytest.approx(0)
    assert expensive["difference"] == pytest.approx(base["high"] + 1 - base["predicted"])
    assert cheap["difference_pct"] < 0


def test_conformal_adjustment_widens_until_target_coverage():
    y = np.arange(100.0)
    lower, upper = np.full(100, 40.0), np.full(100, 60.0)  # covers 21% of y
    margin = conformal_adjustment(lower, upper, y, coverage=0.8)
    covered = ((y >= lower - margin) & (y <= upper + margin)).mean()
    assert margin > 0
    assert 0.8 <= covered <= 0.85


def test_interval_metrics_are_readable_and_on_target():
    metrics = load_interval_metrics()
    assert metrics["target_coverage"] == pytest.approx(0.8)
    assert 0.75 <= metrics["test"]["coverage"] <= 0.85
    assert len(metrics["test_by_price_quartile"]) == 4
    widths = [band["mean_width"] for band in metrics["test_by_price_quartile"]]
    assert widths[-1] > widths[0]
    assert 0 <= metrics["point_prediction_inside_interval"] <= 1
