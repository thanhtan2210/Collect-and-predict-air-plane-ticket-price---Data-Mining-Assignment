"""Load the trained pipeline and predict prices from user input."""
import json
import os
from functools import lru_cache

import joblib

from src.preprocess import (
    ANALYSIS_PATH,
    CATEGORICAL_FEATURES,
    INTERVAL_METRICS_PATH,
    INTERVAL_MODEL_PATH,
    METRICS_PATH,
    MODEL_PATH,
    build_features,
    clean_raw,
    load_raw,
    make_raw_row,
)

_NOT_TRAINED = "{} not found. Train the model first: python -m src.train"


@lru_cache(maxsize=1)
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(_NOT_TRAINED.format(MODEL_PATH))
    return joblib.load(MODEL_PATH)


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(_NOT_TRAINED.format(METRICS_PATH))
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_intervals():
    """Quantile models and CQR margin written by `python -m src.train_intervals`."""
    if not os.path.exists(INTERVAL_MODEL_PATH):
        raise FileNotFoundError(
            f"{INTERVAL_MODEL_PATH} not found. Run: python -m src.train_intervals"
        )
    return joblib.load(INTERVAL_MODEL_PATH)


def load_interval_metrics():
    if not os.path.exists(INTERVAL_METRICS_PATH):
        raise FileNotFoundError(
            f"{INTERVAL_METRICS_PATH} not found. Run: python -m src.train_intervals"
        )
    with open(INTERVAL_METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_analysis():
    """Output of `python -m src.analysis`, or None if it has not been run."""
    if not os.path.exists(ANALYSIS_PATH):
        return None
    with open(ANALYSIS_PATH, encoding="utf-8") as f:
        return json.load(f)


def known_categories():
    """Categories seen by the fitted OneHotEncoder, per categorical feature."""
    encoder = load_pipeline().named_steps["preprocess"].named_transformers_["cat"]
    return {
        col: [str(c) for c in categories]
        for col, categories in zip(CATEGORICAL_FEATURES, encoder.categories_)
    }


def load_clean_data():
    return clean_raw(load_raw())


def known_routes():
    """Map each source city to the destinations that exist in the data."""
    pairs = load_clean_data()[["Source", "Destination"]].drop_duplicates()
    return {
        source: sorted(group["Destination"])
        for source, group in pairs.groupby("Source")
    }


def predict_price(
    airline,
    source,
    destination,
    journey_date,
    dep_time,
    duration_minutes,
    total_stops,
    additional_info="No info",
):
    """Predicted price in INR. Features always come from `build_features`."""
    row = make_raw_row(
        airline, source, destination, journey_date, dep_time,
        duration_minutes, total_stops, additional_info,
    )
    prediction = load_pipeline().predict(build_features(row))[0]
    return max(0.0, float(prediction))


CHEAP, FAIR, EXPENSIVE = "Cheap", "Fair", "Expensive"


def quote_label(offered_price, low, high):
    """Cheap below the interval, Expensive above it, Fair inside (bounds included)."""
    if offered_price < low:
        return CHEAP
    if offered_price > high:
        return EXPENSIVE
    return FAIR


def assess_quote(inputs, offered_price=None):
    """Judge a quoted price against comparable flights in the 2019 data.

    `inputs` holds the arguments of `predict_price`. Returns the point
    prediction, the calibrated 80% interval [low, high] and, when
    `offered_price` is given, a label plus the gap to the prediction.

    The point prediction and the interval come from separate models, so the
    prediction is usually, but not always, inside the interval.
    """
    features = build_features(make_raw_row(**inputs))
    predicted = max(0.0, float(load_pipeline().predict(features)[0]))

    models = load_intervals()
    margin = models["conformal_adjustment"]
    bounds = sorted(float(models[key].predict(features)[0]) for key in ("lower", "upper"))
    low, high = sorted([max(0.0, bounds[0] - margin), max(0.0, bounds[1] + margin)])

    result = {
        "predicted": predicted,
        "low": low,
        "high": high,
        "coverage": models["target_coverage"],
    }
    if offered_price is not None:
        offered_price = float(offered_price)
        result.update(
            offered=offered_price,
            label=quote_label(offered_price, low, high),
            difference=offered_price - predicted,
            difference_pct=(offered_price / predicted - 1) * 100 if predicted else None,
        )
    return result
