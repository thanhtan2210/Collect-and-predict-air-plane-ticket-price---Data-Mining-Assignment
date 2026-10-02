"""Load the trained pipeline and predict prices from user input."""
import json
import os
from functools import lru_cache

import joblib

from src.preprocess import (
    ANALYSIS_PATH,
    CATEGORICAL_FEATURES,
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
