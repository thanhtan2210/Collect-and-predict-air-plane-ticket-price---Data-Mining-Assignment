import numpy as np
import pandas as pd

from src.business_analysis import bootstrap_median_ci, cheapest_by_route
from src.predictor import load_findings


def test_bootstrap_ci_is_deterministic_and_brackets_the_median():
    values = [265, 808, 964, 2278, 2671, 3406, 3648, 4351, 5171, 6326]
    low, high = bootstrap_median_ci(values)
    assert (low, high) == bootstrap_median_ci(values)
    assert low <= np.median(values) <= high
    assert min(values) <= low and high <= max(values)


def test_bootstrap_ci_of_constant_values_has_no_width():
    assert bootstrap_median_ci([100.0] * 8) == (100.0, 100.0)


def test_cheapest_by_route_ignores_small_options():
    rows = (
        [{"Route": "A → B", "Airline": "Tiny", "Total_Stops": "non-stop", "Price": 10}] * 5
        + [{"Route": "A → B", "Airline": "Low", "Total_Stops": "non-stop", "Price": 100}] * 30
        + [{"Route": "A → B", "Airline": "High", "Total_Stops": "1 stop", "Price": 300}] * 30
    )
    result = cheapest_by_route(pd.DataFrame(rows)).iloc[0]
    assert result["cheapest_airline"] == "Low"
    assert result["option_flights"] == 30
    assert result["options_compared"] == 2
    assert result["route_flights"] == 65


def test_findings_file_is_readable_and_consistent():
    findings = load_findings()
    for key in ("stops_premium", "jet_fare_class"):
        result = findings[key]
        assert result["groups"] > 0
        assert result["ci95_low_inr"] <= result["median_diff_inr"] <= result["ci95_high_inr"]
    # Simpson reversal: dearer on average across airlines, cheaper like for like.
    fare = findings["jet_fare_class"]
    assert fare["all_airlines_raw"]["mean_diff_inr"] > 0
    assert fare["median_diff_inr"] < 0
    assert len(findings["cheapest_by_route"]) == 5
