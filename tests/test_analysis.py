import pandas as pd

from src.analysis import controlled_effect


def test_controlled_effect_removes_mix_effect():
    """Simpson's paradox: 'cheap' fares look dearer overall, cheaper per airline."""
    rows = (
        [{"Airline": "Premium", "Fare": "basic", "Price": 900}] * 30
        + [{"Airline": "Premium", "Fare": "standard", "Price": 1000}] * 5
        + [{"Airline": "Budget", "Fare": "basic", "Price": 90}] * 5
        + [{"Airline": "Budget", "Fare": "standard", "Price": 100}] * 30
    )
    summary, table = controlled_effect(
        pd.DataFrame(rows), ["Airline"], "Fare", "basic", "standard", min_n=5
    )
    assert summary["raw_diff_pct"] > 0
    assert round(summary["controlled_diff_pct"], 6) == -10.0
    assert summary["groups_compared"] == 2
    assert summary["groups_where_treated_is_dearer"] == 0


def test_controlled_effect_skips_small_groups():
    rows = (
        [{"Airline": "A", "Fare": "basic", "Price": 90}] * 10
        + [{"Airline": "A", "Fare": "standard", "Price": 100}] * 10
        + [{"Airline": "B", "Fare": "basic", "Price": 500}] * 2
        + [{"Airline": "B", "Fare": "standard", "Price": 100}] * 10
    )
    summary, table = controlled_effect(
        pd.DataFrame(rows), ["Airline"], "Fare", "basic", "standard", min_n=5
    )
    assert list(table["Airline"]) == ["A"]
    assert summary["flights_compared"] == 20
