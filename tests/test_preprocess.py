import datetime

import pandas as pd
import pytest

from src.preprocess import (
    FEATURES,
    build_features,
    clean_raw,
    format_duration,
    iqr_bounds,
    make_raw_row,
    parse_duration,
)

SAMPLE = {
    "Airline": "IndiGo",
    "Date_of_Journey": "24/03/2019",
    "Source": "Banglore",
    "Destination": "New Delhi",
    "Route": "BLR → DEL",
    "Dep_Time": "22:20",
    "Arrival_Time": "01:10 22 Mar",
    "Duration": "2h 50m",
    "Total_Stops": "non-stop",
    "Additional_Info": "No info",
    "Price": 3897,
}


def sample_frame(**overrides):
    return pd.DataFrame([{**SAMPLE, **overrides}])


@pytest.mark.parametrize(
    "text, minutes",
    [("2h 50m", 170), ("19h", 1140), ("5m", 5), (" 1h 5m ", 65), ("47h 40m", 2860)],
)
def test_parse_duration(text, minutes):
    assert parse_duration(text) == minutes


@pytest.mark.parametrize("text", ["", "abc", "2 hours", "h m"])
def test_parse_duration_rejects_garbage(text):
    with pytest.raises(ValueError):
        parse_duration(text)


@pytest.mark.parametrize("text", ["2h 50m", "19h", "5m", "1h 5m"])
def test_format_duration_round_trip(text):
    assert format_duration(parse_duration(text)) == text


@pytest.mark.parametrize("minutes", [5, 60, 170, 1140, 2860])
def test_parse_format_round_trip(minutes):
    assert parse_duration(format_duration(minutes)) == minutes


def test_clean_raw_normalises_labels_and_drops_duplicates():
    df = pd.concat(
        [
            sample_frame(),
            sample_frame(),  # exact duplicate
            sample_frame(Destination="Delhi"),  # duplicate once normalised
            sample_frame(Airline=" Air India ", Additional_Info="No Info"),
            sample_frame(Total_Stops=None),
        ],
        ignore_index=True,
    )
    clean = clean_raw(df)
    assert len(clean) == 2
    assert set(clean["Destination"]) == {"Delhi"}
    assert set(clean["Additional_Info"]) == {"No info"}
    assert list(clean["Airline"]) == ["IndiGo", "Air India"]


def test_build_features_on_sample_row():
    features = build_features(clean_raw(sample_frame()))
    assert list(features.columns) == FEATURES
    assert features.iloc[0].to_dict() == {
        "Airline": "IndiGo",
        "Source": "Banglore",
        "Destination": "Delhi",
        "Additional_Info": "No info",
        "Total_Stops": 0,
        "Journey_Day": 24,
        "Journey_Month": 3,
        "Day_Of_Week": 6,  # 24/03/2019 is a Sunday
        "Is_Weekend": 1,
        "Dep_Hour": 22,
        "Dep_Minute": 20,
        "Arrival_Hour": 1,
        "Arrival_Minute": 10,
        "Duration_Minutes": 170,
    }


def test_serving_features_match_training_features():
    """The key guarantee: user input goes through the same code as training."""
    train_features = build_features(clean_raw(sample_frame()))
    serve_features = build_features(
        make_raw_row(
            airline="IndiGo",
            source="Banglore",
            destination="Delhi",
            journey_date=datetime.date(2019, 3, 24),
            dep_time=datetime.time(22, 20),
            duration_minutes=170,
            total_stops=0,
        )
    )
    pd.testing.assert_frame_equal(serve_features, train_features)
    assert serve_features["Duration_Minutes"].iloc[0] == 170


def test_make_raw_row_accepts_strings():
    from_strings = make_raw_row("IndiGo", "Banglore", "Delhi", "24/03/2019", "22:20", 170, "non-stop")
    from_objects = make_raw_row(
        "IndiGo", "Banglore", "Delhi", datetime.date(2019, 3, 24), datetime.time(22, 20), 170, 0
    )
    pd.testing.assert_frame_equal(from_strings, from_objects)


def test_unknown_stops_raise():
    with pytest.raises(ValueError):
        build_features(sample_frame(Total_Stops="5 stops"))
    with pytest.raises(ValueError):
        make_raw_row("IndiGo", "Banglore", "Delhi", "24/03/2019", "22:20", 170, 7)


def test_iqr_bounds():
    low, high = iqr_bounds(pd.Series([1, 2, 3, 4, 5]))
    assert (low, high) == (-1.0, 7.0)
