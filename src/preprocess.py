"""Preprocessing shared by training and prediction.

Every feature the model sees is produced by `build_features`, both when
training on the raw CSV and when predicting from user input (through
`make_raw_row`). Keeping a single code path avoids train/serve skew.
"""
import datetime
import os
import re

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "IndianFlightdata - Sheet1.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "flight_price_pipeline.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

TARGET = "Price"
CATEGORICAL_FEATURES = ["Airline", "Source", "Destination", "Additional_Info"]
NUMERIC_FEATURES = [
    "Total_Stops",
    "Journey_Day",
    "Journey_Month",
    "Day_Of_Week",
    "Is_Weekend",
    "Dep_Hour",
    "Dep_Minute",
    "Arrival_Hour",
    "Arrival_Minute",
    "Duration_Minutes",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
STOPS_MAP = {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}

RAW_COLUMNS = [
    "Airline",
    "Date_of_Journey",
    "Source",
    "Destination",
    "Route",
    "Dep_Time",
    "Arrival_Time",
    "Duration",
    "Total_Stops",
    "Additional_Info",
    "Price",
]
DATE_FORMAT = "%d/%m/%Y"

_DURATION_RE = re.compile(r"^(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?$")
_STOPS_LABELS = {v: k for k, v in STOPS_MAP.items()}


def parse_duration(text):
    """Convert "2h 50m", "19h" or "5m" to minutes."""
    match = _DURATION_RE.match(str(text).strip().lower())
    if not match or not any(match.groups()):
        raise ValueError(f"Invalid duration: {text!r} (expected e.g. '2h 50m')")
    hours, minutes = (int(g) if g else 0 for g in match.groups())
    return hours * 60 + minutes


def format_duration(minutes):
    """Convert minutes back to the CSV format: 170 -> "2h 50m"."""
    minutes = int(minutes)
    if minutes <= 0:
        raise ValueError(f"Duration must be positive, got {minutes}")
    hours, rest = divmod(minutes, 60)
    if hours and rest:
        return f"{hours}h {rest}m"
    return f"{hours}h" if hours else f"{rest}m"


def load_raw(path=DATA_PATH):
    return pd.read_csv(path, encoding="utf-8")


def clean_raw(df):
    """Normalise labels, then drop missing rows and exact duplicates.

    Must run BEFORE the train/test split so the same flight cannot end up
    on both sides.
    """
    df = df.copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].str.strip()
    df["Destination"] = df["Destination"].replace({"New Delhi": "Delhi"})
    df["Additional_Info"] = df["Additional_Info"].replace({"No Info": "No info"})
    df = df.dropna(subset=[c for c in RAW_COLUMNS if c in df.columns])
    return df.drop_duplicates().reset_index(drop=True)


def build_features(df):
    """Turn rows in the raw CSV layout into the model's feature columns."""
    out = pd.DataFrame(index=df.index)
    for col in CATEGORICAL_FEATURES:
        out[col] = df[col].astype(str).str.strip()

    stops = df["Total_Stops"].map(STOPS_MAP)
    if stops.isna().any():
        unknown = sorted(df.loc[stops.isna(), "Total_Stops"].astype(str).unique())
        raise ValueError(f"Unknown Total_Stops values: {unknown}")
    out["Total_Stops"] = stops.astype(int)

    date = pd.to_datetime(df["Date_of_Journey"], format=DATE_FORMAT)
    out["Journey_Day"] = date.dt.day
    out["Journey_Month"] = date.dt.month
    out["Day_Of_Week"] = date.dt.dayofweek
    out["Is_Weekend"] = (date.dt.dayofweek >= 5).astype(int)

    dep = df["Dep_Time"].str.strip().str.split(":", expand=True)
    out["Dep_Hour"] = dep[0].astype(int)
    out["Dep_Minute"] = dep[1].astype(int)

    # Arrival_Time may carry a date suffix ("01:10 22 Mar"): keep the clock part.
    arrival = df["Arrival_Time"].str.strip().str.split().str[0].str.split(":", expand=True)
    out["Arrival_Hour"] = arrival[0].astype(int)
    out["Arrival_Minute"] = arrival[1].astype(int)

    out["Duration_Minutes"] = df["Duration"].map(parse_duration).astype(int)
    return out[FEATURES]


def make_raw_row(
    airline,
    source,
    destination,
    journey_date,
    dep_time,
    duration_minutes,
    total_stops,
    additional_info="No info",
):
    """Build a one-row dataframe in the raw CSV layout from user input.

    `journey_date` is a date or a "DD/MM/YYYY" string, `dep_time` a time or
    an "HH:MM" string, `total_stops` a count (0-4) or a CSV label ("1 stop").
    Arrival time is departure + duration, modulo 24h.
    """
    if isinstance(journey_date, str):
        journey_date = datetime.datetime.strptime(journey_date.strip(), DATE_FORMAT).date()
    if isinstance(dep_time, str):
        dep_time = datetime.datetime.strptime(dep_time.strip(), "%H:%M").time()
    if not isinstance(total_stops, str):
        if int(total_stops) not in _STOPS_LABELS:
            raise ValueError(f"Unknown Total_Stops values: [{total_stops!r}]")
        total_stops = _STOPS_LABELS[int(total_stops)]

    duration_minutes = int(duration_minutes)
    arrival = (dep_time.hour * 60 + dep_time.minute + duration_minutes) % (24 * 60)
    return pd.DataFrame(
        [
            {
                "Airline": airline,
                "Date_of_Journey": journey_date.strftime(DATE_FORMAT),
                "Source": source,
                "Destination": destination,
                "Route": "",
                "Dep_Time": f"{dep_time.hour:02d}:{dep_time.minute:02d}",
                "Arrival_Time": f"{arrival // 60:02d}:{arrival % 60:02d}",
                "Duration": format_duration(duration_minutes),
                "Total_Stops": total_stops,
                "Additional_Info": additional_info,
            }
        ]
    )


def iqr_bounds(series, k=1.5):
    """Return (low, high) Tukey fences of a numeric series."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)
