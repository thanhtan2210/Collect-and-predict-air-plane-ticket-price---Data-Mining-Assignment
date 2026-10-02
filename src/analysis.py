"""Analysis behind docs/analysis.md: controlled comparisons, error analysis,
fare-condition effects and a time-based validation.

Run from the repo root (after src.train):  python -m src.analysis
Prints every table, writes models/analysis.json and docs/images/analysis_*.png.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.predictor import load_clean_data, load_metrics, load_pipeline
from src.preprocess import (
    ANALYSIS_PATH,
    BASE_DIR,
    DATE_FORMAT,
    FEATURES,
    TARGET,
    build_features,
    iqr_bounds,
)
from src.train import load_split, make_pipeline, make_xgb, regression_metrics

IMAGE_DIR = os.path.join(BASE_DIR, "docs", "images")
COLOR = "#2a78d6"
MIN_GROUP = 20  # minimum flights on each side of a within-group comparison

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def add_columns(clean):
    """Clean data plus the columns the comparisons group by."""
    df = clean.copy()
    date = pd.to_datetime(df["Date_of_Journey"], format=DATE_FORMAT)
    df["Route"] = df["Source"] + " → " + df["Destination"]
    df["Month"] = date.dt.month
    df["Weekend"] = date.dt.dayofweek >= 5
    return df


def controlled_effect(df, strata, column, treated, control, min_n=MIN_GROUP):
    """Price difference `treated` vs `control` within groups of like flights.

    Rows are grouped by `strata`; only groups with at least `min_n` flights
    on both sides are compared. Returns the raw (pooled) difference of
    medians, the per-group table, and the flight-weighted mean of the
    within-group differences.
    """
    sub = df[df[column].isin([treated, control])]
    raw = sub.groupby(column)[TARGET].median()
    table = sub.groupby(strata + [column])[TARGET].agg(["median", "count"]).unstack(column)
    table = table.dropna()
    table = table[(table["count"] >= min_n).all(axis=1)]

    out = pd.DataFrame(
        {
            "n_control": table[("count", control)].astype(int),
            "n_treated": table[("count", treated)].astype(int),
            "median_control": table[("median", control)],
            "median_treated": table[("median", treated)],
        }
    )
    out["diff_inr"] = out["median_treated"] - out["median_control"]
    out["diff_pct"] = (out["median_treated"] / out["median_control"] - 1) * 100
    weights = out["n_control"] + out["n_treated"]
    summary = {
        "raw_diff_inr": float(raw[treated] - raw[control]),
        "raw_diff_pct": float((raw[treated] / raw[control] - 1) * 100),
        "groups_compared": int(len(out)),
        "flights_compared": int(weights.sum()),
        "flights_total": int(len(sub)),
        "controlled_diff_inr": float(np.average(out["diff_inr"], weights=weights)) if len(out) else None,
        "controlled_diff_pct": float(np.average(out["diff_pct"], weights=weights)) if len(out) else None,
        "groups_where_treated_is_dearer": int((out["diff_inr"] > 0).sum()),
    }
    return summary, out.reset_index()


def print_effect(title, summary, table):
    print(f"\n{title}")
    print(f"  raw difference of medians: {summary['raw_diff_inr']:+,.0f} INR ({summary['raw_diff_pct']:+.1f}%)")
    if summary["groups_compared"]:
        print(
            f"  within like-for-like groups: {summary['controlled_diff_inr']:+,.0f} INR "
            f"({summary['controlled_diff_pct']:+.1f}%), {summary['groups_compared']} groups, "
            f"{summary['flights_compared']:,} of {summary['flights_total']:,} flights, "
            f"dearer in {summary['groups_where_treated_is_dearer']} groups"
        )
        print(table.round(1).to_string(index=False))
    else:
        print("  no group has enough flights on both sides")


def segment_errors(frame, by):
    """Error of the saved model on test rows, per segment."""
    grouped = frame.groupby(by, observed=True)
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "median_price": grouped["actual"].median(),
            "mae": grouped["abs_error"].mean(),
            "median_abs_pct_error": grouped["abs_pct_error"].median(),
            "bias": grouped["error"].mean(),
        }
    )
    return out.reset_index()


def noise_floor(clean):
    """How much prices differ between rows with identical model features."""
    features = build_features(clean)
    features[TARGET] = clean[TARGET].to_numpy()
    grouped = features.groupby(FEATURES)[TARGET]
    size, median = grouped.transform("size"), grouped.transform("median")
    spread = grouped.transform("max") - grouped.transform("min")
    repeated = size > 1
    return {
        "rows": int(len(features)),
        "rows_sharing_features": int(repeated.sum()),
        "rows_sharing_features_with_different_price": int((repeated & (spread > 0)).sum()),
        "mae_vs_group_median_on_those_rows": float(
            (features[TARGET] - median)[repeated & (spread > 0)].abs().mean()
        ),
    }


def temporal_validation(clean):
    """Train on March-May, test on June (same model and outlier rule)."""
    X, y = build_features(clean), clean[TARGET]
    train, test = X["Journey_Month"] <= 5, X["Journey_Month"] == 6
    low, high = iqr_bounds(y[train])
    keep = train & y.between(low, high)

    model = make_pipeline(make_xgb()).fit(X[keep], y[keep])
    y_pred = model.predict(X[test])
    in_range = (y[test] <= high).to_numpy()

    # Reference point: median price of the same route/airline/stops in training.
    key = ["Airline", "Source", "Destination", "Total_Stops"]
    lookup = X[keep].assign(price=y[keep]).groupby(key)["price"].median().rename("baseline")
    baseline = X[test].join(lookup, on=key)["baseline"].fillna(y[keep].median())

    return {
        "train_rows": int(keep.sum()),
        "test_rows": int(test.sum()),
        "price_iqr_high": high,
        "xgboost": {
            "full": regression_metrics(y[test], y_pred),
            "in_range": regression_metrics(y[test][in_range], y_pred[in_range]),
        },
        "group_median_baseline": {"full": regression_metrics(y[test], baseline)},
        "train_median_price": float(y[keep].median()),
        "test_median_price": float(y[test].median()),
    }


def random_split_baseline(X_train, y_train, X_test, y_test):
    """The same group-median reference on the random split used by src.train."""
    key = ["Airline", "Source", "Destination", "Total_Stops"]
    lookup = X_train.assign(price=y_train).groupby(key)["price"].median().rename("baseline")
    baseline = X_test.join(lookup, on=key)["baseline"].fillna(y_train.median())
    return regression_metrics(y_test, baseline)


def data_quality(raw_rows, clean):
    features = build_features(clean)
    expected = (features["Dep_Hour"] * 60 + features["Dep_Minute"] + features["Duration_Minutes"]) % 1440
    actual = features["Arrival_Hour"] * 60 + features["Arrival_Minute"]
    return {
        "raw_rows": raw_rows,
        "rows_removed_by_cleaning": raw_rows - int(len(clean)),
        "rows_where_arrival_differs_from_departure_plus_duration": int((expected != actual).sum()),
        "rows_with_duration_under_30_minutes": int((features["Duration_Minutes"] < 30).sum()),
        "airlines_with_under_30_flights": int((clean["Airline"].value_counts() < 30).sum()),
        "additional_info_values_with_under_30_flights": int(
            (clean["Additional_Info"].value_counts() < 30).sum()
        ),
    }


def barh(ax, labels, values, xlabel, fmt="{:,.0f}"):
    bars = ax.barh(labels, values, color=COLOR)
    ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=3, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.15)


def save(fig, name):
    os.makedirs(IMAGE_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGE_DIR, name), dpi=130)
    plt.close(fig)


def main():
    clean = load_clean_data()
    df = add_columns(clean)
    metrics = load_metrics()
    results = {}

    # 1. Controlled comparisons -------------------------------------------
    print("=" * 78 + "\n1. CONTROLLED COMPARISONS\n" + "=" * 78)
    like_for_like = ["Route", "Airline", "Total_Stops", "Month"]
    summary, table = controlled_effect(df, like_for_like, "Weekend", True, False)
    print_effect("Weekend vs weekday (same route, airline, stops and month)", summary, table)
    weights = table["n_control"] + table["n_treated"]
    by_month = [
        {
            "month": int(month),
            "groups": int(len(group)),
            "groups_where_weekend_is_dearer": int((group["diff_inr"] > 0).sum()),
            "controlled_diff_pct": float(np.average(group["diff_pct"], weights=weights[group.index])),
        }
        for month, group in table.groupby("Month")
    ]
    print("\n  The same within-group difference, split by month:")
    for row in by_month:
        print(
            f"    month {row['month']}: {row['controlled_diff_pct']:+.1f}% over {row['groups']} groups "
            f"(weekend dearer in {row['groups_where_weekend_is_dearer']})"
        )
    results["weekend"] = {"summary": summary, "by_month": by_month, "groups": table.to_dict("records")}

    summary, stops_table = controlled_effect(df, ["Route", "Airline"], "Total_Stops", "1 stop", "non-stop")
    print_effect("1 stop vs non-stop (same route and airline)", summary, stops_table)
    results["one_stop_vs_non_stop"] = {"summary": summary, "groups": stops_table.to_dict("records")}

    by_route = df.groupby(["Route", "Total_Stops"])[TARGET].agg(["median", "count"]).unstack("Total_Stops")
    print("\nFlights per route and stop count:")
    print(by_route["count"].fillna(0).astype(int).to_string())
    results["flights_by_route_and_stops"] = (
        by_route["count"].fillna(0).astype(int).reset_index().to_dict("records")
    )

    # 2. Fare conditions --------------------------------------------------
    print("\n" + "=" * 78 + "\n2. FARE CONDITIONS (Additional_Info)\n" + "=" * 78)
    results["additional_info"] = {}
    info_tables = {}
    for value in ["In-flight meal not included", "No check-in baggage included"]:
        summary, table = controlled_effect(
            df, ["Route", "Airline", "Total_Stops"], "Additional_Info", value, "No info"
        )
        print_effect(f"'{value}' vs 'No info' (same route, airline and stops)", summary, table)
        airlines = df.loc[df["Additional_Info"] == value, "Airline"].value_counts()
        print("  flights with this remark, by airline: " + ", ".join(f"{a} {n:,}" for a, n in airlines.items()))
        results["additional_info"][value] = {
            "summary": summary,
            "flights_by_airline": {a: int(n) for a, n in airlines.items()},
            "groups": table.to_dict("records"),
        }
        info_tables[value] = table

    # 3. Error analysis ---------------------------------------------------
    print("\n" + "=" * 78 + "\n3. ERROR ANALYSIS (saved model, hold-out test set)\n" + "=" * 78)
    X_train, y_train, X_test, y_test, info = load_split()
    pred = load_pipeline().predict(X_test)
    frame = df.loc[X_test.index, ["Airline", "Route", "Total_Stops", "Additional_Info"]].copy()
    frame["actual"], frame["predicted"] = y_test, pred
    frame["error"] = frame["predicted"] - frame["actual"]
    frame["abs_error"] = frame["error"].abs()
    frame["abs_pct_error"] = frame["abs_error"] / frame["actual"] * 100
    frame["Price band (INR)"] = pd.cut(
        frame["actual"],
        [0, 5000, 10000, 15000, info["price_iqr_high"], np.inf],
        labels=["< 5,000", "5,000-10,000", "10,000-15,000", "15,000-23,090", "> 23,090 (outliers)"],
    )
    results["errors"] = {}
    error_tables = {}
    for by in ["Route", "Airline", "Total_Stops", "Price band (INR)"]:
        table = segment_errors(frame, by)
        error_tables[by] = table
        results["errors"][by] = table.to_dict("records")
        print(f"\nBy {by}:")
        print(table.round(1).to_string(index=False))

    abs_error = frame["abs_error"].sort_values(ascending=False)
    worst = int(round(len(abs_error) * 0.05))
    concentration = {
        "median_abs_error": float(abs_error.median()),
        "mean_abs_error": float(abs_error.mean()),
        "share_within_500_inr": float((abs_error <= 500).mean()),
        "share_of_total_error_from_worst_5pct": float(abs_error.iloc[:worst].sum() / abs_error.sum()),
        "median_abs_pct_error": float(frame["abs_pct_error"].median()),
    }
    results["errors"]["concentration"] = concentration
    print(
        f"\nMedian absolute error {concentration['median_abs_error']:,.0f} INR vs mean "
        f"{concentration['mean_abs_error']:,.0f} INR; "
        f"{concentration['share_within_500_inr']:.1%} of predictions within 500 INR; worst 5% of "
        f"rows carry {concentration['share_of_total_error_from_worst_5pct']:.1%} of the total error; "
        f"median absolute percentage error {concentration['median_abs_pct_error']:.1f}%"
    )

    noise = noise_floor(clean)
    results["noise_floor"] = noise
    print(
        f"\nIdentical features, different price: {noise['rows_sharing_features_with_different_price']:,} "
        f"of {noise['rows']:,} rows share every model feature with another row yet differ in price; "
        f"on those rows even the group median is off by "
        f"{noise['mae_vs_group_median_on_those_rows']:,.0f} INR on average"
    )

    # 4. Time-based validation --------------------------------------------
    print("\n" + "=" * 78 + "\n4. TIME-BASED VALIDATION (train March-May, test June)\n" + "=" * 78)
    temporal = temporal_validation(clean)
    temporal["random_split"] = {
        "xgboost": metrics["test"],
        "group_median_baseline": {"full": random_split_baseline(X_train, y_train, X_test, y_test)},
    }
    results["temporal"] = temporal
    monthly = df.groupby("Month")[TARGET].agg(["count", "median"]).reset_index()
    results["price_by_month"] = monthly.to_dict("records")
    print(f"Train {temporal['train_rows']:,} rows (March-May), test {temporal['test_rows']:,} rows (June)")
    rows = [
        ("Random 80/20 split", temporal["random_split"]),
        ("Train Mar-May, test June", temporal),
    ]
    for label, block in rows:
        xgb, base = block["xgboost"]["full"], block["group_median_baseline"]["full"]
        print(
            f"  {label:<26} XGBoost R2 = {xgb['r2']:.4f}, MAE = {xgb['mae']:,.0f} INR | "
            f"route/airline/stops median R2 = {base['r2']:.4f}, MAE = {base['mae']:,.0f} INR"
        )
    june_outliers = temporal["test_rows"] - temporal["xgboost"]["in_range"]["n"]
    in_range = metrics["test"]["in_range"]
    print(
        f"  June has {june_outliers} rows above the outlier fence, so the like-for-like random-split "
        f"figure is the in-range one: R2 = {in_range['r2']:.4f}, MAE = {in_range['mae']:,.0f} INR"
    )
    print("\nFlights and median price by month:")
    print(monthly.to_string(index=False))

    # 5. Data quality -----------------------------------------------------
    print("\n" + "=" * 78 + "\n5. DATA QUALITY\n" + "=" * 78)
    quality = data_quality(metrics["dataset"]["raw_rows"], clean)
    results["data_quality"] = quality
    for key, value in quality.items():
        print(f"  {key}: {value:,}")

    # Figures -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(stops_table) + 1.5))
    labels = stops_table["Airline"] + ", " + stops_table["Route"]
    order = stops_table["diff_inr"].sort_values(ascending=False).index
    barh(ax, labels[order], stops_table["diff_inr"][order], "Median price of 1 stop minus non-stop (INR)",
         "{:+,.0f}")
    ax.set_title("Extra cost of one stop, same airline and route", loc="left", fontsize=11)
    save(fig, "analysis_stops.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, by, title in zip(axes, ["Route", "Price band (INR)"], ["route", "actual price (INR)"]):
        table = error_tables[by]
        barh(ax, table[by].astype(str), table["mae"], "Test MAE (INR)")
        ax.set_title(f"Test error by {title}", loc="left", fontsize=11)
    save(fig, "analysis_errors.png")

    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    labels = ["Random split\nXGBoost", "Random split\ngroup median", "June hold-out\nXGBoost",
              "June hold-out\ngroup median"]
    values = [
        temporal["random_split"]["xgboost"]["full"]["mae"],
        temporal["random_split"]["group_median_baseline"]["full"]["mae"],
        temporal["xgboost"]["full"]["mae"],
        temporal["group_median_baseline"]["full"]["mae"],
    ]
    barh(ax, labels, values, "Test MAE (INR)")
    ax.set_title("Interpolating within the period vs predicting the next month", loc="left", fontsize=11)
    save(fig, "analysis_temporal.png")

    with open(ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    print(f"\nSaved {ANALYSIS_PATH} and figures in {IMAGE_DIR}")


if __name__ == "__main__":
    main()
