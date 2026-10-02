"""Findings expressed in money, with confounders held constant.

A case study of the 2019 data, written for a travel-agency user: what a
connection costs, what the cheaper Jet Airways fare class saves, and which
option is cheapest on each route.

Run from the repo root:  python -m src.business_analysis
Writes reports/findings.json and three CSV tables in reports/.
"""
import json
import os

import numpy as np
import pandas as pd

from src.analysis import add_columns, controlled_effect
from src.predictor import load_clean_data
from src.preprocess import FINDINGS_PATH, REPORT_DIR, TARGET

MIN_PER_SIDE = 20  # flights on each side of a within-group comparison
MIN_OPTION = 30  # flights for an airline/stops option to be ranked on a route
BOOTSTRAP_SAMPLES = 10_000
SEED = 42

MEAL = "In-flight meal not included"
STANDARD = "No info"


def bootstrap_median_ci(values, samples=BOOTSTRAP_SAMPLES, seed=SEED):
    """95% percentile interval for the median of group-level differences.

    Whole groups are resampled with replacement, so the interval reflects
    how much the answer depends on which groups happen to be comparable.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True)
    low, high = np.percentile(np.median(draws, axis=1), [2.5, 97.5])
    return float(low), float(high)


def summarise(summary, table):
    """Median within-group difference with its bootstrap interval."""
    low, high = bootstrap_median_ci(table["diff_inr"])
    return {
        "groups": int(len(table)),
        "flights_compared": summary["flights_compared"],
        "flights_total": summary["flights_total"],
        "median_diff_inr": float(table["diff_inr"].median()),
        "ci95_low_inr": low,
        "ci95_high_inr": high,
        "ci_excludes_zero": bool(low > 0 or high < 0),
        "median_diff_pct": float(table["diff_pct"].median()),
        "min_diff_inr": float(table["diff_inr"].min()),
        "max_diff_inr": float(table["diff_inr"].max()),
        "groups_with_positive_diff": int((table["diff_inr"] > 0).sum()),
        "raw_diff_inr": summary["raw_diff_inr"],
        "raw_diff_pct": summary["raw_diff_pct"],
    }


def stops_premium(df):
    """1 stop vs non-stop within the same airline and route."""
    summary, table = controlled_effect(
        df, ["Route", "Airline"], "Total_Stops", "1 stop", "non-stop", min_n=MIN_PER_SIDE
    )
    table = table.rename(
        columns={
            "n_control": "n_non_stop",
            "n_treated": "n_one_stop",
            "median_control": "median_non_stop",
            "median_treated": "median_one_stop",
        }
    )
    return summarise(summary, table), table


def jet_fare_class(df):
    """Jet Airways only: 'meal not included' vs standard fare, same route and stops."""
    jet = df[df["Airline"] == "Jet Airways"]
    summary, table = controlled_effect(
        jet, ["Route", "Total_Stops"], "Additional_Info", MEAL, STANDARD, min_n=MIN_PER_SIDE
    )
    table = table.rename(
        columns={
            "n_control": "n_standard",
            "n_treated": "n_no_meal",
            "median_control": "median_standard",
            "median_treated": "median_no_meal",
        }
    )
    result = summarise(summary, table)

    # The same comparison without any control, to show the reversal caused
    # by airline mix (confounding).
    both = df[df["Additional_Info"].isin([MEAL, STANDARD])]
    by_remark = both.groupby("Additional_Info")[TARGET].agg(["count", "median", "mean"])
    result["all_airlines_raw"] = {
        "n_no_meal": int(by_remark.loc[MEAL, "count"]),
        "n_standard": int(by_remark.loc[STANDARD, "count"]),
        "median_diff_inr": float(by_remark.loc[MEAL, "median"] - by_remark.loc[STANDARD, "median"]),
        "mean_diff_inr": float(by_remark.loc[MEAL, "mean"] - by_remark.loc[STANDARD, "mean"]),
    }
    jet_by_remark = jet[jet["Additional_Info"].isin([MEAL, STANDARD])].groupby(
        "Additional_Info")[TARGET].agg(["count", "median", "mean"])
    result["jet_airways_raw"] = {
        "n_no_meal": int(jet_by_remark.loc[MEAL, "count"]),
        "n_standard": int(jet_by_remark.loc[STANDARD, "count"]),
        "median_diff_inr": float(jet_by_remark.loc[MEAL, "median"] - jet_by_remark.loc[STANDARD, "median"]),
        "mean_diff_inr": float(jet_by_remark.loc[MEAL, "mean"] - jet_by_remark.loc[STANDARD, "mean"]),
    }
    result["no_meal_flights_by_airline"] = {
        airline: int(n) for airline, n in df.loc[df["Additional_Info"] == MEAL, "Airline"].value_counts().items()
    }
    return result, table


def cheapest_by_route(df):
    """Per route: the airline + stops option with the lowest median price."""
    rows = []
    for route, flights in df.groupby("Route"):
        options = flights.groupby(["Airline", "Total_Stops"])[TARGET].agg(
            n="count", median="median",
            q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
        )
        options = options[options["n"] >= MIN_OPTION].sort_values("median")
        best = options.iloc[0]
        price = flights[TARGET]
        rows.append(
            {
                "Route": route,
                "cheapest_airline": best.name[0],
                "cheapest_stops": best.name[1],
                "option_flights": int(best["n"]),
                "option_median": float(best["median"]),
                "option_q25": float(best["q25"]),
                "option_q75": float(best["q75"]),
                "options_compared": int(len(options)),
                "dearest_option_median": float(options["median"].iloc[-1]),
                "route_flights": int(len(price)),
                "route_median": float(price.median()),
                "route_q25": float(price.quantile(0.25)),
                "route_q75": float(price.quantile(0.75)),
                "route_cv": float(price.std() / price.mean()),
                "saving_vs_route_median": float(price.median() - best["median"]),
            }
        )
    return pd.DataFrame(rows)


def print_summary(title, result):
    print(f"\n{title}")
    print(
        f"  median within-group difference: {result['median_diff_inr']:+,.0f} INR "
        f"({result['median_diff_pct']:+.1f}%), 95% CI [{result['ci95_low_inr']:+,.0f}, "
        f"{result['ci95_high_inr']:+,.0f}] INR"
        f"{'' if result['ci_excludes_zero'] else '  <- interval contains 0'}"
    )
    print(
        f"  {result['groups']} groups, {result['flights_compared']:,} of "
        f"{result['flights_total']:,} flights; range {result['min_diff_inr']:+,.0f} to "
        f"{result['max_diff_inr']:+,.0f} INR; positive in {result['groups_with_positive_diff']} groups"
    )


def main():
    df = add_columns(load_clean_data())
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("Case study of fares for March-June 2019. Not advice on current prices.")

    stops, stops_table = stops_premium(df)
    print_summary("1. One stop vs non-stop, same airline and route", stops)
    print(f"  raw difference of medians (no control): {stops['raw_diff_inr']:+,.0f} INR")
    print(stops_table.round(1).to_string(index=False))

    fare, fare_table = jet_fare_class(df)
    print_summary(
        "2. Jet Airways fare class: 'meal not included' vs standard, same route and stops", fare
    )
    for label, raw in [("all airlines", fare["all_airlines_raw"]), ("Jet Airways only", fare["jet_airways_raw"])]:
        print(
            f"  raw, {label}: median {raw['median_diff_inr']:+,.0f} INR, mean "
            f"{raw['mean_diff_inr']:+,.0f} INR ({raw['n_no_meal']:,} no-meal vs "
            f"{raw['n_standard']:,} standard)"
        )
    print(f"  no-meal flights by airline: {fare['no_meal_flights_by_airline']}")
    print(fare_table.round(1).to_string(index=False))

    cheapest = cheapest_by_route(df)
    print(f"\n3. Cheapest airline + stops option per route (options with >= {MIN_OPTION} flights)")
    print(cheapest.round(2).to_string(index=False))

    stops_table.to_csv(os.path.join(REPORT_DIR, "stops_premium_by_group.csv"), index=False, encoding="utf-8")
    fare_table.to_csv(os.path.join(REPORT_DIR, "jet_fare_class_by_group.csv"), index=False, encoding="utf-8")
    cheapest.to_csv(os.path.join(REPORT_DIR, "cheapest_option_by_route.csv"), index=False, encoding="utf-8")
    findings = {
        "scope": "Case study of fares for March-June 2019 on 5 Indian domestic routes.",
        "min_flights_per_side": MIN_PER_SIDE,
        "min_flights_per_option": MIN_OPTION,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "stops_premium": stops,
        "jet_fare_class": fare,
        "cheapest_by_route": cheapest.to_dict("records"),
    }
    with open(FINDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nSaved {FINDINGS_PATH} and 3 CSV tables in {REPORT_DIR}")


if __name__ == "__main__":
    main()
