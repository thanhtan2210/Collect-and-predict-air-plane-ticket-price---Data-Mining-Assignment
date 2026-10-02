"""Streamlit app. Run from the repo root:  streamlit run app.py"""
import datetime
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src import predictor
from src.preprocess import DATE_FORMAT, REPORT_DIR, STOPS_MAP, TARGET, format_duration

COLOR = "#2a78d6"
MIN_FLIGHTS = 30
DATA_START, DATA_END = datetime.date(2019, 3, 1), datetime.date(2019, 6, 30)

st.set_page_config(page_title="Flight Price Prediction", page_icon="✈️", layout="wide")


@st.cache_resource
def get_pipeline():
    return predictor.load_pipeline()


@st.cache_data
def get_data():
    return predictor.load_clean_data()


@st.cache_data
def get_metrics():
    return predictor.load_metrics()


@st.cache_resource
def get_intervals():
    return predictor.load_intervals()


@st.cache_data
def get_interval_metrics():
    return predictor.load_interval_metrics()


@st.cache_data
def get_findings():
    return predictor.load_findings()


@st.cache_data
def get_report(name):
    return pd.read_csv(os.path.join(REPORT_DIR, name), encoding="utf-8")


@st.cache_data
def get_analysis():
    return predictor.load_analysis()


@st.cache_data
def get_routes():
    return predictor.known_routes()


def bar_chart(df, x, y, x_title, height=None):
    """Horizontal single-colour bar chart, largest value on top."""
    fig = px.bar(df.sort_values(x), x=x, y=y, orientation="h", text_auto=",.0f")
    fig.update_traces(marker_color=COLOR)
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=None,
        height=height or 120 + 34 * len(df),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig


def predict_tab(data, metrics):
    categories = predictor.known_categories()
    routes = get_routes()

    left, right = st.columns(2)
    with left:
        source = st.selectbox("From", sorted(routes))
        destination = st.selectbox("To", routes[source])
        airline = st.selectbox(
            "Airline", categories["Airline"], index=categories["Airline"].index("IndiGo")
        )
        additional_info = st.selectbox(
            "Additional info",
            categories["Additional_Info"],
            index=categories["Additional_Info"].index("No info"),
        )
    with right:
        journey_date = st.date_input(
            "Date of journey", datetime.date(2019, 5, 15), format="DD/MM/YYYY"
        )
        dep_time = st.time_input("Departure time", datetime.time(9, 0), step=300)
        hours_col, minutes_col = st.columns(2)
        hours = hours_col.number_input("Duration (hours)", 0, 47, 2)
        minutes = minutes_col.number_input("Duration (minutes)", 0, 59, 50, step=5)
        stops = st.selectbox(
            "Total stops", list(STOPS_MAP.values()), format_func=lambda n: f"{n}" if n else "Non-stop"
        )
    offered = st.number_input(
        "Offered price (INR)", min_value=0, value=None, step=100,
        placeholder="Optional: enter a quote to check whether it is cheap, fair or expensive",
    )

    if not DATA_START <= journey_date <= DATA_END:
        st.warning(
            "The training data only covers March-June 2019. Predictions for other "
            "dates are extrapolations and should not be trusted."
        )

    duration = int(hours) * 60 + int(minutes)
    if duration <= 0:
        st.error("Flight duration must be greater than zero.")
        return

    get_pipeline()
    get_intervals()
    quote = predictor.assess_quote(
        dict(
            airline=airline, source=source, destination=destination, journey_date=journey_date,
            dep_time=dep_time, duration_minutes=duration, total_stops=stops,
            additional_info=additional_info,
        ),
        offered,
    )
    route = data.loc[(data["Source"] == source) & (data["Destination"] == destination), TARGET]
    q25, median, q75 = route.quantile([0.25, 0.5, 0.75])
    coverage = get_interval_metrics()["test"]["coverage"]

    st.divider()
    if offered is not None:
        verdicts = {
            predictor.CHEAP: (st.success, "below the range of comparable flights"),
            predictor.FAIR: (st.info, "within the range of comparable flights"),
            predictor.EXPENSIVE: (st.warning, "above the range of comparable flights"),
        }
        show, where = verdicts[quote["label"]]
        show(
            f"**{quote['label']}** - the quote of {offered:,.0f} INR is {where}. It is "
            f"{abs(quote['difference']):,.0f} INR ({abs(quote['difference_pct']):.1f}%) "
            f"{'above' if quote['difference'] > 0 else 'below'} the predicted price."
        )
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted price", f"{quote['predicted']:,.0f} INR")
    col2.metric("80% price range", f"{quote['low']:,.0f} - {quote['high']:,.0f} INR")
    col3.metric(f"Route median ({len(route):,} flights)", f"{median:,.0f} INR")
    st.caption(
        f"The range is built to contain 80% of prices for flights like this one; on the hold-out "
        f"test set it contained {coverage:.1%}. {source} → {destination}, "
        f"{format_duration(duration)}: half of the flights on this route in the data cost between "
        f"{q25:,.0f} and {q75:,.0f} INR. Based on fares from March-June 2019: a case study, not "
        f"advice on current prices."
    )
    if quote["point_prediction"] != quote["predicted"]:
        st.caption(
            f"The point model estimated {quote['point_prediction']:,.0f} INR, outside the range, "
            "so the predicted price is shown at the nearest end of the range."
        )


def insights_tab(data):
    left, right = st.columns(2)

    by_airline = data.groupby("Airline")[TARGET].agg(["median", "count"]).reset_index()
    hidden = by_airline[by_airline["count"] < MIN_FLIGHTS]
    with left:
        st.subheader("Median price by airline")
        st.plotly_chart(
            bar_chart(by_airline[by_airline["count"] >= MIN_FLIGHTS], "median", "Airline",
                      "Median price (INR)"),
            width="stretch",
        )
        st.caption(
            f"{len(hidden)} airlines with fewer than {MIN_FLIGHTS} flights are hidden: "
            + ", ".join(hidden["Airline"]) + "."
        )

    by_stops = data.groupby("Total_Stops")[TARGET].agg(["median", "count"]).reset_index()
    by_stops["Stops"] = by_stops["Total_Stops"] + " (n = " + by_stops["count"].map("{:,}".format) + ")"
    with right:
        st.subheader("Median price by number of stops")
        st.plotly_chart(bar_chart(by_stops, "median", "Stops", "Median price (INR)"), width="stretch")

    st.subheader("Weekday vs weekend")
    weekday = pd.to_datetime(data["Date_of_Journey"], format=DATE_FORMAT).dt.dayofweek
    weekend = data.loc[weekday >= 5, TARGET]
    week = data.loc[weekday < 5, TARGET]
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Weekday median ({len(week):,} flights)", f"{week.median():,.0f} INR")
    col2.metric(f"Weekend median ({len(weekend):,} flights)", f"{weekend.median():,.0f} INR")
    col3.metric("Weekend premium", f"{(weekend.median() / week.median() - 1) * 100:+.1f}%")

    like_for_like_section()

    with st.expander(f"Show the cleaned data ({len(data):,} rows)"):
        st.dataframe(data, width="stretch")


def like_for_like_section():
    """Raw gaps next to gaps between otherwise comparable flights."""
    analysis = get_analysis()
    if analysis is None:
        return
    info = analysis["additional_info"]
    comparisons = [
        ("Weekend vs weekday", "route, airline, stops, month", analysis["weekend"]["summary"]),
        ("1 stop vs non-stop", "route, airline", analysis["one_stop_vs_non_stop"]["summary"]),
        ("Meal not included vs standard fare", "route, airline, stops",
         info["In-flight meal not included"]["summary"]),
        ("No check-in baggage vs standard fare", "route, airline, stops",
         info["No check-in baggage included"]["summary"]),
    ]
    st.subheader("Raw gap vs like-for-like gap")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Comparison": name,
                    "Raw difference of medians": f"{s['raw_diff_pct']:+.1f}%",
                    "Within comparable flights": f"{s['controlled_diff_pct']:+.1f}%",
                    "Held constant": held,
                    "Groups compared": s["groups_compared"],
                }
                for name, held, s in comparisons
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    march = next(m for m in analysis["weekend"]["by_month"] if m["month"] == 3)
    st.caption(
        "The medians above mix different routes and airlines. Comparing only flights that share "
        "the listed attributes changes the picture: the weekend premium is "
        f"{march['controlled_diff_pct']:+.1f}% in March and small in the other months, and a fare "
        "without a meal is cheaper than the standard fare, not dearer. Details in docs/analysis.md."
    )


def effect_metrics(result, unit):
    """Headline numbers of one controlled comparison."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Median difference", f"{result['median_diff_inr']:+,.0f} INR")
    col2.metric(
        "95% confidence interval",
        f"{result['ci95_low_inr']:+,.0f} to {result['ci95_high_inr']:+,.0f} INR",
    )
    col3.metric("Groups compared", f"{result['groups']} {unit}")


def business_tab():
    findings = get_findings()
    if findings is None:
        st.info("Run `python -m src.business_analysis` to generate the findings.")
        return
    st.caption(
        "A case study of fares for March-June 2019 on five routes, written for a travel-agency "
        "user. Not advice on current prices. Differences are medians of within-group "
        f"differences; a group needs at least {findings['min_flights_per_side']} flights on each "
        "side. Confidence intervals are bootstrapped by resampling groups."
    )

    stops = findings["stops_premium"]
    st.subheader("1. A connection costs more than a direct flight")
    effect_metrics(stops, "airline-route pairs")
    st.caption(
        f"One stop vs non-stop within the same airline and route, {stops['flights_compared']:,} "
        f"flights. Dearer in {stops['groups_with_positive_diff']} of {stops['groups']} groups, from "
        f"{stops['min_diff_inr']:+,.0f} to {stops['max_diff_inr']:+,.0f} INR. Without the control "
        f"the gap looks like {stops['raw_diff_inr']:+,.0f} INR."
    )
    st.dataframe(
        get_report("stops_premium_by_group.csv").rename(
            columns={
                "n_non_stop": "Non-stop flights", "n_one_stop": "1-stop flights",
                "median_non_stop": "Non-stop median (INR)", "median_one_stop": "1-stop median (INR)",
                "diff_inr": "Difference (INR)", "diff_pct": "Difference (%)",
            }
        ),
        hide_index=True, width="stretch",
        column_config={"Difference (%)": st.column_config.NumberColumn(format="%.1f")},
    )

    fare = findings["jet_fare_class"]
    raw = fare["all_airlines_raw"]
    st.subheader("2. Jet Airways: the fare without a meal is the cheaper fare class")
    effect_metrics(fare, "route-stops pairs")
    st.caption(
        f"Jet Airways only, 'In-flight meal not included' vs the standard fare on the same route "
        f"and number of stops, {fare['flights_compared']:,} flights. Cheaper in all "
        f"{fare['groups']} groups. This is a difference between two Jet Airways fare classes, not "
        f"the price of a meal. Simpson's paradox: across all airlines the no-meal fares are "
        f"{raw['mean_diff_inr']:+,.0f} INR on average ({raw['median_diff_inr']:+,.0f} INR at the "
        f"median) against the standard fares, because "
        f"{fare['no_meal_flights_by_airline']['Jet Airways']:,} of {raw['n_no_meal']:,} of them are "
        f"Jet Airways, an expensive airline."
    )
    st.dataframe(
        get_report("jet_fare_class_by_group.csv").rename(
            columns={
                "Total_Stops": "Stops", "n_standard": "Standard flights", "n_no_meal": "No-meal flights",
                "median_standard": "Standard median (INR)", "median_no_meal": "No-meal median (INR)",
                "diff_inr": "Difference (INR)", "diff_pct": "Difference (%)",
            }
        ),
        hide_index=True, width="stretch",
        column_config={"Difference (%)": st.column_config.NumberColumn(format="%.1f")},
    )

    st.subheader("3. Cheapest option on each route")
    cheapest = get_report("cheapest_option_by_route.csv")
    st.dataframe(
        pd.DataFrame(
            {
                "Route": cheapest["Route"],
                "Cheapest option": cheapest["cheapest_airline"] + ", " + cheapest["cheapest_stops"],
                "Flights": cheapest["option_flights"],
                "Median (INR)": cheapest["option_median"],
                "25-75% (INR)": cheapest["option_q25"].map("{:,.0f}".format) + " - "
                + cheapest["option_q75"].map("{:,.0f}".format),
                "Route median (INR)": cheapest["route_median"],
                "Route 25-75% (INR)": cheapest["route_q25"].map("{:,.0f}".format) + " - "
                + cheapest["route_q75"].map("{:,.0f}".format),
                "Route CV": cheapest["route_cv"].round(2),
                "Route flights": cheapest["route_flights"],
            }
        ),
        hide_index=True, width="stretch",
    )
    st.caption(
        f"Airline and stop-count options with at least {findings['min_flights_per_option']} "
        "flights, ranked by median price. CV is the coefficient of variation of all prices on the "
        "route (standard deviation / mean): the higher it is, the more there is to gain from "
        "comparing options."
    )


def performance_tab(metrics):
    info, test = metrics["dataset"], metrics["test"]
    st.subheader(f"{metrics['best_model']} on the hold-out test set")
    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{test['full']['r2']:.3f}")
    col2.metric("MAE", f"{test['full']['mae']:,.0f} INR")
    col3.metric("RMSE", f"{test['full']['rmse']:,.0f} INR")
    st.caption(
        f"All {test['full']['n']:,} test rows, price outliers included. On the "
        f"{test['in_range']['n']:,} rows priced up to {info['price_iqr_high']:,.0f} INR (the "
        f"outlier fence learned on the training set): R² {test['in_range']['r2']:.3f}, "
        f"MAE {test['in_range']['mae']:,.0f} INR, RMSE {test['in_range']['rmse']:,.0f} INR."
    )

    intervals = get_interval_metrics()
    st.subheader("80% price range: coverage and width")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Actual price (INR)": f"{band['price_from']:,.0f} - {band['price_to']:,.0f}",
                    "Test rows": band["n"],
                    "Coverage": f"{band['coverage']:.1%}",
                    "Mean width (INR)": f"{band['mean_width']:,.0f}",
                }
                for band in intervals["test_by_price_quartile"]
            ]
            + [
                {
                    "Actual price (INR)": "All test rows",
                    "Test rows": intervals["test"]["n"],
                    "Coverage": f"{intervals['test']['coverage']:.1%}",
                    "Mean width (INR)": f"{intervals['test']['mean_width']:,.0f}",
                }
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Two quantile models (10% and 90%) calibrated with conformalized quantile regression on "
        f"{intervals['calibration_rows']:,} held-out training rows (adjustment "
        f"{intervals['conformal_adjustment_inr']:+,.0f} INR per side). Target coverage "
        f"{intervals['target_coverage']:.0%}. Price bands are quartiles of the actual test price; the "
        f"top band includes the price outliers that were removed from training."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Model comparison (5-fold CV)")
        cv = pd.DataFrame(
            [
                {
                    "Model": name,
                    "CV R²": f"{m['r2_mean']:.4f} ± {m['r2_std']:.4f}",
                    "CV MAE (INR)": f"{m['mae_mean']:,.0f} ± {m['mae_std']:,.0f}",
                }
                for name, m in metrics["cv"].items()
            ]
        )
        st.dataframe(cv, hide_index=True, width="stretch")
        st.caption(
            f"Cross-validated on {info['train_rows']:,} training rows "
            f"({info['train_outliers_removed']} price outliers removed from training only)."
        )
    with right:
        st.subheader("Feature importance")
        importance = pd.DataFrame(metrics["feature_importance"])
        fig = bar_chart(importance, "importance", "feature", "Importance (gain share)")
        fig.update_traces(texttemplate="%{x:.3f}")
        st.plotly_chart(fig, width="stretch")
        st.caption("One-hot columns are summed back to their original feature.")


def main():
    st.title("✈️ Flight Ticket Price Prediction")
    st.caption("Indian domestic flights, March-June 2019 · 5 routes · prices in INR")
    try:
        data, metrics = get_data(), get_metrics()
        get_pipeline()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    tab_predict, tab_insights, tab_business, tab_performance = st.tabs(
        ["Predict", "Market insights", "Business findings", "Model performance"]
    )
    with tab_predict:
        predict_tab(data, metrics)
    with tab_insights:
        insights_tab(data)
    with tab_business:
        business_tab()
    with tab_performance:
        performance_tab(metrics)


main()
