"""Streamlit app. Run from the repo root:  streamlit run app.py"""
import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from src import predictor
from src.preprocess import DATE_FORMAT, STOPS_MAP, TARGET, format_duration

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
    price = predictor.predict_price(
        airline, source, destination, journey_date, dep_time, duration, stops, additional_info
    )
    route = data.loc[(data["Source"] == source) & (data["Destination"] == destination), TARGET]
    q25, median, q75 = route.quantile([0.25, 0.5, 0.75])
    mae = metrics["test"]["full"]["mae"]

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted price", f"{price:,.0f} INR")
    col2.metric("Typical error (test MAE)", f"± {mae:,.0f} INR")
    col3.metric(f"Route median ({len(route):,} flights)", f"{median:,.0f} INR")
    st.caption(
        f"{source} → {destination}, {format_duration(duration)}: half of the flights on this "
        f"route in the data cost between {q25:,.0f} and {q75:,.0f} INR (25th-75th percentile)."
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

    with st.expander(f"Show the cleaned data ({len(data):,} rows)"):
        st.dataframe(data, width="stretch")


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

    tab_predict, tab_insights, tab_performance = st.tabs(
        ["Predict", "Market insights", "Model performance"]
    )
    with tab_predict:
        predict_tab(data, metrics)
    with tab_insights:
        insights_tab(data)
    with tab_performance:
        performance_tab(metrics)


main()
