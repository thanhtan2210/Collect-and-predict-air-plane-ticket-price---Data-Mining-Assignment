"""Draw the README figures from the saved results, in a light and a dark version.

Every number comes from models/*.json, reports/*.json|csv or the saved point
model applied to the test split of src.train; nothing is recomputed with new
logic. If the reproduced test predictions or the stops table disagree with
the stored metrics, the script stops instead of drawing.

Each `fig_*(data, theme)` returns a matplotlib Figure, so the app draws the
same charts from the same code.

Run from the repo root:  python -m src.make_figures
Writes docs/images/fig_*_{light,dark}.svg.
"""
import contextlib
import functools
import json
import os
import threading

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter, PercentFormatter

from src.ablation import ABLATION_PATH
from src.predictor import load_findings, load_interval_metrics, load_metrics, load_pipeline
from src.preprocess import BASE_DIR, REPORT_DIR
from src.train import load_split, regression_metrics

IMAGE_DIR = os.path.join(BASE_DIR, "docs", "images")
STOPS_TABLE_PATH = os.path.join(REPORT_DIR, "stops_premium_by_group.csv")

THEMES = {
    "light": {
        "background": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e1e0d9",
        "primary": "#2a78d6", "secondary": "#eb6834", "neutral": "#898781",
    },
    "dark": {
        "background": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#2c2c2a",
        "primary": "#3987e5", "secondary": "#d95926", "neutral": "#898781",
    },
}
# 540 x 300 pt = 720 x 400 CSS px. Text stays text, so the reader's own sans font is used.
FIGSIZE = (7.5, 300 / 72)
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "svg.fonttype": "none",
    "svg.hashsalt": "flight-price-figures",
    "axes.unicode_minus": False,
}
# pyplot and rcParams are global, so figures are drawn one at a time.
_LOCK = threading.RLock()
BAR_WIDTH = 0.6
LINE_WIDTH = 2
TOP_FEATURES = 10


def r2_label(value):
    return f"{value:.4f}"


def percent_label(value):
    return f"{value:.1%}"


def inr(value):
    return f"{value:,.0f}"


def load_figure_data():
    """Everything the figures show, read from models/, reports/ and the test split."""
    with open(ABLATION_PATH, encoding="utf-8") as f:
        ablation = json.load(f)
    findings = load_findings()
    stops_table = pd.read_csv(STOPS_TABLE_PATH, encoding="utf-8")
    if stops_table["diff_inr"].median() != findings["stops_premium"]["median_diff_inr"]:
        raise ValueError(
            f"{STOPS_TABLE_PATH} does not match reports/findings.json. "
            "Run: python -m src.business_analysis"
        )

    metrics = load_metrics()
    _, _, X_test, y_test, _ = load_split()
    predicted = load_pipeline().predict(X_test)
    reproduced = regression_metrics(y_test, predicted)
    if not all(np.isclose(reproduced[key], value) for key, value in metrics["test"]["full"].items()):
        raise ValueError(
            "The saved model does not reproduce models/metrics.json on the test split. "
            "Run: python -m src.train"
        )
    return {
        "ablation": ablation,
        "metrics": metrics,
        "intervals": load_interval_metrics(),
        "stops": findings["stops_premium"],
        "stops_table": stops_table,
        "actual": y_test.to_numpy(),
        "predicted": predicted,
    }


@contextlib.contextmanager
def style():
    """The rcParams the figures are drawn and saved with (fonts are resolved when saving)."""
    with _LOCK, plt.rc_context(RC):
        yield


def figure(draw):
    """Make a drawing function pure: `draw(data, theme)` with theme "light" or "dark"."""
    @functools.wraps(draw)
    def wrapper(data, theme="light"):
        with style():
            return draw(data, THEMES[theme])
    return wrapper


def start(theme, title, question, left=0.09, bottom=0.13, grid="x"):
    """Empty axes with the title, the question as subtitle and a value-axis grid."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor(theme["background"])
    ax.set_facecolor(theme["background"])
    fig.subplots_adjust(left=left, right=0.95, top=0.8, bottom=bottom)
    fig.text(0.03, 0.93, title, fontsize=12.5, fontweight="bold", color=theme["text"])
    fig.text(0.03, 0.865, question, fontsize=9.5, color=theme["muted"])

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(which="both", length=0, colors=theme["muted"], labelsize=9)
    ax.grid(axis=grid, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(theme["muted"])
    ax.yaxis.label.set_color(theme["muted"])
    return fig, ax


@figure
def fig_ablation(data, theme):
    sets = data["ablation"]["feature_sets"][::-1]
    fig, ax = start(
        theme, "Feature ablation", "How much does each group of features add to the model?", left=0.36
    )
    values = [s["r2_mean"] for s in sets]
    ax.barh([s["feature_set"] for s in sets], values, height=BAR_WIDTH, color=theme["primary"])
    for position, value in enumerate(values):
        ax.text(value + 0.012, position, r2_label(value), va="center", color=theme["text"])
    ax.set_xlim(0, 1.08)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("5-fold CV R² on the training set (XGBoost)")
    return fig


@figure
def fig_models(data, theme):
    metrics = data["metrics"]
    names = list(metrics["cv"])[::-1]
    fig, ax = start(
        theme, "Model comparison", "Which model scores best in cross-validation?", left=0.2
    )
    for position, name in enumerate(names):
        mean, std = metrics["cv"][name]["r2_mean"], metrics["cv"][name]["r2_std"]
        color = theme["primary"] if name == metrics["best_model"] else theme["neutral"]
        # The error bar is shorter than a dot would be wide, so it is drawn as a
        # block from mean - std to mean + std with a notch at the mean.
        ax.barh(position, 2 * std, left=mean - std, height=0.3, color=color)
        ax.plot(
            [mean, mean], [position - 0.15, position + 0.15],
            color=theme["background"], linewidth=LINE_WIDTH,
        )
        ax.text(
            mean + std + 0.008, position, f"{r2_label(mean)} ± {r2_label(std)}",
            va="center", color=theme["text"],
        )
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlim(0.65, 1.0)
    ax.set_xlabel("5-fold CV R² on the training set: block = mean ± 1 standard deviation (axis starts at 0.65)")
    return fig


@figure
def fig_pred_vs_actual(data, theme):
    metrics = data["metrics"]
    test, fence = metrics["test"]["full"], metrics["dataset"]["price_iqr_high"]
    actual, predicted = data["actual"], data["predicted"]
    fig, ax = start(
        theme, "Predicted vs actual price", "How close are the predictions on the hold-out test set?",
        left=0.11, bottom=0.15, grid="both",
    )
    low, high = min(actual.min(), predicted.min()) * 0.9, max(actual.max(), predicted.max()) * 1.1
    ax.plot([low, high], [low, high], color=theme["neutral"], linewidth=LINE_WIDTH)
    ax.axvline(fence, color=theme["secondary"], linewidth=LINE_WIDTH, linestyle=(0, (4, 3)))
    ax.scatter(actual, predicted, s=14, alpha=0.3, color=theme["primary"], linewidths=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ticks = [2000, 5000, 10000, 20000, 50000]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks(ticks)
        axis.set_major_formatter(FuncFormatter(lambda value, _: inr(value)))
        axis.set_minor_formatter(NullFormatter())
    ax.grid(which="minor", visible=False)
    ax.set_xlabel("Actual price (INR, log scale)")
    ax.set_ylabel("Predicted price (INR, log scale)")
    ax.text(
        0.03, 0.93, f"R² {test['r2']:.3f}, MAE {inr(test['mae'])} INR on {test['n']:,} test rows",
        transform=ax.transAxes, va="top", color=theme["text"],
    )
    ax.text(
        fence * 1.04, low * 1.12, f"no training price\nabove {inr(fence)} INR",
        va="bottom", color=theme["muted"], fontsize=8.5,
    )
    ax.text(high * 0.95, high * 0.8, "y = x", ha="right", va="top", color=theme["muted"], fontsize=8.5)
    return fig


@figure
def fig_interval_coverage(data, theme):
    intervals = data["intervals"]
    bands, target = intervals["test_by_price_quartile"], intervals["target_coverage"]
    fig, ax = start(
        theme, "Coverage of the 80% price range",
        "Does the range hold for cheap and expensive tickets alike?", bottom=0.2, grid="y",
    )
    positions = range(len(bands))
    ax.bar(positions, [b["coverage"] for b in bands], width=BAR_WIDTH, color=theme["primary"])
    ax.axhline(target, color=theme["secondary"], linewidth=LINE_WIDTH)
    for position, band in zip(positions, bands):
        ax.text(
            position, max(band["coverage"], target) + 0.03, percent_label(band["coverage"]),
            ha="center", color=theme["text"],
        )
    ax.set_xlim(-0.5, len(bands) + 0.1)
    ax.text(len(bands) + 0.08, target + 0.03, f"target {target:.0%}", ha="right", color=theme["muted"])
    ax.set_xticks(
        positions,
        [
            f"{inr(b['price_from'])} - {inr(b['price_to'])} INR\nmean width {inr(b['mean_width'])} INR"
            for b in bands
        ],
    )
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Test prices inside the range")
    ax.set_xlabel("Quartile of the actual test price")
    return fig


@figure
def fig_importance(data, theme):
    top = data["metrics"]["feature_importance"][:TOP_FEATURES][::-1]
    fig, ax = start(
        theme, f"Feature importance, top {TOP_FEATURES}", "What does the price model rely on?", left=0.2
    )
    values = [f["importance"] for f in top]
    ax.barh([f["feature"] for f in top], values, height=BAR_WIDTH, color=theme["primary"])
    for position in range(len(top) - 3, len(top)):
        ax.text(
            values[position] + 0.005, position, f"{values[position]:.3f}",
            va="center", color=theme["text"],
        )
    ax.set_xlim(0, max(values) * 1.12)
    ax.set_xlabel("Share of total gain (one-hot columns summed per feature)")
    return fig


@figure
def fig_stops_premium(data, theme):
    stops, table = data["stops"], data["stops_table"].sort_values("diff_inr")
    fig, ax = start(
        theme, "Price of a connection", "How much more does one stop cost on the same airline and route?",
        left=0.33, bottom=0.15,
    )
    ax.axvspan(stops["ci95_low_inr"], stops["ci95_high_inr"], color=theme["secondary"], alpha=0.15, linewidth=0)
    ax.axvline(stops["median_diff_inr"], color=theme["secondary"], linewidth=LINE_WIDTH)
    positions = range(len(table))
    ax.plot(table["diff_inr"], positions, "o", markersize=8, color=theme["primary"])
    ax.set_yticks(positions, table["Airline"] + ", " + table["Route"])
    ax.set_ylim(-0.7, len(table) + 1.3)
    ax.set_xlim(0, table["diff_inr"].max() * 1.08)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: inr(value)))
    ax.text(
        stops["ci95_high_inr"] + 120, len(table) + 1.0,
        f"median +{inr(stops['median_diff_inr'])} INR (line)\n"
        f"95% CI +{inr(stops['ci95_low_inr'])} to +{inr(stops['ci95_high_inr'])} (band)",
        va="top", color=theme["text"],
    )
    ax.set_xlabel("Median price, 1 stop minus non-stop (INR)")
    return fig


# name -> (drawing function, <title> read by screen readers)
FIGURES = {
    "fig_ablation": (fig_ablation, "Bar chart of cross-validated R2 for three feature sets"),
    "fig_models": (fig_models, "Cross-validated R2 of four models with standard deviations"),
    "fig_pred_vs_actual": (fig_pred_vs_actual, "Scatter plot of predicted against actual test prices"),
    "fig_interval_coverage": (
        fig_interval_coverage, "Bar chart of price range coverage by quartile of the test price",
    ),
    "fig_importance": (fig_importance, "Bar chart of the ten most important features"),
    "fig_stops_premium": (
        fig_stops_premium, "Dot plot of the one-stop price premium for each airline and route",
    ),
}


# One conclusion per figure, shown under it in the README and in the app. The
# numbers are filled in by `captions` from the data the figure is drawn from.
CAPTIONS = {
    "fig_ablation": (
        "The date and time features lift CV R² from {base} to {with_dates}, and the fare remark "
        "in Additional_Info lifts it again to **{full}**"
    ),
    "fig_models": (
        "XGBoost leads at **{xgboost}** with the log-target variant ({log_target}) inside its "
        "one-standard-deviation block, while the Ridge baseline stays at {ridge}"
    ),
    "fig_pred_vs_actual": (
        "Predictions follow the diagonal up to the outlier fence of **{fence} INR** and fall "
        "well below it for the dearer test fares"
    ),
    "fig_interval_coverage": (
        "Coverage stays between {lowest} and {highest} in every price quartile, while the mean "
        "width of the range grows from {cheapest} INR for the cheapest fares to "
        "**{dearest} INR** for the dearest"
    ),
    "fig_importance": (
        "Airline ({airline}) and Additional_Info ({additional_info}) carry more than half of "
        "the total gain, while the largest date or time feature, Journey_Month, has {month}"
    ),
    "fig_stops_premium": (
        "One stop was dearer than non-stop in all {groups} airline and route groups, by {low} "
        "to {high} INR with a median of **{median} INR**"
    ),
}


def captions(data):
    """CAPTIONS with their numbers filled in, keyed by figure name."""
    cv = data["metrics"]["cv"]
    importance = {f["feature"]: f"{f['importance']:.3f}" for f in data["metrics"]["feature_importance"]}
    sets, bands, stops = data["ablation"]["feature_sets"], data["intervals"]["test_by_price_quartile"], data["stops"]
    coverage = [band["coverage"] for band in bands]
    values = {
        "fig_ablation": {
            "base": r2_label(sets[0]["r2_mean"]),
            "with_dates": r2_label(sets[1]["r2_mean"]),
            "full": r2_label(sets[2]["r2_mean"]),
        },
        "fig_models": {
            "xgboost": r2_label(cv["XGBoost"]["r2_mean"]),
            "log_target": r2_label(cv["XGBoost (log target)"]["r2_mean"]),
            "ridge": r2_label(cv["Ridge"]["r2_mean"]),
        },
        "fig_pred_vs_actual": {"fence": inr(data["metrics"]["dataset"]["price_iqr_high"])},
        "fig_interval_coverage": {
            "lowest": percent_label(min(coverage)),
            "highest": percent_label(max(coverage)),
            "cheapest": inr(bands[0]["mean_width"]),
            "dearest": inr(bands[-1]["mean_width"]),
        },
        "fig_importance": {
            "airline": importance["Airline"],
            "additional_info": importance["Additional_Info"],
            "month": importance["Journey_Month"],
        },
        "fig_stops_premium": {
            "groups": stops["groups"],
            "low": inr(stops["min_diff_inr"]),
            "high": inr(stops["max_diff_inr"]),
            "median": inr(stops["median_diff_inr"]),
        },
    }
    return {name: text.format(**values[name]) for name, text in CAPTIONS.items()}


def main(out_dir=IMAGE_DIR):
    data = load_figure_data()
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    with style():
        for theme, colors in THEMES.items():
            for name, (draw, title) in FIGURES.items():
                fig = draw(data, theme)
                path = os.path.join(out_dir, f"{name}_{theme}.svg")
                fig.savefig(
                    path, format="svg", facecolor=colors["background"],
                    metadata={"Title": title, "Date": None},
                )
                plt.close(fig)
                paths.append(path)
                print(f"Saved {path}")
    return paths


if __name__ == "__main__":
    main()
