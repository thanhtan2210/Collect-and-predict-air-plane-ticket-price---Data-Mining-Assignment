"""Train an 80% prediction interval for the ticket price.

Two XGBoost quantile models (q = 0.10 and q = 0.90) are trained with the
same cleaning, split and training-only outlier rule as src.train, then the
interval is calibrated with conformalized quantile regression (CQR): 20% of
the training set is held out, and the interval is widened (or narrowed) by
the amount needed to cover 80% of those held-out prices.

Run from the repo root:  python -m src.train_intervals
Writes models/flight_price_intervals.joblib and models/interval_metrics.json.
The point model and models/metrics.json are not touched.
"""
import json
import math
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.predictor import load_pipeline
from src.preprocess import INTERVAL_METRICS_PATH, INTERVAL_MODEL_PATH, MODEL_DIR
from src.train import RANDOM_STATE, load_split, make_pipeline

LOWER_QUANTILE, UPPER_QUANTILE = 0.10, 0.90
TARGET_COVERAGE = UPPER_QUANTILE - LOWER_QUANTILE
CALIBRATION_SIZE = 0.2


def make_quantile_xgb(quantile):
    return XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile,
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def raw_interval(models, X):
    """Uncalibrated [q10, q90]; the two are sorted so they never cross."""
    lower, upper = models["lower"].predict(X), models["upper"].predict(X)
    return np.minimum(lower, upper), np.maximum(lower, upper)


def conformal_adjustment(lower, upper, y, coverage=TARGET_COVERAGE):
    """CQR margin: the finite-sample quantile of how far prices fall outside."""
    scores = np.maximum(lower - y, y - upper)
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, level, method="higher"))


def predict_interval(models, X):
    """Calibrated interval: the raw one moved outwards by the CQR margin."""
    lower, upper = raw_interval(models, X)
    margin = models["conformal_adjustment"]
    lower, upper = lower - margin, upper + margin
    # A negative margin could make a very narrow interval cross.
    return np.minimum(lower, upper), np.maximum(lower, upper)


def interval_metrics(y, lower, upper):
    inside = (y >= lower) & (y <= upper)
    return {
        "n": int(len(y)),
        "coverage": float(inside.mean()),
        "mean_width": float((upper - lower).mean()),
        "median_width": float(np.median(upper - lower)),
    }


def main():
    X_train, y_train, X_test, y_test, info = load_split()
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=CALIBRATION_SIZE, random_state=RANDOM_STATE
    )
    print(
        f"Fit on {len(X_fit):,} rows, calibrate on {len(X_cal):,} rows, "
        f"test on {len(X_test):,} rows (outliers kept)"
    )

    models = {
        "lower": make_pipeline(make_quantile_xgb(LOWER_QUANTILE)).fit(X_fit, y_fit),
        "upper": make_pipeline(make_quantile_xgb(UPPER_QUANTILE)).fit(X_fit, y_fit),
        "lower_quantile": LOWER_QUANTILE,
        "upper_quantile": UPPER_QUANTILE,
        "target_coverage": TARGET_COVERAGE,
    }
    cal_lower, cal_upper = raw_interval(models, X_cal)
    models["conformal_adjustment"] = conformal_adjustment(cal_lower, cal_upper, y_cal.to_numpy())

    y = y_test.to_numpy()
    raw_lower, raw_upper = raw_interval(models, X_test)
    lower, upper = predict_interval(models, X_test)
    point = load_pipeline().predict(X_test)

    # Four bands of the actual test price, cut at its quartiles.
    bands = pd.qcut(y_test, 4)
    by_band = []
    for band in bands.cat.categories:
        mask = (bands == band).to_numpy()
        by_band.append(
            {
                "price_from": float(band.left),
                "price_to": float(band.right),
                **interval_metrics(y[mask], lower[mask], upper[mask]),
            }
        )
    in_range = y <= info["price_iqr_high"]

    metrics = {
        "target_coverage": TARGET_COVERAGE,
        "quantiles": [LOWER_QUANTILE, UPPER_QUANTILE],
        "fit_rows": int(len(X_fit)),
        "calibration_rows": int(len(X_cal)),
        "conformal_adjustment_inr": models["conformal_adjustment"],
        "calibration_coverage_before_adjustment": float(
            ((y_cal >= cal_lower) & (y_cal <= cal_upper)).mean()
        ),
        "test_before_adjustment": interval_metrics(y, raw_lower, raw_upper),
        "test": interval_metrics(y, lower, upper),
        "test_in_range": interval_metrics(y[in_range], lower[in_range], upper[in_range]),
        "test_by_price_quartile": by_band,
        "test_share_below_interval": float((y < lower).mean()),
        "test_share_above_interval": float((y > upper).mean()),
        "point_prediction_inside_interval": float(((point >= lower) & (point <= upper)).mean()),
    }

    print(
        f"\nUncalibrated interval covers {metrics['calibration_coverage_before_adjustment']:.1%} of "
        f"calibration prices; CQR adjustment = {metrics['conformal_adjustment_inr']:+,.0f} INR per side"
    )
    for label, m in [("before adjustment", metrics["test_before_adjustment"]),
                     ("calibrated", metrics["test"]),
                     ("calibrated, in-range prices", metrics["test_in_range"])]:
        print(
            f"  Test, {label:<28} coverage = {m['coverage']:.1%} | "
            f"mean width = {m['mean_width']:,.0f} INR | median width = {m['median_width']:,.0f} INR"
        )
    print("\nBy quartile of the actual test price:")
    for band in by_band:
        print(
            f"  {band['price_from']:>8,.0f} - {band['price_to']:>8,.0f} INR (n={band['n']:>3}): "
            f"coverage = {band['coverage']:.1%} | mean width = {band['mean_width']:,.0f} INR"
        )
    print(
        f"\nMisses: {metrics['test_share_below_interval']:.1%} of prices below the interval, "
        f"{metrics['test_share_above_interval']:.1%} above"
    )
    print(
        f"The point prediction lies inside the interval for "
        f"{metrics['point_prediction_inside_interval']:.1%} of test rows"
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(models, INTERVAL_MODEL_PATH, compress=3)
    with open(INTERVAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    print(f"\nSaved {INTERVAL_MODEL_PATH}\nSaved {INTERVAL_METRICS_PATH}")


if __name__ == "__main__":
    main()
