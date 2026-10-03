"""Describe the saved models: inputs, outputs, hyperparameters, training data.

Everything is read from the saved model files, the metrics written next to
them and the training split; nothing is typed by hand and nothing is retrained.

Run from the repo root:  python -m src.model_card
Writes reports/model_card.json and prints the Markdown tables used in the README.
"""
import json
import math
import os
import subprocess

import pandas as pd
import sklearn
import xgboost

from src.predictor import (
    CHEAP,
    EXPENSIVE,
    FAIR,
    known_routes,
    load_interval_metrics,
    load_intervals,
    load_metrics,
    load_pipeline,
)
from src.preprocess import (
    BASE_DIR,
    DATE_FORMAT,
    INTERVAL_MODEL_PATH,
    MODEL_PATH,
    REPORT_DIR,
    TARGET,
    clean_raw,
    load_raw,
)
from src.train import load_split

MODEL_CARD_PATH = os.path.join(REPORT_DIR, "model_card.json")


def _same(a, b):
    both_nan = isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)
    return both_nan or a == b


def non_default_params(model):
    """Parameters of a fitted estimator that differ from a freshly built one."""
    defaults = type(model)().get_params()
    return {
        name: value
        for name, value in sorted(model.get_params().items())
        if name not in defaults or not _same(value, defaults[name])
    }


def model_commit(path):
    """Commit that last changed a model file, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", os.path.relpath(path, BASE_DIR)],
            cwd=BASE_DIR, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def describe_inputs(pipeline, X_train):
    """Feature columns as the fitted ColumnTransformer sees them."""
    preprocess = pipeline.named_steps["preprocess"]
    encoder = preprocess.named_transformers_["cat"]
    cat_cols, num_cols = preprocess.transformers_[0][2], preprocess.transformers_[1][2]
    categorical = [
        {"name": col, "dtype": str(X_train[col].dtype), "values": [str(c) for c in categories]}
        for col, categories in zip(cat_cols, encoder.categories_)
    ]
    numeric = [
        {
            "name": col,
            "dtype": str(X_train[col].dtype),
            "train_min": int(X_train[col].min()),
            "train_max": int(X_train[col].max()),
        }
        for col in num_cols
    ]
    return {"categorical": categorical, "numeric": numeric, "routes": known_routes()}


def build_model_card():
    pipeline, intervals = load_pipeline(), load_intervals()
    dataset, interval_metrics = load_metrics()["dataset"], load_interval_metrics()
    X_train, y_train, _, _, _ = load_split()
    dates = pd.to_datetime(
        clean_raw(load_raw()).loc[X_train.index, "Date_of_Journey"], format=DATE_FORMAT
    )

    return {
        "inputs": describe_inputs(pipeline, X_train),
        "outputs": {
            "predicted_price": {
                "unit": "INR",
                "description": "Point prediction, clamped into the price range when shown.",
            },
            "price_range": {
                "unit": "INR",
                "target_coverage": intervals["target_coverage"],
                "quantiles": [intervals["lower_quantile"], intervals["upper_quantile"]],
                "conformal_adjustment_inr": intervals["conformal_adjustment"],
                "description": "[q10 - adjustment, q90 + adjustment], floored at 0.",
            },
            "label": {
                CHEAP: "offered price below the lower end of the price range",
                FAIR: "offered price inside the price range (ends included)",
                EXPENSIVE: "offered price above the upper end of the price range",
            },
        },
        "hyperparameters": {
            "point": non_default_params(pipeline.named_steps["model"]),
            "lower": non_default_params(intervals["lower"].named_steps["model"]),
            "upper": non_default_params(intervals["upper"].named_steps["model"]),
            "calibration": {
                "method": "conformalized quantile regression",
                "target_coverage": intervals["target_coverage"],
                "fit_rows": interval_metrics["fit_rows"],
                "calibration_rows": interval_metrics["calibration_rows"],
                "training_outliers": intervals["training_outliers"],
            },
        },
        "training_data": {
            "clean_rows": dataset["clean_rows"],
            "train_rows": dataset["train_rows"],
            "train_outliers_removed": dataset["train_outliers_removed"],
            "test_rows": dataset["test_rows"],
            "journey_date_from": dates.min().date().isoformat(),
            "journey_date_to": dates.max().date().isoformat(),
            "price_iqr_low": dataset["price_iqr_low"],
            "price_iqr_high": dataset["price_iqr_high"],
            "train_price_min": int(y_train.min()),
            "train_price_max": int(y_train.max()),
            "target": TARGET,
        },
        "versions": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "point_model_commit": model_commit(MODEL_PATH),
            "interval_models_commit": model_commit(INTERVAL_MODEL_PATH),
        },
    }


def markdown_tables(card):
    """The README tables, so they are never typed by hand."""
    lines = ["| Column | Type | Valid values |", "| --- | --- | --- |"]
    for col in card["inputs"]["categorical"]:
        lines.append(f"| `{col['name']}` | {col['dtype']} | {', '.join(col['values'])} |")
    for col in card["inputs"]["numeric"]:
        lines.append(
            f"| `{col['name']}` | {col['dtype']} | {col['train_min']} to {col['train_max']} |"
        )

    params = card["hyperparameters"]
    names = sorted(set(params["point"]) | set(params["lower"]) | set(params["upper"]))
    lines += [
        "",
        "| Hyperparameter | Point model | 10% quantile model | 90% quantile model |",
        "| --- | --- | --- | --- |",
    ]
    for name in names:
        cells = [str(params[key].get(name, "default")) for key in ("point", "lower", "upper")]
        lines.append(f"| `{name}` | {' | '.join(cells)} |")
    return "\n".join(lines)


def main(path=MODEL_CARD_PATH):
    card = build_model_card()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
        f.write("\n")
    print(markdown_tables(card))
    print(f"\nSaved {path}")
    return card


if __name__ == "__main__":
    main()
