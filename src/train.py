"""Train and evaluate the flight price models.

Run from the repo root:  python -m src.train
"""
import json
import os
import time

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.preprocess import (
    CATEGORICAL_FEATURES,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET,
    build_features,
    clean_raw,
    iqr_bounds,
    load_raw,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_split(filter_outliers=True):
    """Clean, split, then filter price outliers on the training set only.

    Returns (X_train, y_train, X_test, y_test, info). The IQR fences are
    learned from the training prices; the test set keeps its outliers.
    `filter_outliers=False` returns the training set with its outliers
    (same split, same `info`); the point model always uses the default.
    """
    raw = load_raw()
    clean = clean_raw(raw)
    X, y = build_features(clean), clean[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    low, high = iqr_bounds(y_train)
    keep = y_train.between(low, high)
    info = {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "rows_removed_by_cleaning": int(len(raw) - len(clean)),
        "train_rows_before_outlier_filter": int(len(y_train)),
        "train_outliers_removed": int((~keep).sum()),
        "train_rows": int(keep.sum()),
        "test_rows": int(len(y_test)),
        "price_iqr_low": low,
        "price_iqr_high": high,
    }
    if not filter_outliers:
        return X_train, y_train, X_test, y_test, info
    return X_train[keep], y_train[keep], X_test, y_test, info


def make_preprocessor(categorical=CATEGORICAL_FEATURES, numeric=NUMERIC_FEATURES, scale=False):
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
            ("num", StandardScaler() if scale else "passthrough", list(numeric)),
        ]
    )


def make_xgb():
    return XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_pipeline(model, categorical=CATEGORICAL_FEATURES, numeric=NUMERIC_FEATURES, scale=False):
    return Pipeline(
        [("preprocess", make_preprocessor(categorical, numeric, scale)), ("model", model)]
    )


def make_models():
    return {
        "Ridge": make_pipeline(Ridge(random_state=RANDOM_STATE), scale=True),
        "RandomForest": make_pipeline(
            RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
            )
        ),
        "XGBoost": make_pipeline(make_xgb()),
        "XGBoost (log target)": make_pipeline(
            TransformedTargetRegressor(
                regressor=make_xgb(), func=np.log1p, inverse_func=np.expm1
            )
        ),
    }


def make_cv():
    return KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def regression_metrics(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def feature_importance(pipeline, top=15):
    """Importance per original feature (one-hot columns summed back)."""
    model = pipeline.named_steps["model"]
    if isinstance(model, TransformedTargetRegressor):
        model = model.regressor_
    if not hasattr(model, "feature_importances_"):
        return []
    importances = model.feature_importances_

    preprocess = pipeline.named_steps["preprocess"]
    encoder = preprocess.named_transformers_["cat"]
    cat_cols, num_cols = preprocess.transformers_[0][2], preprocess.transformers_[1][2]

    totals, start = {}, 0
    for col, categories in zip(cat_cols, encoder.categories_):
        totals[col] = float(importances[start:start + len(categories)].sum())
        start += len(categories)
    for col in num_cols:
        totals[col] = float(importances[start])
        start += 1

    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)[:top]
    return [{"feature": name, "importance": value} for name, value in ranked]


def main():
    started = time.time()
    X_train, y_train, X_test, y_test, info = load_split()
    print(f"Raw rows: {info['raw_rows']:,} -> clean rows: {info['clean_rows']:,}")
    print(
        f"Train: {info['train_rows']:,} rows ({info['train_outliers_removed']} outliers "
        f"outside [{info['price_iqr_low']:,.0f}, {info['price_iqr_high']:,.0f}] INR removed)"
        f" | Test: {info['test_rows']:,} rows (outliers kept)"
    )

    models = make_models()
    cv_results = {}
    print("\n5-fold CV on the training set")
    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline, X_train, y_train, cv=make_cv(),
            scoring=("r2", "neg_mean_absolute_error"),
        )
        r2, mae = scores["test_r2"], -scores["test_neg_mean_absolute_error"]
        cv_results[name] = {
            "r2_mean": float(r2.mean()),
            "r2_std": float(r2.std()),
            "mae_mean": float(mae.mean()),
            "mae_std": float(mae.std()),
        }
        print(f"  {name:<22} R2 = {r2.mean():.4f} +/- {r2.std():.4f} | MAE = {mae.mean():,.0f} INR")

    best_name = max(cv_results, key=lambda name: cv_results[name]["r2_mean"])
    best = models[best_name].fit(X_train, y_train)

    # The test set is touched exactly once, by the model chosen on CV.
    y_pred = best.predict(X_test)
    in_range = (y_test <= info["price_iqr_high"]).to_numpy()
    test_metrics = {
        "full": regression_metrics(y_test, y_pred),
        "in_range": regression_metrics(y_test[in_range], y_pred[in_range]),
    }
    print(f"\nBest model: {best_name}")
    for label, m in test_metrics.items():
        print(
            f"  Test ({label}, n={m['n']:,}): R2 = {m['r2']:.4f} | "
            f"MAE = {m['mae']:,.0f} INR | RMSE = {m['rmse']:,.0f} INR"
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best, MODEL_PATH, compress=3)
    metrics = {
        "dataset": info,
        "cv": cv_results,
        "best_model": best_name,
        "test": test_metrics,
        "feature_importance": feature_importance(best),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    print(f"\nSaved {MODEL_PATH}\nSaved {METRICS_PATH}")
    print(f"Done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
