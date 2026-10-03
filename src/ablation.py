"""Feature ablation: CV R2 of XGBoost on the training set for 3 feature sets.

Run from the repo root:  python -m src.ablation
Writes reports/ablation.json.
"""
import json
import os

from sklearn.model_selection import cross_val_score

from src.preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES, REPORT_DIR
from src.train import load_split, make_cv, make_pipeline, make_xgb

ABLATION_PATH = os.path.join(REPORT_DIR, "ablation.json")

BASE = (["Airline", "Source", "Destination"], ["Total_Stops", "Duration_Minutes"])
# Same column order as the full model, so set 3 reproduces src.train exactly.
FEATURE_SETS = {
    "1. Airline + route + stops + duration": BASE,
    "2. Set 1 + date/time features": (BASE[0], NUMERIC_FEATURES),
    "3. Set 2 + Additional_Info (full model)": (CATEGORICAL_FEATURES, NUMERIC_FEATURES),
}


def main():
    X_train, y_train, _, _, _ = load_split()
    print(f"XGBoost, 5-fold CV R2 on the training set ({len(X_train):,} rows)")
    results = []
    for name, (categorical, numeric) in FEATURE_SETS.items():
        pipeline = make_pipeline(make_xgb(), categorical, numeric)
        scores = cross_val_score(
            pipeline, X_train[categorical + numeric], y_train, cv=make_cv(), scoring="r2"
        )
        print(f"  {name:<42} R2 = {scores.mean():.4f} +/- {scores.std():.4f}")
        results.append(
            {
                "feature_set": name,
                "features": categorical + numeric,
                "r2_mean": float(scores.mean()),
                "r2_std": float(scores.std()),
            }
        )

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(ABLATION_PATH, "w", encoding="utf-8") as f:
        json.dump({"train_rows": int(len(X_train)), "feature_sets": results}, f, indent=2)
        f.write("\n")
    print(f"\nSaved {ABLATION_PATH}")


if __name__ == "__main__":
    main()
