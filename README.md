# Flight Ticket Price Prediction - Data Mining Assignment

Predicts the price of Indian domestic flight tickets from itinerary details (airline, route, date, departure time, duration, stops, fare conditions). The repo contains a reproducible training pipeline, a feature ablation, tests, and a Streamlit app.

Every number in this README is printed by a script in this repo and can be regenerated with the commands below.

| Predict | Market insights | Model performance |
| --- | --- | --- |
| ![Predict tab](docs/images/predict.png) | ![Market insights tab](docs/images/insights.png) | ![Model performance tab](docs/images/performance.png) |

---

## Results

Produced by `python -m src.train` (also stored in [models/metrics.json](models/metrics.json)).

**Data:** 10,683 raw rows → 10,462 after cleaning (220 exact duplicates and 1 row with missing values removed). 80/20 split: 8,296 training rows after removing 73 price outliers, 2,093 test rows with outliers kept.

**Model comparison - 5-fold cross-validation on the training set**

| Model | CV R² | CV MAE (INR) |
| --- | --- | --- |
| Ridge (baseline) | 0.7042 ± 0.0073 | 1,639 ± 37 |
| RandomForest | 0.9136 ± 0.0049 | 656 ± 18 |
| **XGBoost** | **0.9279 ± 0.0035** | 610 ± 15 |
| XGBoost (log target) | 0.9268 ± 0.0040 | 606 ± 17 |

XGBoost has the highest CV R² and is the model that gets refit on the full training set and saved. The log-target variant is within one standard deviation of it.

**Hold-out test set - XGBoost, evaluated once**

| Test subset | Rows | R² | MAE (INR) | RMSE (INR) |
| --- | --- | --- | --- | --- |
| All test rows (outliers included) | 2,093 | 0.8722 | 681 | 1,632 |
| Price ≤ 23,090 INR (training IQR fence) | 2,072 | 0.9294 | 588 | 1,078 |

The 21 test rows above the fence (16 of them Jet Airways) account for the gap between the two rows: the model was never trained on prices that high.

**Feature ablation - XGBoost, 5-fold CV R² on the training set** (`python -m src.ablation`)

| Feature set | CV R² |
| --- | --- |
| 1. Airline + Source + Destination + Total_Stops + Duration_Minutes | 0.6567 ± 0.0297 |
| 2. Set 1 + all date/time features | 0.8069 ± 0.0090 |
| 3. Set 2 + Additional_Info (full model) | 0.9279 ± 0.0035 |

---

## Key findings

Full write-up with tables and charts: [docs/analysis.md](docs/analysis.md). Produced by `python -m src.analysis`.

Raw gaps between medians mix different routes and airlines, so each attribute is also compared within groups of otherwise comparable flights.

| Comparison | Raw gap of medians | Like-for-like gap | Held constant |
| --- | --- | --- | --- |
| Weekend vs weekday | +7.1% | +6.6% | route, airline, stops, month |
| 1 stop vs non-stop | +119.9% | +55.6% | route, airline |
| "In-flight meal not included" vs standard fare | +30.0% | -26.5% | route, airline, stops |
| "No check-in baggage included" vs standard fare | -51.3% | -1.4% | route, airline, stops |

- **A fare without a meal looks 30% dearer and is in fact 26% cheaper.** The remark sits almost entirely on Jet Airways, the most expensive airline; within the same airline, route and stops it is cheaper in all 7 comparable groups.
- **One stop costs more than non-stop in every comparable group**, but about half of the raw gap comes from which routes have connecting flights.
- **The weekend premium is a March effect:** +20.6% in March, between +1.5% and +2.8% in the other months.
- **Model error is concentrated in a small tail.** The median absolute error is 294 INR against a mean of 681 INR; the worst 5% of test rows carry 38.7% of the total error, and fares above the outlier fence are underpredicted by 9,864 INR on average.
- **Forecasting a new month is harder than the headline score suggests.** Training on March-May and testing on June gives R² 0.846 and MAE 1,091 INR, against R² 0.929 and MAE 588 INR for comparable prices under the random split.

---

## Validation choices

- **Duplicates are removed before the split.** The raw file has 220 fully duplicated rows. Splitting first would put copies of the same row in both train and test and inflate the test score.
- **Outlier fences are learned on the training set only.** The IQR bounds on price (upper fence 23,090 INR) are computed from training prices and applied to training rows only. The test set keeps its outliers, and results are reported both on the whole test set and on the in-range rows.
- **Model selection uses cross-validation, not the test set.** The four models are compared with 5-fold CV on the training set; the test set is used once, for the selected model.
- **One preprocessing function for training and prediction.** `src/preprocess.py::build_features` is the only place features are computed. Predictions build a row in the raw CSV layout (`make_raw_row`) and pass it through the same function. This fixes a bug in the previous CLI, which never filled the duration feature and therefore always predicted with a flight duration of 0. A test asserts that the serving path yields exactly the training features.
- **Label normalisation.** "New Delhi" and "Delhi" (Destination) are the same place, and "No Info" / "No info" (Additional_Info) are the same value; both are merged.
- **Fixed seeds.** All splits and models use `random_state=42`; running `src.train` twice yields an identical `metrics.json`.

### Is `Additional_Info` leakage?

No. `Additional_Info` describes the fare conditions of the ticket ("In-flight meal not included", "No check-in baggage included", "1 Long layover", ...). It is a property of the product that is shown to the buyer at booking time, not something derived from the price afterwards, so it is available when a prediction is needed. It is also the single most useful addition in the ablation (0.81 → 0.93), because it separates fare classes that share the same airline, route and schedule.

---

## Project structure

```plaintext
├── app.py                     # Streamlit app (3 tabs)
├── src/
│   ├── preprocess.py          # Cleaning + feature engineering shared by train and predict
│   ├── train.py               # Model comparison, test evaluation, saves model + metrics
│   ├── ablation.py            # Feature ablation
│   ├── analysis.py            # Like-for-like comparisons, error analysis, time-based validation
│   ├── predictor.py           # Loads the saved pipeline, predicts from user input
│   ├── predict_cli.py         # Terminal interface
│   └── summary.py             # LaTeX tables generated from models/metrics.json
├── tests/                     # pytest: preprocessing and predictor
├── models/
│   ├── flight_price_pipeline.joblib   # Fitted sklearn Pipeline (one-hot + XGBoost)
│   ├── metrics.json                   # All metrics reported above
│   └── analysis.json                  # All numbers in docs/analysis.md
├── data/
│   └── IndianFlightdata - Sheet1.csv  # Raw data used by the pipeline
├── notebook/                  # Original exploratory notebooks (see note below)
├── docs/
│   ├── analysis.md            # Key findings
│   └── images/                # App screenshots and analysis charts
└── .github/workflows/ci.yml   # Tests + training on every push
```

`data/` also contains `flight_cleaned.csv`, `airports.csv`, `flight_price.xlsx` and `archive.zip`. They are leftovers from the exploration phase and are not used by the pipeline.

---

## How to run

Requires Python 3.11+ (developed and tested on Python 3.13). The package versions in `requirements.txt` are pinned because the saved `.joblib` model must be loaded with the same library versions that wrote it; the pinned NumPy needs Python 3.12 or newer, so on Python 3.11 install a NumPy 2.4 release and retrain with `python -m src.train`.

All commands are run from the repo root.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements-dev.txt

python -m pytest -q               # tests
python -m src.train               # compare models, save model + metrics
python -m src.ablation            # feature ablation
python -m src.analysis            # numbers and charts for docs/analysis.md
python -m src.summary             # LaTeX tables from metrics.json
python -m src.predict_cli         # predict in the terminal
streamlit run app.py              # web app
```

The trained model is committed, so the app and the CLI work right after cloning without retraining.

### Deploy on Streamlit Community Cloud

1. Push the repo to GitHub (the app needs `app.py`, `requirements.txt`, `src/`, `models/` and `data/IndianFlightdata - Sheet1.csv`).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and click **Create app**.
3. Select this repository, the branch, and `app.py` as the main file.
4. Under **Advanced settings**, choose Python 3.13 so that the pinned versions match the saved model.
5. Click **Deploy**. Dependencies are installed from `requirements.txt`.

---

## Data and limitations

The data is an existing public Kaggle dataset of Indian domestic flight fares; this project did not collect it.

- **Four months of one year.** Journeys run from March to June 2019. The model knows nothing about other seasons or about price levels after 2019, and the app warns when a date outside this window is entered.
- **Five routes.** Banglore → Delhi, Delhi → Cochin, Kolkata → Banglore, Mumbai → Hyderabad and Chennai → Kolkata. The app only offers these.
- **No booking date.** How far in advance a ticket is bought is a major price driver and is not in the data.
- **Random split.** The headline scores come from a random split over rows, so they describe interpolation within the same period. The time-based check in [docs/analysis.md](docs/analysis.md) shows the error on an unseen month is clearly higher.
- **Rare categories.** No business-class row remains in the training set after the outlier filter, so the model cannot price business fares.

## About the notebooks

The notebooks in `notebook/` are the original exploration for the course assignment and are kept unchanged. Their numbers differ from the ones above (for example XGBoost test R² 0.842) because they were computed before the evaluation was fixed: duplicates were kept, outliers were filtered before the train/test split, and the feature set was different. The numbers in this README supersede them.

---

## Authors
- [Phan Thanh Tan, 2213076
- Tran Minh Tam, 2212085
- Vu Duc Lam, 2211824]
- Course Project: Data Mining
