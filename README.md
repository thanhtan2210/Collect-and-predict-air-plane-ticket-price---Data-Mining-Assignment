
# Flight Ticket Price Prediction - Data Mining Assignment

This project focuses on data mining and building a machine learning model to predict flight ticket prices based on itinerary information. The project covers the entire workflow, from data cleaning and feature engineering to model optimization and result analysis.

---

## Project Structure

```plaintext
Collect-and-predict-air-plane-ticket-price/
├── data/
│   ├── IndianFlightdata - Sheet1.csv    # Raw data
│   └── data_for_assignment/
│       └── cleaned_flight_data.csv      # Cleaned & encoded data
├── notebook/
│   ├── eda.ipynb                        # Data analysis & Preprocessing
│   ├── model_training.ipynb             # Model training (RF, XGBoost) & Tuning
│   └── visual_report_data.ipynb         # Visualization & Metrics for reporting
├── models/
│   ├── best_flight_price_model_optimized.json  # Best XGBoost model
│   └── rf_flight_price_model.pkl               # Baseline model (Random Forest)
├── doc/
│   └── dataset_requirement.md           # Project requirements
└── requirements.txt                     # Required libraries
```

---

## Prerequisites

The project requires **Python 3.8+**. To install the necessary libraries, run the following command:

```bash
pip install -r requirements.txt
```

*Main libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, xgboost, joblib.*

---

## Workflow

### Step 1: EDA & Preprocessing
Open the `notebook/eda.ipynb` file to perform:
- Data cleaning (Missing values, Duplicates).
- Outlier handling using the IQR method.
- Feature Engineering: Extracting date/month, calculating flight duration, adding the `is_weekend` variable...
- Data transformation (One-Hot Encoding).

### Step 2: Model Training
Open the `notebook/model_training.ipynb` file to:
- Split the dataset into Train/Test sets (80/20).
- Train and compare **Random Forest** and **XGBoost** models.
- Hyperparameter Tuning using `RandomizedSearchCV`.

### Step 3: Reporting
Open the `notebook/visual_report_data.ipynb` file to export charts:
- **Feature Importance:** Which factors affect ticket prices the most.
- **Cross-Validation:** Evaluating model stability.
- **Residual Analysis:** Analyzing prediction errors.
- **Market Insights:** Ticket price trends by airline and time.

---

## Key Results

Based on practical experiments, the model achieved impressive metrics:
- **R-squared ($R^2$):** ~0.84 - 0.89 (explains over 85% of price variance).
- **RMSE:** Average error optimized after Tuning.
- **Best model:** XGBoost after tuning and outlier handling.

---

## Authors
- [Phan Thanh Tan, 2213076
- Tran Minh Tam, 2212085
- Vu Duc Lam, 2211824]
- Course Project: Data Mining