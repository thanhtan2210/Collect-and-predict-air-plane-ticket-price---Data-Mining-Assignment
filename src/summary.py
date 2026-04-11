import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import xgboost as xgb
# Giả định bạn đã có y_test và các y_pred từ notebook trước
# Đây là code mẫu để export table
data = {
    'Algorithm': ['Random Forest (Baseline)', 'XGBoost (Default)', 'XGBoost (Optimized)'],
    'R-squared': [0.8095, 0.8421, 0.8423],
    'MAE (INR)': [1118.71, 1071.32, 1122.08],
    'RMSE (INR)': [1754.00, 1597.19, 1596.05]
}
df_res = pd.DataFrame(data)
print(df_res.to_latex(index=False,
      caption="Model Performance Comparison", label="tab:results"))
