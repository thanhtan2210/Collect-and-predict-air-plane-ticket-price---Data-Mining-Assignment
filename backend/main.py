from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
import joblib
import pandas as pd

app = FastAPI(title="Flight Price Prediction API")

# Cấu hình CORS để Frontend (cổng 3000) có thể gọi được API (cổng 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Trong thực tế nên để "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load mô hình (giả sử bạn đã lưu mô hình có tên model.pkl)
try:
    model = joblib.load('model.pkl')
except Exception as e:
    print("Chưa tìm thấy file model.pkl, vui lòng thêm file vào thư mục backend!")
    model = None

# 2. Định nghĩa Dữ liệu đầu vào (Validate từ phía Backend)
class FlightData(BaseModel):
    airline: str
    source: str
    destination: str
    duration: float
    total_stops: int

# 3. Tạo API Endpoint (POST)
@app.post("/api/predict")
def predict_price(data: FlightData):
    if model is None:
        raise HTTPException(status_code=500, detail="Mô hình chưa được tải lên server.")
    
    try:
        # Chuyển đổi dữ liệu thành DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # (Thêm các bước tiền xử lý như One-hot encoding, Label encoding ở đây nếu model yêu cầu)
        
        # Dự đoán
        prediction = model.predict(input_df)
        
        return {
            "status": "success",
            "predicted_price": round(float(prediction[0]), 2)
        }
    except Exception as e:
        # Xử lý lỗi dữ liệu không hợp lệ khi đưa vào model
        raise HTTPException(status_code=400, detail=f"Lỗi khi xử lý dữ liệu: {str(e)}")