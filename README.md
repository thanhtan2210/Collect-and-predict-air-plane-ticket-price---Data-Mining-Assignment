# 🛫 Hướng dẫn Cài đặt và Chạy Dự án: Dự đoán Giá Vé Máy Bay

Tài liệu này hướng dẫn chi tiết các bước để thiết lập môi trường và khởi chạy hệ thống dự đoán giá vé máy bay với kiến trúc tách biệt **Backend (FastAPI)** và **Frontend (Node.js)**.

---

## 📂 Cấu trúc Thư mục (Project Structure)

Đảm bảo thư mục dự án của bạn được sắp xếp như sau:

```plaintext
Collect-and-predict-air-plane-ticket-price/
├── backend/
│   ├── main.py              # API xử lý logic và model AI
│   ├── model.pkl            # File mô hình đã huấn luyện
│   └── requirements.txt     # Danh sách thư viện Python
└── frontend/
    ├── server.js            # Máy chủ web Node.js
    ├── package.json         # Cấu hình dự án Node.js
    └── public/
        ├── index.html       # Giao diện chính
        └── script.js        # Logic xử lý và gọi API
```

---

## 🛠 Yêu cầu hệ thống (Prerequisites)

Trước khi bắt đầu, hãy cài đặt các phần mềm sau:

- **Python (3.8+)**
- **Node.js (18.x+)**

---

## 🟢 PHẦN 1: Cấu hình và Chạy BACKEND (Python / FastAPI)

### Bước 1: Di chuyển vào thư mục Backend

```bash
cd D:\DataMining\Collect-and-predict-air-plane-ticket-price---Data-Mining-Assignment\backend
```

---

### Bước 2: Cài đặt thư viện Python

```bash
pip install fastapi uvicorn pydantic joblib pandas scikit-learn
```

---

### Bước 3: Khởi chạy Server Backend

```bash
uvicorn main:app --reload
```

**Ghi chú:**  
Khi thấy:
```
Uvicorn running on http://127.0.0.1:8000
```
→ Backend đã chạy thành công.

---

## 🔵 PHẦN 2: Cấu hình và Chạy FRONTEND (Node.js / Express)

### Bước 1: Di chuyển vào thư mục Frontend

```bash
cd D:\DataMining\Collect-and-predict-air-plane-ticket-price---Data-Mining-Assignment\frontend
```

---

### Bước 2: Khởi tạo và Cài đặt thư viện

```bash
del package.json
npm init -y
npm install express
```

---

### Bước 3: Khởi chạy Server Frontend

```bash
node server.js
```

**Ghi chú:**  
Khi thấy:
```
Frontend server đang chạy tại: http://localhost:3000
```
→ Frontend đã hoạt động.

---

## 🚀 Hướng dẫn sử dụng

1. Truy cập: `http://localhost:3000`
2. Nhập:
   - Hãng hàng không
   - Điểm đi
   - Điểm đến
   - Thời gian bay
   - Số trạm dừng
3. Nhấn **"Dự Đoán Giá Vé"**
4. Xem kết quả hiển thị

---

## ⚠️ Lưu ý khi gặp lỗi

- **Connection Error** → Backend chưa chạy
- **Module not found** → chạy lại `pip install` hoặc `npm install`
- **Model error** → kiểm tra file `backend/model.pkl`
