# 📈 LSTM Gold Price Predictor

Một ứng dụng Web Deep Learning dự đoán giá vàng thế giới (USD / ounce) sử dụng mạng nơ-ron bộ nhớ ngắn hạn dài (LSTM). Dự án bao gồm giao diện trực quan, bảng điều khiển phân tích số liệu và khả năng dự báo giá vàng trong tương lai.

## ✨ Tính năng nổi bật
- **Độ chính xác cao**: Đạt độ chính xác ~98.5% (MAPE = 1.54%, R² = 0.949) trên tập dữ liệu test.
- **Giao diện hiện đại**: Thiết kế Dark-mode, phong cách Glassmorphism sang trọng với biểu đồ tương tác Chart.js.
- **Dự đoán tương lai (Future Forecasting)**: Cho phép chọn bất kỳ ngày nào trong tương lai để dự đoán giá bằng phương pháp đệ quy (recursive forecasting).
- **Tốc độ siêu tốc**: Backend được tối ưu hóa bằng `@tf.function` của TensorFlow, cho tốc độ nội suy dự đoán 1 năm chỉ trong vòng chưa tới 1 giây.

## 🛠️ Công nghệ sử dụng
- **Machine Learning**: TensorFlow / Keras, Scikit-learn, Pandas, NumPy
- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, Vanilla JS, Chart.js

## 🚀 Hướng dẫn cài đặt

### 1. Yêu cầu hệ thống
- Python 3.8 trở lên

### 2. Cài đặt thư viện
Mở terminal và chạy lệnh sau để cài đặt các thư viện cần thiết:
```bash
pip install tensorflow flask scikit-learn pandas numpy joblib
```

### 3. Chạy ứng dụng
Khởi động server Flask:
```bash
python app.py
```
Mở trình duyệt và truy cập vào địa chỉ: `http://127.0.0.1:5000`

## 🧠 Kiến trúc mô hình
- 3 Lớp LSTM (128 → 64 → 32 units) kết hợp với các lớp Dropout (0.2, 0.1) để giảm thiểu Overfitting.
- Hàm mất mát (Loss function): Huber Loss (giúp model chống lại nhiễu/outlier cực tốt).
- Optimizer: Adam (Learning rate = 0.001) với Early Stopping & ReduceLROnPlateau.

## 📊 Dữ liệu (Dataset)
- Dữ liệu lịch sử giá vàng thế giới (`goldstock.csv`) trong vòng 10 năm (2014 - 2024) với hơn 2.500 điểm dữ liệu.
- Chuỗi thời gian (Time-series) sử dụng Lookback Window = 60 ngày giao dịch.
