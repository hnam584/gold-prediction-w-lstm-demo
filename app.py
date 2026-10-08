"""
Flask backend for Gold Price LSTM Demo
"""
from flask import Flask, jsonify, request, send_from_directory
import numpy as np
import json, os, joblib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__, static_folder="static", static_url_path="")

# Load model & data at startup
MODEL = None
SCALER = None
CHART_DATA = None
LOOKBACK = 60

PREDICT_FN = None

def load_assets():
    global MODEL, SCALER, CHART_DATA, PREDICT_FN
    import tensorflow as tf
    MODEL  = tf.keras.models.load_model("model/lstm_gold.keras")
    SCALER = joblib.load("model/scaler.pkl")
    with open("model/chart_data.json", "r") as f:
        CHART_DATA = json.load(f)

    @tf.function
    def _step(x_in):
        return MODEL(x_in, training=False)

    # Warmup graph
    _step(tf.zeros((1, LOOKBACK, 1), dtype=tf.float32))
    PREDICT_FN = _step
    print("Assets loaded and graph warmed up.")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/chart-data")
def chart_data():
    return jsonify(CHART_DATA)

@app.route("/api/predict-future", methods=["POST"])
def predict_future():
    """Predict N days into the future using recursive forecasting."""
    data = request.get_json()
    n_days = int(data.get("days", 30))
    n_days = max(1, min(n_days, 1500))

    # Start from last LOOKBACK real prices (unscaled)
    last_seq = np.array(CHART_DATA["last_sequence"], dtype="float32").reshape(-1, 1)
    scaled_seq = SCALER.transform(last_seq).flatten().tolist()

    window = list(scaled_seq[-LOOKBACK:])
    scaled_preds = []

    for _ in range(n_days):
        x = np.array(window[-LOOKBACK:], dtype=np.float32).reshape(1, LOOKBACK, 1)
        pred_s = float(PREDICT_FN(x)[0, 0])
        scaled_preds.append(pred_s)
        window.append(pred_s)

    # Vectorized inverse transform all at once (super fast)
    real_preds = SCALER.inverse_transform(np.array(scaled_preds).reshape(-1, 1)).flatten()
    predictions = [round(float(p), 2) for p in real_preds]

    # Generate business-day dates from last_date + 1
    import pandas as pd
    last_date = pd.Timestamp(CHART_DATA["last_date"])
    future_dates = []
    d = last_date
    for _ in range(n_days):
        d += pd.offsets.BDay(1)
        future_dates.append(d.strftime("%Y-%m-%d"))

    return jsonify({"dates": future_dates, "prices": predictions})

if __name__ == "__main__":
    load_assets()
    app.run(debug=False, port=5000)
