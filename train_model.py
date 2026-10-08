"""
Train LSTM model for Gold Price Prediction
Dataset: goldstock.csv (2014-2024, ~2512 rows)
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

print("TensorFlow version:", tf.__version__)

# 1. LOAD & CLEAN DATA
df = pd.read_csv("goldstock.csv", index_col=0, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)
df = df.dropna(subset=["Close"])
print(f"Data: {len(df)} rows | {df['Date'].min().date()} to {df['Date'].max().date()}")

prices = df[["Close"]].values.astype("float32")

# 2. TRAIN / TEST SPLIT (80/20, chronological)
LOOKBACK = 60
split = int(len(prices) * 0.80)
train_raw = prices[:split]
test_raw  = prices[split:]

# 3. SCALING
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_raw)
test_scaled  = scaler.transform(test_raw)

# 4. CREATE SLIDING WINDOW
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, LOOKBACK)
full_test = np.concatenate([train_scaled[-LOOKBACK:], test_scaled], axis=0)
X_test, y_test = create_sequences(full_test, LOOKBACK)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

# 5. BUILD MODEL
tf.random.set_seed(42)
model = Sequential([
    Input(shape=(LOOKBACK, 1)),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.1),
    Dense(16, activation="relu"),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss="huber", metrics=["mae"])
model.summary()

# 6. TRAIN
callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
]
history = model.fit(
    X_train, y_train,
    epochs=150, batch_size=32,
    validation_split=0.1,
    callbacks=callbacks, verbose=1,
)

# 7. EVALUATE
pred_scaled = model.predict(X_test)
pred_prices = scaler.inverse_transform(pred_scaled)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = float(np.sqrt(mean_squared_error(actual_prices, pred_prices)))
mae  = float(mean_absolute_error(actual_prices, pred_prices))
mape = float(np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100)
r2   = float(1 - np.sum((actual_prices - pred_prices)**2) / np.sum((actual_prices - np.mean(actual_prices))**2))

print(f"\nRMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.4f}% | R2: {r2:.4f}")

# 8. SAVE
os.makedirs("model", exist_ok=True)
model.save("model/lstm_gold.keras")
joblib.dump(scaler, "model/scaler.pkl")

chart_data = {
    "dates_train": df["Date"][:split].dt.strftime("%Y-%m-%d").tolist(),
    "prices_train": [round(float(v), 2) for v in df["Close"][:split].tolist()],
    "dates_test": df["Date"][split:].dt.strftime("%Y-%m-%d").tolist(),
    "prices_test": [round(float(v), 2) for v in actual_prices.flatten().tolist()],
    "prices_pred": [round(float(v), 2) for v in pred_prices.flatten().tolist()],
    "metrics": {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape": round(mape, 4),
        "r2": round(r2, 4),
        "train_size": int(split),
        "test_size": int(len(test_raw)),
        "lookback": LOOKBACK,
        "epochs_trained": len(history.history["loss"]),
        "model_params": int(model.count_params()),
    },
    "loss_history": {
        "train_loss": [round(float(v), 6) for v in history.history["loss"]],
        "val_loss": [round(float(v), 6) for v in history.history["val_loss"]],
    },
    "last_sequence": [round(float(v), 4) for v in prices[-LOOKBACK:].flatten().tolist()],
    "last_date": df["Date"].iloc[-1].strftime("%Y-%m-%d"),
    "scaler_min": float(scaler.data_min_[0]),
    "scaler_max": float(scaler.data_max_[0]),
}

with open("model/chart_data.json", "w") as f:
    json.dump(chart_data, f)

print("Saved: model/lstm_gold.keras, model/scaler.pkl, model/chart_data.json")
