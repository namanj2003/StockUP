import os
import logging
import warnings
import requests
import numpy as np
import pandas as pd
import datetime

import streamlit as st
import plotly.graph_objects as go

from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
load_dotenv()

st.set_page_config(page_title="LSTM-Only Upstox Example", layout="wide")

###############################################################################
# 1. Fetch daily OHLCV stock data from Upstox
###############################################################################
def fetch_upstox_data(instrument_key, start_date, end_date):
    headers_stock = {
        "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",
        "Accept": "application/json"
    }
    url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/day/{end_date}/{start_date}"
    logging.info(f"Fetching Upstox data from: {url}")

    response = requests.get(url, headers=headers_stock)
    response.raise_for_status()

    data = response.json().get("data", {}).get("candles", [])
    if not data:
        raise ValueError("No stock data returned for the given date range.")

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df.sort_values("timestamp", ascending=True, inplace=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    st.write("Before dropna(): df.shape =", df.shape)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()
    st.write("After dropna(): df.shape  =", df.shape)

    df.sort_values("timestamp", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["day_gap"] = df["timestamp"].diff().dt.days
    gap_counts = df["day_gap"].value_counts(dropna=False).sort_index()
    st.write("Gap summary (in days):")
    st.write(gap_counts)

    return df

###############################################################################
# 2. Forward-Fill to Daily Frequency (Fill missing days with previous trading day's value)
###############################################################################
def forward_fill_daily(df, date_col="Date"):
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)
    full_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_dates, method="ffill")
    df.index.name = date_col
    df = df.reset_index()
    return df

###############################################################################
# 3. Create Sequences for LSTM
###############################################################################
def create_sequences(data_arr, window=60):
    X, y = [], []
    for i in range(window, len(data_arr)):
        X.append(data_arr[i-window:i, 0])
        y.append(data_arr[i, 0])
    return np.array(X), np.array(y)

###############################################################################
# 4. LSTM Pipeline: Train on Train/Test Split but Predict Entire Series
###############################################################################
def run_lstm_entire_prediction(instrument_key, start_date, end_date):
    # A) Fetch data
    df = fetch_upstox_data(instrument_key, start_date, end_date)
    if df.empty:
        st.warning("No data found. Check date range or instrument key.")
        return

    st.write("Raw Data (Head):")
    st.dataframe(df.head())

    # B) Convert to daily frequency with forward-fill
    df["Date"] = df["timestamp"].dt.date
    daily_df = df[["Date", "close"]].copy()
    daily_df = forward_fill_daily(daily_df, date_col="Date")
    st.write("After forward-fill (Head):")
    st.dataframe(daily_df.head())

    # C) Prepare dataset (use entire daily_df for prediction)
    daily_df.sort_values("Date", inplace=True)
    daily_df.reset_index(drop=True, inplace=True)
    daily_df["Actual"] = daily_df["close"]
    daily_df.drop(columns=["close"], inplace=True)
    dataset = daily_df["Actual"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # D) Train/Test Split (for training and metrics)
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    window_size = 60
    X_train, y_train = create_sequences(train_data, window_size)
    X_test, y_test = create_sequences(test_data, window_size)
    X_train = X_train.reshape((X_train.shape[0], window_size, 1))
    X_test = X_test.reshape((X_test.shape[0], window_size, 1))

    # E) Build and train LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    st.write("Training LSTM model on train portion (5 epochs, batch_size=1)...")
    model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=2)

    # F) Create sequences for the entire dataset so we can predict for all days from index=window_size onward
    X_all, y_all = create_sequences(scaled_data, window_size)
    X_all = X_all.reshape((X_all.shape[0], window_size, 1))
    predictions_all = model.predict(X_all)
    predictions_all = scaler.inverse_transform(predictions_all.reshape(-1, 1))

    # G) Create a continuous column "Predicted" starting at index = window_size
    daily_df["Predicted"] = np.nan
    pred_indices = np.arange(window_size, window_size + len(predictions_all))
    daily_df.loc[pred_indices, "Predicted"] = predictions_all.ravel()

    # H) Mark train/test rows for reference (optional)
    daily_df["DataType"] = "Train"
    daily_df.loc[train_size:, "DataType"] = "Test"

    # I) Plot the entire series: Actual (continuous) and Predicted (from window_size onward)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_df["Date"],
        y=daily_df["Actual"],
        mode="lines",
        name="Actual",
        line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=daily_df["Date"],
        y=daily_df["Predicted"],
        mode="lines",
        name="Predicted",
        line=dict(color="red")
    ))
    fig.update_layout(
        title="LSTM-Only: Entire Series Predictions",
        xaxis=dict(type="date", tickangle=45),
        yaxis_title="Price",
        legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0)")
    )
    st.plotly_chart(fig, use_container_width=True)

    # J) Evaluate test metrics (only on test portion)
    test_start_idx = max(train_size, window_size)
    df_test = daily_df.iloc[test_start_idx:].copy()
    df_test = df_test.dropna(subset=["Predicted"])
    mae = mean_absolute_error(df_test["Actual"], df_test["Predicted"])
    rmse = np.sqrt(mean_squared_error(df_test["Actual"], df_test["Predicted"]))
    r2 = r2_score(df_test["Actual"], df_test["Predicted"])

    st.write("### Test Metrics (on test portion)")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("R² Score", f"{r2*100:.2f}%")

    # K) Forecast next day using last window_size days
    last_window = scaled_data[-window_size:]
    X_next = last_window.reshape((1, window_size, 1))
    next_day_scaled = model.predict(X_next)
    next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]
    st.write("### Next-Day Forecast")
    st.write(f"Predicted closing price for {instrument_key} on the next day: **{next_day_price:.2f}**")

    model_save_path = "improved_lstm_only_model_final.keras"
    model.save(model_save_path)
    
    st.success(f"Model saved as {model_save_path}")

    return model, scaler, model_save_path

###############################################################################
# MAIN
###############################################################################
def main():
    st.title("LSTM-Only Entire Series Predictions (No Gaps)")
    default_instrument_key = "NSE_EQ|INE542W01025"  # example instrument key
    default_start_date = "2018-01-01"
    default_end_date = datetime.date.today().strftime("%Y-%m-%d")

    instrument_key = st.text_input("Instrument Key", value=default_instrument_key)
    start_date = st.text_input("Start Date (YYYY-MM-DD)", value=default_start_date)
    end_date = st.text_input("End Date (YYYY-MM-DD)", value=default_end_date)

    if st.button("Run LSTM-Only Model"):
        try:
            run_lstm_entire_prediction(instrument_key, start_date, end_date)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
