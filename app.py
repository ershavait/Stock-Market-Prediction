import yfinance as yf
import pandas as pd
import streamlit as st
import joblib
from datetime import timedelta
from pandas.tseries.offsets import BDay

# Configure Streamlit page layout
st.set_page_config(page_title="Stock OHLC Predictor", layout="wide")

# App title and purpose
st.title("📈 Stock Market OHLC Prediction App")
st.write(
    "Predicts **next trading day's Open, High, Low, Close (OHLC)** "
    "using trained regression models."
)

# Take stock/index symbol from user
symbol = st.text_input(
    "Enter Stock Symbol (e.g. ^NSEI, AAPL, RELIANCE.NS)",
    value="^NSEI"
)

# Run only when button is clicked
if st.button("Run Prediction"):

    st.success("Running prediction...")

    # Load trained scaler and OHLC regression models
    scaler = joblib.load("models/scaler.pkl")
    model_open  = joblib.load("models/model_open.pkl")
    model_high  = joblib.load("models/model_high.pkl")
    model_low   = joblib.load("models/model_low.pkl")
    model_close = joblib.load("models/model_close.pkl")

    # Download historical data up to day T
    df = yf.download(symbol, start="2015-01-01")

    # Stop execution if symbol is invalid
    if df.empty:
        st.error("No data found. Check the stock symbol.")
        st.stop()

    # Create features using past price information only
    df["Return"] = df["Close"].pct_change()
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()

    # Remove rows with incomplete feature values
    df = df.dropna()

    # Select same features used during training
    FEATURES = ["Return", "MA_5", "MA_10", "MA_20"]
    X = df[FEATURES]
    X_scaled = scaler.transform(X)

    # -------- Predict next trading day --------
    X_last = X_scaled[-1].reshape(1, -1)

    # Predict next-day OHLC values independently
    open_pred  = model_open.predict(X_last)[0]
    high_pred  = model_high.predict(X_last)[0]
    low_pred   = model_low.predict(X_last)[0]
    close_pred = model_close.predict(X_last)[0]

    # Compute next trading day (skip weekends)
    last_date = df.index[-1]
    predicted_date = last_date + BDay(1)

    # Display predicted results(Output)
    st.subheader(f"📅 Predicted OHLC for {predicted_date.date()}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Open",  f"{open_pred:.2f}")
    col2.metric("High",  f"{high_pred:.2f}")
    col3.metric("Low",   f"{low_pred:.2f}")
    col4.metric("Close", f"{close_pred:.2f}")

    # Show recent data used for prediction
    st.subheader("📊 Recent Market Data Used")
    st.dataframe(df.tail())




