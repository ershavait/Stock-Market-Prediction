import yfinance as yf
import pandas as pd
import streamlit as st
import joblib
from datetime import timedelta
from pandas.tseries.offsets import BDay

# ---------------- Page config ----------------
st.set_page_config(page_title="Stock OHLC Predictor", layout="wide")

st.title("📈 Stock Market OHLC Prediction App")
st.write(
    "Predicts **next trading day's Open, High, Low, Close (OHLC)** "
    "using trained regression models."
)

# ---------------- Input ----------------
symbol = st.text_input(
    "Enter Stock Symbol (e.g. ^NSEI, AAPL, RELIANCE.NS)",
    value="^NSEI"
)

# ---------------- Button ----------------
if st.button("Run Prediction"):

    st.success("Button clicked. Running prediction...")

    try:
        # -------- Load models --------
       scaler = joblib.load("models/scaler.pkl")
model_open  = joblib.load("models/model_open.pkl")
model_high  = joblib.load("models/model_high.pkl")
model_low   = joblib.load("models/model_low.pkl")
model_close = joblib.load("models/model_close.pkl")


        # -------- Download data --------
        df = yf.download(symbol, start="2015-01-01")

        if df.empty:
            st.error("No data found. Check the stock symbol.")
            st.stop()

        # -------- Feature engineering --------
        df["Return"] = df["Close"].pct_change()
        df["MA_5"] = df["Close"].rolling(5).mean()
        df["MA_10"] = df["Close"].rolling(10).mean()
        df["MA_20"] = df["Close"].rolling(20).mean()

        df = df.dropna()

        FEATURES = ["Return", "MA_5", "MA_10", "MA_20"]
        X = df[FEATURES]

        X_scaled = scaler.transform(X)

        # -------- Predict next day --------
        X_last = X_scaled[-1].reshape(1, -1)

        open_pred  = model_open.predict(X_last)[0]
        high_pred  = model_high.predict(X_last)[0]
        low_pred   = model_low.predict(X_last)[0]
        close_pred = model_close.predict(X_last)[0]

        # -------- Predicted date --------
        last_date = df.index[-1]

        predicted_date = last_date + BDay(1)


        # -------- Output --------
        st.subheader(f"📅 Predicted OHLC for {predicted_date.date()}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Open",  f"{open_pred:.2f}")
        col2.metric("High",  f"{high_pred:.2f}")
        col3.metric("Low",   f"{low_pred:.2f}")
        col4.metric("Close", f"{close_pred:.2f}")

        st.subheader("📊 Recent Market Data Used")
        st.dataframe(df.tail())

    except Exception as e:
        st.error("Something went wrong")
        st.exception(e)

