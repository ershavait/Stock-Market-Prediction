# Stock Market Prediction (OHLC)

## 📌 Overview
This project predicts the **next trading day’s Open, High, Low, and Close (OHLC)**
prices using supervised machine learning regression models trained on historical
stock market data.

The complete pipeline includes data preprocessing, feature engineering, model
training, and deployment via a Streamlit web application.

---

## 📊 Dataset
- Source: Yahoo Finance (via `yfinance`)
- Instrument: NIFTY 50 (`^NSEI`)
- Frequency: Daily OHLC data
- Time Period: 2010 – Present

---

## 🧠 Feature Engineering
The following features are computed from day *t*:

- Daily Return
- 5-day Moving Average (MA₅)
- 10-day Moving Average (MA₁₀)
- 20-day Moving Average (MA₂₀)

These features are used to predict OHLC values for day *t+1*.

---

## ⚙️ Models Used
- StandardScaler for feature normalization
- Linear Regression models:
  - One model each for Open, High, Low, Close
- Loss Function: Mean Squared Error (MSE)

---

## 🚀 Application
The trained models are deployed using **Streamlit**.
The application:
- Fetches the latest available market data
- Computes technical indicators
- Predicts next-day OHLC values
- Automatically updates predictions with new data

---

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
