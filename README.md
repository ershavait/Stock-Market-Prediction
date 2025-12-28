# 📈 Next-Day OHLC Prediction Using Machine Learning

A time-series regression–based machine learning project that predicts the **Open, High, Low, and Close (OHLC)** prices of the **next trading day (T+1)** using historical market data available up to day **T**.

This project emphasizes **correct mathematical formulation, feature engineering, and prevention of information leakage**, rather than speculative trading performance.

---

## 🔍 Project Overview

Financial markets generate noisy and dynamic time-series data.  
The goal of this project is to model the mapping:

> **Today’s market state → Tomorrow’s OHLC prices**

using engineered technical indicators and classical machine learning regression models.

Key characteristics:
- Regression-based (not classification)
- Time-series–aware formulation
- Separate models for each OHLC component
- Emphasis on interpretability and correctness

---

## 🧠 Problem Formulation

Given historical daily market data up to day **t**, predict:

\[
[O_{t+1},\ H_{t+1},\ L_{t+1},\ C_{t+1}]
\]

using only information available up to day **t**.

---

## 📊 Dataset

- **Source:** Yahoo Finance API  
- **Instrument:** NIFTY 50 Index (`^NSEI`)  
- **Frequency:** Daily  
- **Time Range (Training):** Approximately **2010 → recent years**

### Raw Data Fields
- Open
- High
- Low
- Close
- Volume

---

## ⚙️ Feature Engineering

For each trading day \(t\), the following features are constructed:

- Daily return  
- 5-day moving average  
- 10-day moving average  
- 20-day moving average  

### Feature Vector
\[
X_t = [r_t,\ MA_5(t),\ MA_{10}(t),\ MA_{20}(t)]
\]

These features capture short-term momentum and trend information.

---

## 🔢 Feature Scaling

All features are standardized using **StandardScaler**:

\[
X^{(\text{scaled})} = \frac{X - \mu}{\sigma}
\]

- Mean (\(\mu\)) and standard deviation (\(\sigma\)) are computed only on training data
- The same scaler is reused during inference (`scaler.pkl`)
- Prevents numerical instability and information leakage

---

## 🤖 Model Architecture

- **Model Type:** Linear Regression (Scikit-learn)
- **Approach:** Separate models for each OHLC component

| Target | Model File |
|------|-----------|
| Open | `model_open.pkl` |
| High | `model_high.pkl` |
| Low  | `model_low.pkl`  |
| Close| `model_close.pkl` |

### Regression Equation
\[
\hat{P}_{t+1} =
w_1 r_t + w_2 MA_5(t) + w_3 MA_{10}(t) + w_4 MA_{20}(t) + b
\]

---

## 📉 Loss Function & Training

- **Loss Function:** Mean Squared Error (MSE)

\[
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
\]

- Models are trained using historical data while preserving temporal order
- No future information is used during training or inference

---

## 📈 Evaluation Methodology

- Historical backtesting on unseen data
- Error-based evaluation using MSE
- Visual inspection of predicted vs actual price trends

The goal is realistic forecasting rather than trading optimization.

---

## 🖥️ Application (Deployment)

An interactive **Streamlit web application** is provided for inference.

### Features:
- User inputs a stock or index symbol
- Recent market data is fetched
- Next trading day’s OHLC is predicted
- Results are displayed in a dashboard

Main file:

```bash
pip install -r requirements.txt
streamlit run app.py

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
 streamlit run app.py

