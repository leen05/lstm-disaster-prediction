# 🌍 LSTM-Based Disaster Prediction System

A web-based AI/ML application for forecasting disasters using LSTM (Long Short-Term Memory) networks. Built with **Streamlit**, this tool allows users to upload sensor data and receive predictions to help in early disaster detection and risk mitigation.

---

## 📌 Features

- 📈 Time-series forecasting using LSTM neural networks
- 📊 Real-time data visualization with Plotly
- 📂 Upload CSV sensor data
- ⚠️ Predict disaster likelihood from environmental metrics
- 📉 Shows RMSE and MAE evaluation metrics
- 🔒 Handles missing or small datasets gracefully
- ☁️ Deployable via Streamlit Community Cloud

---

## 🧠 Tech Stack

- **Frontend/UI:** Streamlit
- **Data Handling:** Pandas, NumPy
- **Visualization:** Plotly
- **Model:** TensorFlow / Keras (LSTM)
- **Preprocessing:** MinMaxScaler (Sklearn)
- **Deployment:** Streamlit Cloud

---

## 🚀 How to Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/lstm-disaster-prediction.git
   cd lstm-disaster-prediction

Install dependencies

pip install -r requirements.txt
streamlit run main.py
