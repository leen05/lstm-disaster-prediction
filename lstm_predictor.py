import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Configuration Parameters (from your UI)
SEQUENCE_LENGTH = 60  # Using max configured value
EPOCHS = 500
LSTM_UNITS = 256
DROPOUT_RATE = 0.5
DISASTER_THRESHOLD = 100.0

# 1. Data Preparation
def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df

def preprocess_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Value']])
    return scaled_data, scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length-1):
        X.append(data[i:(i+seq_length)])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# 2. Model Architecture - CORRECTED VERSION
def build_model(seq_length, lstm_units, dropout_rate):
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(seq_length, 1)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# 3. Training with Validation
def train_model(model, X_train, y_train):
    early_stop = EarlyStopping(monitor='val_loss', patience=20)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    return history

# 4. Dynamic Threshold Calculation
def calculate_dynamic_threshold(predictions, window=30, multiplier=1.2):
    rolling_avg = pd.Series(predictions.flatten()).rolling(window).mean()
    return rolling_avg.max() * multiplier

# 5. Prediction with Alerts
def predict_with_alerts(model, X_test, threshold):
    predictions = model.predict(X_test)
    alerts = np.where(predictions > threshold, 1, 0)
    return predictions, alerts

# Main Execution
if __name__ == "__main__":
    # Load and preprocess data
    df = load_data('data.csv')
    scaled_data, scaler = preprocess_data(df)
    
    # Create sequences
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)
    
    # Split data (80% train, 20% test)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build and train model
    model = build_model(SEQUENCE_LENGTH, LSTM_UNITS, DROPOUT_RATE)
    history = train_model(model, X_train, y_train)
    
    # Make predictions
    test_predictions, _ = predict_with_alerts(model, X_test, DISASTER_THRESHOLD)
    dynamic_threshold = calculate_dynamic_threshold(test_predictions)
    
    # Compare thresholds
    print(f"Fixed Threshold: {DISASTER_THRESHOLD}")
    print(f"Dynamic Threshold: {dynamic_threshold}")
    
    # Visualize results
    plt.figure(figsize=(12,6))
    plt.plot(test_predictions, label='Predictions')
    plt.axhline(DISASTER_THRESHOLD, color='r', linestyle='--', label='Fixed Threshold')
    plt.axhline(dynamic_threshold, color='g', linestyle=':', label='Dynamic Threshold')
    plt.legend()
    plt.title("Prediction Threshold Analysis")
    plt.savefig('threshold_analysis.png')
    plt.show()