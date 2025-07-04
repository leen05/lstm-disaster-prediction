import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and prepare data
df = pd.read_csv("data.csv")
print("First 5 rows:\n", df.head())
print("\nColumn names:", df.columns)

# Convert 'Date' to datetime and set index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Rename column if needed
if 'Value' not in df.columns:
    df.rename(columns={df.columns[0]: 'Value'}, inplace=True)

# Drop missing values
df.dropna(subset=['Value'], inplace=True)
print(f"Total usable rows: {len(df)}")

# Scale the data
data = df['Value'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Automatically choose sequence length - be more conservative
min_data_needed = 10  # Minimum data points needed
if len(scaled_data) < min_data_needed:
    print(f"⚠️  Warning: You only have {len(scaled_data)} data points.")
    print(f"   For proper LSTM training, you need at least {min_data_needed} points.")
    print("   Consider adding more data to your CSV file.")
    
    # For demonstration purposes, let's create synthetic data
    print("\n🔄 Generating synthetic data for demonstration...")
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    # Create a simple trend with some noise
    synthetic_values = np.linspace(1, 50, 50) + np.random.normal(0, 2, 50)
    synthetic_df = pd.DataFrame({'Date': dates, 'Value': synthetic_values})
    synthetic_df.set_index('Date', inplace=True)
    
    # Use synthetic data
    data = synthetic_df['Value'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(data)
    print(f"Using {len(scaled_data)} synthetic data points")

# Choose sequence length
max_seq_len = 10  # Reduced from 30
seq_length = min(max_seq_len, len(scaled_data) // 3)  # Use 1/3 of data length
seq_length = max(seq_length, 2)  # Minimum sequence length of 2
print(f"\nUsing sequence length: {seq_length}")

# Create sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, seq_length)
print(f"Total sequences created: {len(X)}")

if len(X) < 5:  # Need at least 5 sequences for meaningful training
    raise ValueError(f"❌ Only {len(X)} sequences created. Need at least 5 for training.")

# Reshape input for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train-test split - ensure we get at least 1 training sample
split = max(1, int(len(X) * 0.8))  # At least 1 training sample
split = min(split, len(X) - 1)     # Leave at least 1 for testing

X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Build the model with smaller architecture for small datasets
model = Sequential()
model.add(LSTM(min(20, len(X_train)), input_shape=(seq_length, 1)))  # Smaller LSTM
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model with adjusted batch size
batch_size = min(8, len(X_train))  # Don't exceed training data size
epochs = min(20, max(5, len(X_train)))  # Adjust epochs based on data size

print(f"\nTraining model with batch_size={batch_size}, epochs={epochs}...")
history = model.fit(X_train, y_train, 
                   epochs=epochs, 
                   batch_size=batch_size,
                   verbose=1,
                   validation_split=0.1 if len(X_train) > 10 else 0)

# Predict and inverse transform
if len(X_test) > 0:
    predicted = model.predict(X_test)
    predicted = scaler.inverse_transform(predicted)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(y_test_inv, label='Actual', marker='o')
    plt.plot(predicted, label='Predicted', marker='s')
    plt.title("LSTM Disaster Forecast - Test Results")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Training History")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and print accuracy metrics
    mse = np.mean((y_test_inv - predicted) ** 2)
    mae = np.mean(np.abs(y_test_inv - predicted))
    print(f"\n📊 Model Performance:")
    print(f"   Mean Squared Error (MSE): {mse:.4f}")
    print(f"   Mean Absolute Error (MAE): {mae:.4f}")
    
else:
    print("❌ No test data available for prediction.")

print("\n✅ Training completed!")
print("\n💡 Tips for better results:")
print("   1. Add more data points to your CSV (aim for 100+ rows)")
print("   2. Use real disaster-related time series data")
print("   3. Include multiple features (temperature, humidity, etc.)")
print("   4. Experiment with different sequence lengths")