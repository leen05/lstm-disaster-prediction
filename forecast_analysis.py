import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load data
df = pd.read_csv("data.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df, label="Actual Data")
plt.axhline(y=100, color='r', linestyle='--', label="Disaster Threshold (100)")
plt.title("Realistic Time Series with Threshold")
plt.legend()
plt.show()

# Forecast (ARIMA model)
model = ARIMA(df, order=(1, 1, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=7)  # Next 7 days
print("Forecasted Values:\n", forecast)