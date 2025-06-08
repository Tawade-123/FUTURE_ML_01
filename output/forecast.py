import pandas as pd
from prophet import Prophet

# Sample sales data: Replace this with your real dataset
data = {
    'ds': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'y': (50 + 2 * pd.Series(range(100)))  # example increasing sales
}
df = pd.DataFrame(data)

# Train Prophet model
model = Prophet()
model.fit(df)

# Create future dataframe for 30 days ahead
future = model.make_future_dataframe(periods=30)

# Predict future sales
forecast = model.predict(future)

# Extract the columns: 'ds' for date, 'yhat' for predicted sales
forecast_output = forecast[['ds', 'yhat']]

# Rename columns to match dashboard expectation
forecast_output.columns = ['date', 'forecast']

# Save to CSV (used in your dashboard upload)
forecast_output.to_csv('forecast_output.csv', index=False)

print("✅ Forecast CSV saved as 'forecast_output.csv'")
