import yfinance as yf
import pandas as pd
import numpy as np

tickers = ['AAPL', 'JPM', 'PG', 'XOM']
start_date = '2021-03-01'
end_date = '2026-03-01'

print("Downloading data...")

# Explicitly set auto_adjust=True. This adjusts all historical pricing for splits/dividends
# and ensures the 'Close' column contains the accurate data we need for quant modeling.
raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

# Extract the adjusted closing prices
data = raw_data['Close']

# Calculate daily logarithmic returns
log_returns = np.log(data / data.shift(1))

# Drop the first row (NaN due to the shift)
log_returns = log_returns.dropna()

print("\nFirst 5 rows of Log Returns:")
print(log_returns.head())

print("\nSummary Statistics (Look at the Min values for tail risk):")
print(log_returns.describe())













