"""
Backtest script for the EURUSD AI Trading System
"""
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit

print("=" * 60)
print("BACKTESTING EURUSD AI TRADING SYSTEM")
print("=" * 60)

# Load model
model = joblib.load("models/voting_ensemble.pkl")
scaler = joblib.load("models/feature_scaler.pkl")
with open("models/model_metadata.json", "r") as f:
    import json
    metadata = json.load(f)

feature_cols = metadata['feature_columns']

# Load data
ticker = yf.Ticker("EURUSD=X")
df = ticker.history(period="5y", interval="1d")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.columns = ['open', 'high', 'low', 'close', 'volume']

# Calculate indicators (simplified for backtest)
# This would need full indicator calculation similar to bot.py
print("✅ Model loaded. Run full notebook for backtest.")