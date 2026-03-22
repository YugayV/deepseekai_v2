"""
Streamlit Dashboard for EURUSD AI Trading Bot
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import joblib
import json
from datetime import datetime, timedelta
import os

st.set_page_config(page_title="EURUSD AI Trader", layout="wide")
st.title("🤖 EURUSD AI Trading Bot with DeepSeek")
st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load("models/xgb_regime_classifier.pkl")
        scaler = joblib.load("models/feature_scaler.pkl")
        with open("models/model_metadata.json", "r") as f:
            metadata = json.load(f)
        return model, scaler, metadata
    except:
        return None, None, None

model, scaler, metadata = load_models()

# Load portfolio state
@st.cache_data(ttl=60)
def load_portfolio():
    try:
        with open("data/portfolio_state.json", "r") as f:
            return json.load(f)
    except:
        return None

# Load trade history
@st.cache_data(ttl=60)
def load_trades():
    try:
        df = pd.read_csv("data/trade_history.csv")
        return df
    except:
        return pd.DataFrame()

# Fetch live data
@st.cache_data(ttl=60)
def fetch_live_data():
    ticker = yf.Ticker("EURUSD=X")
    df = ticker.history(period="3mo", interval="1d")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return df

# ============================================
# SIDEBAR - Metrics
# ============================================
st.sidebar.header("📊 Portfolio Status")
portfolio = load_portfolio()

if portfolio:
    col1, col2, col3 = st.sidebar.columns(3)
    col1.metric("Balance", f"${portfolio.get('balance', 0):.2f}")
    col2.metric("Equity", f"${portfolio.get('equity', 0):.2f}")
    pnl = (portfolio.get('equity', 0) - 10000) / 10000 * 100
    col3.metric("PnL", f"{pnl:.2f}%", delta=f"{pnl:.2f}%")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Open Positions")
    for pos in portfolio.get('positions', []):
        st.sidebar.text(f"{pos['side'].upper()}: {pos['size']:.2f} @ {pos['entry_price']:.5f}")
    
    st.sidebar.markdown(f"*Last price: {portfolio.get('last_price', 0):.5f}*")
else:
    st.sidebar.info("No portfolio data yet. Run the bot first.")

# ============================================
# MAIN - Charts
# ============================================
df_live = fetch_live_data()

# Price chart
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.7, 0.3],
                    vertical_spacing=0.05)

fig.add_trace(go.Candlestick(
    x=df_live.index[-100:],
    open=df_live['open'][-100:],
    high=df_live['high'][-100:],
    low=df_live['low'][-100:],
    close=df_live['close'][-100:],
    name="EURUSD"
), row=1, col=1)

# Volume
fig.add_trace(go.Bar(
    x=df_live.index[-100:],
    y=df_live['volume'][-100:],
    name="Volume",
    marker_color='steelblue'
), row=2, col=1)

fig.update_layout(height=600, title_text="EURUSD Price Action")
fig.update_xaxes(title_text="Date", row=2, col=1)
st.plotly_chart(fig, use_container_width=True)

# ============================================
# TRADE HISTORY
# ============================================
st.markdown("---")
st.subheader("📋 Trade History")

trades = load_trades()
if not trades.empty:
    st.dataframe(trades.tail(20), use_container_width=True)
else:
    st.info("No trades recorded yet.")

# ============================================
# MODEL INFO
# ============================================
st.markdown("---")
st.subheader("🤖 Model Information")

if metadata:
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", metadata.get('model_type', 'XGBoost'))
    col2.metric("Balanced Accuracy", f"{metadata.get('performance', {}).get('balanced_accuracy', 0):.2%}")
    col3.metric("Lookahead", f"{metadata.get('lookahead_days', 5)} days")
    
    st.markdown("**Top Features (Alligator + Fractals)**")
    st.json(metadata.get('feature_columns', [])[:10])
else:
    st.warning("Model not loaded. Train first.")

# ============================================
# REFRESH BUTTON
# ============================================
st.markdown("---")
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")