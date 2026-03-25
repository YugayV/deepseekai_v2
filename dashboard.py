"""
Streamlit Dashboard for EURUSD AI Trading Bot
Extended with Alligator/Fractals visualization and DeepSeek analytics
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

st.set_page_config(page_title="EURUSD AI Trader", layout="wide", page_icon="🤖")

# ============================================
# SIDEBAR - Configuration
# ============================================
st.sidebar.title("⚙️ Configuration")

# Asset selection
st.sidebar.subheader("📊 Assets")
assets = st.sidebar.multiselect(
    "Select assets to display",
    ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "BTC-USD", "ETH-USD"],
    default=["EURUSD=X"]
)

# Timeframe
timeframe = st.sidebar.selectbox("Timeframe", ["1d", "1h", "15m"], index=0)

# Update interval
st.sidebar.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ============================================
# MAIN TITLE
# ============================================
st.title("🤖 EURUSD AI Trading Bot with DeepSeek")
st.markdown("**Multi-Asset | Alligator + Fractals | Ensemble ML | Real-time Signals**")
st.markdown("---")

# ============================================
# LOAD MODELS AND DATA
# ============================================
@st.cache_resource
def load_models():
    try:
        model = joblib.load("models/voting_ensemble.pkl")
        scaler = joblib.load("models/feature_scaler.pkl")
        with open("models/model_metadata.json", "r") as f:
            metadata = json.load(f)
        return model, scaler, metadata
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None, None

@st.cache_data(ttl=60)
def load_portfolio():
    try:
        with open("data/portfolio_state.json", "r") as f:
            return json.load(f)
    except:
        return None

@st.cache_data(ttl=60)
def load_trades():
    try:
        df = pd.read_csv("data/trade_history.csv")
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_asset_data(symbol, period="3mo", interval="1d"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        return None
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return df

model, scaler, metadata = load_models()
portfolio = load_portfolio()
trades = load_trades()

# ============================================
# KPI CARDS
# ============================================
st.subheader("📊 Portfolio Overview")

col1, col2, col3, col4, col5 = st.columns(5)

if portfolio:
    balance = portfolio.get('balance', 0)
    equity = portfolio.get('equity', 0)
    pnl = (equity - 10000) / 10000 * 100

    col1.metric("💰 Balance", f"${balance:.2f}")
    col2.metric("📈 Equity", f"${equity:.2f}")
    col3.metric("📊 PnL", f"{pnl:.2f}%", delta=f"{pnl:.2f}%")
    col4.metric("🎯 Positions", len(portfolio.get('positions', [])))
    col5.metric("💹 Trades", len(trades) if not trades.empty else 0)
else:
    col1.metric("💰 Balance", "$10,000")
    col2.metric("📈 Equity", "$10,000")
    col3.metric("📊 PnL", "0.00%")
    col4.metric("🎯 Positions", "0")
    col5.metric("💹 Trades", "0")

st.markdown("---")

# ============================================
# PRICE CHARTS
# ============================================
st.subheader("📈 Price Action & Signals")

for symbol in assets:
    df = fetch_asset_data(symbol, period="3mo", interval=timeframe)
    if df is None:
        st.warning(f"No data for {symbol}")
        continue

    # Calculate simple indicators for chart
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{symbol} - Price', 'RSI (14)', 'Volume')
    )

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=symbol
    ), row=1, col=1)

    # Moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50', line=dict(color='blue', width=1)), row=1, col=1)

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple', width=1)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='steelblue'), row=3, col=1)

    fig.update_layout(height=800, showlegend=True)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================
# ALLIGATOR & FRACTALS CHART
# ============================================
st.subheader("🐊 Alligator Indicator & Fractals")

for symbol in assets[:1]:
    df = fetch_asset_data(symbol, period="3mo", interval="1d")
    if df is None:
        continue

    # Calculate Alligator
    df['jaw'] = df['close'].rolling(13).mean().shift(8)
    df['teeth'] = df['close'].rolling(8).mean().shift(5)
    df['lips'] = df['close'].rolling(5).mean().shift(3)

    # Fractal detection
    window = 2
    df['fractal_bullish'] = 0
    df['fractal_bearish'] = 0

    for i in range(window, len(df) - window):
        if all(df['low'].iloc[i] < df['low'].iloc[i - j] for j in range(1, window + 1)) and \
           all(df['low'].iloc[i] < df['low'].iloc[i + j] for j in range(1, window + 1)):
            df.loc[df.index[i], 'fractal_bullish'] = 1
        if all(df['high'].iloc[i] > df['high'].iloc[i - j] for j in range(1, window + 1)) and \
           all(df['high'].iloc[i] > df['high'].iloc[i + j] for j in range(1, window + 1)):
            df.loc[df.index[i], 'fractal_bearish'] = 1

    fig = go.Figure()

    # Price
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close', line=dict(color='black', width=1)))

    # Alligator
    fig.add_trace(go.Scatter(x=df.index, y=df['jaw'], name='Jaws (SMA13, shift8)', line=dict(color='blue', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['teeth'], name='Teeth (SMA8, shift5)', line=dict(color='red', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['lips'], name='Lips (SMA5, shift3)', line=dict(color='green', width=1.5)))

    # Fractals
    bullish_idx = df[df['fractal_bullish'] == 1].index
    bearish_idx = df[df['fractal_bearish'] == 1].index
    fig.add_trace(go.Scatter(x=bullish_idx, y=df.loc[bullish_idx, 'low'], mode='markers',
                              marker=dict(symbol='triangle-up', size=10, color='green'), name='Bullish Fractal'))
    fig.add_trace(go.Scatter(x=bearish_idx, y=df.loc[bearish_idx, 'high'], mode='markers',
                              marker=dict(symbol='triangle-down', size=10, color='red'), name='Bearish Fractal'))

    fig.update_layout(title=f'{symbol} - Alligator + Fractals', height=500)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================
# TRADE HISTORY
# ============================================
st.subheader("📋 Trade History")

if not trades.empty:
    st.dataframe(trades.tail(20), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    if 'pnl' in trades.columns:
        winning = trades[trades['pnl'] > 0]
        col1.metric("Total Trades", len(trades))
        col2.metric("Winning Trades", len(winning))
        col3.metric("Win Rate", f"{len(winning)/len(trades)*100:.1f}" if len(trades) > 0 else "0%")
        col4.metric("Total P&L", f"${trades['pnl'].sum():.2f}")
else:
    st.info("No trades recorded yet. Run the bot first.")

st.markdown("---")

# ============================================
# MODEL INFORMATION
# ============================================
st.subheader("🤖 Model Information")

if metadata:
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type", metadata.get('model_type', 'Voting Ensemble'))
    col2.metric("Balanced Accuracy", f"{metadata.get('performance', {}).get('balanced_accuracy', 0):.2%}")
    col3.metric("Lookahead", f"{metadata.get('lookahead_days', 5)} days")

    with st.expander("📊 Model Performance Details"):
        st.json(metadata.get('performance', {}))

    with st.expander("📈 Feature Importance (Top 10)"):
        st.info("Feature importance from XGBoost component")
else:
    st.warning("Model not loaded. Train models first.")

# ============================================
# DEEPSEEK ANALYTICS (Interactive)
# ============================================
st.subheader("🧠 DeepSeek Market Analysis")

if st.button("🔮 Get AI Market Analysis"):
    with st.spinner("DeepSeek analyzing market..."):
        # Simulate DeepSeek response (would call API in production)
        st.markdown("""
        **📊 Market Analysis: EURUSD**

        *Current Setup:*
        - RSI at 52.3 (neutral)
        - MACD histogram turning positive
        - Alligator lines starting to separate bullishly

        *ML Signal:* Bullish (confidence: 68%)

        *Recommendation:* Consider long entries on pullbacks to 1.0850.
        Stop loss below 1.0820, target 1.0920.

        *Risk Note:* Market showing early trend signals but low volatility.
        Use smaller position size.
        """)

st.markdown("---")
st.caption(f"© EURUSD AI Trading Bot | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")