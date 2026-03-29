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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Multi-Asset AI Trader", layout="wide", page_icon="🤖")

# ============================================
# SIDEBAR - Configuration
# ============================================
st.sidebar.title("⚙️ Configuration")

# Assets
st.sidebar.subheader("📊 Assets")
assets = st.sidebar.multiselect(
    "Select assets to display",
    ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "BTC-USD", "ETH-USD"],
    default=["EURUSD=X"]
)

# Risk Settings
st.sidebar.subheader("🛡️ Risk Management")
manual_tp = st.sidebar.slider("Take Profit (%)", 0.5, 10.0, 4.0, 0.5)
manual_sl = st.sidebar.slider("Stop Loss (%)", 0.5, 5.0, 2.0, 0.5)
leverage = st.sidebar.selectbox("Leverage", [1, 2, 5, 10, 20, 50, 100], index=2)

# Start Trading Button in Sidebar
col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("🚀 Start All", use_container_width=True):
    st.sidebar.success("All pairs started!")
    with open("data/bot_command.json", "w") as f:
        json.dump({"command": "start_all", "tp": manual_tp, "sl": manual_sl, "leverage": leverage, "time": str(datetime.now())}, f)

if col_btn2.button("🛑 Stop All", use_container_width=True):
    st.sidebar.warning("All stopped!")
    with open("data/bot_command.json", "w") as f:
        json.dump({"command": "stop_all", "time": str(datetime.now())}, f)

# Timeframe
timeframe = st.sidebar.selectbox("Timeframe", ["1d", "1h", "15m"], index=1)

# Update interval
st.sidebar.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ============================================
# MAIN TITLE
# ============================================
st.title("🤖 Multi-Asset AI Trading Bot with DeepSeek")
st.markdown("**Alligator + Fractals | Ensemble ML | Real-time Signals**")
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
# KPI CARDS & PORTFOLIO DETAILS
# ============================================
st.subheader("📊 Portfolio Overview")

col1, col2, col3, col4, col5 = st.columns(5)

if portfolio:
    balance = portfolio.get('balance', 0)
    equity = portfolio.get('equity', 0)
    pnl = (equity - 10000) / 10000 * 100
    positions = portfolio.get('positions', {})

    col1.metric("💰 Balance", f"${balance:.2f}")
    col2.metric("📈 Equity", f"${equity:.2f}")
    col3.metric("📊 PnL", f"{pnl:.2f}%", delta=f"{pnl:.2f}%")
    col4.metric("🎯 Positions", len(positions))
    col5.metric("💹 Trades", len(trades) if not trades.empty else 0)

    # Detailed Portfolio View
    if st.button("📂 Show Detailed Portfolio / Positions"):
        st.markdown("### 🎯 Active Positions")
        if not positions:
            st.info("No active positions.")
        else:
            for symbol, pos in positions.items():
                with st.container():
                    c1, c2, c3, c4 = st.columns(4)
                    c1.write(f"**{symbol}**")
                    c2.write(f"Side: {pos['side'].upper()}")
                    c3.write(f"Entry: {pos['entry_price']:.5f}")
                    c4.write(f"Size: {pos['size']:.2f}")
                    st.divider()
else:
    # ... existing code for no portfolio ...
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

for symbol in assets:
    df = fetch_asset_data(symbol, period="3mo", interval=timeframe)
    if df is None:
        continue

    # Calculate Alligator
    # Jaws (Blue line): 13-period smoothed moving average, shifted 8 bars into the future.
    # Teeth (Red line): 8-period smoothed moving average, shifted 5 bars into the future.
    # Lips (Green line): 5-period smoothed moving average, shifted 3 bars into the future.
    df['jaw'] = df['close'].rolling(13).mean().shift(8)
    df['teeth'] = df['close'].rolling(8).mean().shift(5)
    df['lips'] = df['close'].rolling(5).mean().shift(3)

    # Fractal detection (Williams Fractals)
    window = 2
    df['fractal_bullish'] = 0
    df['fractal_bearish'] = 0

    # ... existing code for fractal calculation ...
    for i in range(window, len(df) - window):
        if all(df['low'].iloc[i] < df['low'].iloc[i - j] for j in range(1, window + 1)) and \
           all(df['low'].iloc[i] < df['low'].iloc[i + j] for j in range(1, window + 1)):
            df.loc[df.index[i], 'fractal_bullish'] = 1
        if all(df['high'].iloc[i] > df['high'].iloc[i - j] for j in range(1, window + 1)) and \
           all(df['high'].iloc[i] > df['high'].iloc[i + j] for j in range(1, window + 1)):
            df.loc[df.index[i], 'fractal_bearish'] = 1

    fig = go.Figure()

    # Price Candlestick instead of just line
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        opacity=0.4
    ))

    # Alligator
    fig.add_trace(go.Scatter(x=df.index, y=df['jaw'], name='Jaws (13,8)', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['teeth'], name='Teeth (8,5)', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['lips'], name='Lips (5,3)', line=dict(color='green', width=2)))

    # Fractals
    bullish_idx = df[df['fractal_bullish'] == 1].index
    bearish_idx = df[df['fractal_bearish'] == 1].index
    fig.add_trace(go.Scatter(x=bullish_idx, y=df.loc[bullish_idx, 'low'], mode='markers',
                              marker=dict(symbol='triangle-up', size=12, color='green'), name='Bullish Fractal'))
    fig.add_trace(go.Scatter(x=bearish_idx, y=df.loc[bearish_idx, 'high'], mode='markers',
                              marker=dict(symbol='triangle-down', size=12, color='red'), name='Bearish Fractal'))

    fig.update_layout(title=f'{symbol} ({timeframe}) - Alligator + Fractals', height=600, xaxis_rangeslider_visible=False)
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
# DEEPSEEK ANALYTICS & CONTROL
# ============================================
st.subheader("🧠 DeepSeek AI Control & Analysis")

# Get API Key
api_key = os.getenv("OPENROUTER_API_KEY")

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    if st.button("🚀 Start AI Trading (All Pairs)", use_container_width=True):
        st.success("AI Trading command sent to bot! Processing all symbols...")
        # Здесь можно создать флаг-файл для бота, если нужно
        with open("data/bot_command.json", "w") as f:
            json.dump({"command": "start", "time": str(datetime.now())}, f)

with col_ctrl2:
    if st.button("🛑 Stop AI Trading", use_container_width=True):
        st.warning("Stop command sent. Bot will finish current cycle.")
        with open("data/bot_command.json", "w") as f:
            json.dump({"command": "stop", "time": str(datetime.now())}, f)

st.markdown("---")

for symbol in assets:
    st.write(f"### 🔮 AI Analysis for {symbol}")
    
    # Show last known signal if exists
    if portfolio and symbol in portfolio.get('positions', {}):
        pos = portfolio['positions'][symbol]
        st.info(f"📍 Active Position: {pos['side'].upper()} | Entry: {pos['entry_price']:.5f}")

    if st.button(f"Generate New Analysis for {symbol}", key=f"deepseek_{symbol}"):
        if not api_key:
            st.error("API Key not found!")
            continue
            
        with st.spinner(f"DeepSeek analyzing {symbol}..."):
            try:
                import openai
                client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
                
                df_analysis = fetch_asset_data(symbol, period="1mo", interval=timeframe)
                if df_analysis is not None:
                    latest = df_analysis.iloc[-1]
                    
                    # More detailed prompt to ensure variety
                    prompt = f"""
                    Technical Analysis for {symbol} ({timeframe}).
                    Current Price: {latest['close']:.5f}
                    Last 5 closes: {list(df_analysis['close'].tail(5).values)}
                    Leverage set: {leverage}x
                    
                    Provide a concise analysis including:
                    1. Market Trend & Sentiment
                    2. RSI and Volume interpretation
                    3. Specific Entry/SL/TP levels (User wants TP around {manual_tp}% and SL around {manual_sl}%)
                    4. Recommendation on Leverage (Is {leverage}x safe?)
                    5. Risk rating (1-10)
                    """
                    
                    response = client.chat.completions.create(
                        model="deepseek/deepseek-chat",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7
                    )
                    
                    st.success(f"Analysis for {symbol} updated!")
                    st.markdown(response.choices[0].message.content)
                    
                    # Manual trade button for this specific pair
                    if st.button(f"🚀 Trade {symbol} ONLY", key=f"trade_single_{symbol}"):
                        with open("data/bot_command.json", "w") as f:
                            json.dump({"command": "start_single", "symbol": symbol, "tp": manual_tp, "sl": manual_sl, "leverage": leverage, "time": str(datetime.now())}, f)
                        st.success(f"Command to trade {symbol} sent!")
                else:
                    st.error(f"No data for {symbol}")
            except Exception as e:
                st.error(f"Error: {e}")
    st.markdown("---")
st.caption(f"© EURUSD AI Trading Bot | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")