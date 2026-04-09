"""
Streamlit Dashboard for EURUSD AI Trading Bot
Extended with Alligator/Fractals visualization and AI analytics
"""

import os
import threading
import asyncio

DEFAULT_STREAMLIT_PORT = 8501

def _resolve_streamlit_port() -> int:
    raw = os.getenv("PORT")
    if raw is None:
        return DEFAULT_STREAMLIT_PORT
    raw = str(raw).strip()
    if not raw:
        return DEFAULT_STREAMLIT_PORT
    try:
        port = int(raw)
    except Exception:
        return DEFAULT_STREAMLIT_PORT
    if port <= 0 or port > 65535:
        return DEFAULT_STREAMLIT_PORT
    return port

STREAMLIT_PORT = _resolve_streamlit_port()

os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
os.environ.setdefault("STREAMLIT_SERVER_PORT", str(STREAMLIT_PORT))
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import joblib
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Multi-Asset AI Trader", layout="wide", page_icon="🤖")

RUN_BOT_IN_PROCESS = (os.getenv("RUN_BOT_IN_PROCESS") or "true").strip().lower() == "true"

@st.cache_resource
def _ensure_bot_running():
    import bot as bot_module

    def _runner():
        try:
            asyncio.run(bot_module.main())
        except Exception as e:
            print(f"BOT_BACKGROUND_ERROR: {e}", flush=True)

    t = threading.Thread(target=_runner, daemon=True, name="bot_background")
    t.start()
    return True

if RUN_BOT_IN_PROCESS:
    _ensure_bot_running()

# ============================================
# SIDEBAR - Configuration
# ============================================
st.sidebar.title("⚙️ Configuration")

DATA_DIR = os.getenv("TRADEBOT_DATA_DIR", "data")
CMD_PATH = os.path.join(DATA_DIR, "bot_command.json")
os.makedirs(DATA_DIR, exist_ok=True)

BOT_API_URL = (os.getenv("BOT_API_URL") or "").rstrip("/")

def _post_bot_command(payload: dict):
    if RUN_BOT_IN_PROCESS:
        try:
            import bot as bot_module
            bot_instance = getattr(bot_module, "bot_instance", None)
            bot_loop = getattr(bot_module, "BOT_LOOP", None)
            if bot_instance and bot_loop:
                fut = asyncio.run_coroutine_threadsafe(bot_instance.apply_command(payload), bot_loop)
                fut.result(timeout=5)
                return {"ok": True}
        except Exception:
            pass

    if BOT_API_URL:
        import urllib.request
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            BOT_API_URL + "/command",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")

    with open(CMD_PATH, "w") as f:
        json.dump(payload, f)
    return {"ok": True}


def _get_bot_json(path: str):
    if not BOT_API_URL:
        return None
    import urllib.request
    with urllib.request.urlopen(BOT_API_URL + path, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")

# Assets
st.sidebar.subheader("📊 Assets")
assets = st.sidebar.multiselect(
    "Select assets to display",
    ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "BTC-USD", "ETH-USD"],
    default=["EURUSD=X"]
)

# Trading Mode
st.sidebar.subheader("🔌 Trading Mode")
trading_mode = st.sidebar.radio("Select Mode", ["Demo (Paper)", "Real (API)"], index=0)
is_real = trading_mode == "Real (API)"

if is_real:
    exchange_id = st.sidebar.selectbox("Exchange", ["bybit", "binance"], index=0)
    api_key_input = st.sidebar.text_input("API Key", type="password")
    api_secret_input = st.sidebar.text_input("API Secret", type="password")
    if st.sidebar.button("💾 Save API Config"):
        _post_bot_command({
            "command": "update_api",
            "mode": "real",
            "exchange": exchange_id,
            "key": api_key_input,
            "secret": api_secret_input,
            "time": str(datetime.now())
        })
        st.sidebar.success("API Config saved!")
else:
    if st.sidebar.button("🔄 Reset to Demo"):
        _post_bot_command({"command": "update_api", "mode": "demo", "time": str(datetime.now())})
        st.sidebar.info("Switched to Demo")

# Risk Settings
st.sidebar.subheader("🛡️ Risk Management")
manual_tp = st.sidebar.slider("Take Profit (%)", 0.5, 10.0, 4.0, 0.5)
manual_sl = st.sidebar.slider("Stop Loss (%)", 0.5, 5.0, 2.0, 0.5)
leverage = st.sidebar.selectbox("Leverage", [1, 2, 5, 10, 20, 50, 100], index=2)

st.sidebar.subheader("🕒 Demo Limits")
max_trades_per_day = st.sidebar.slider("Max trades per day", 1, 20, 5, 1)
daily_tp_target_percent = st.sidebar.slider("Daily TP target (%)", 1.0, 50.0, 10.0, 1.0)

st.sidebar.subheader("🧠 Strategy Filters")
strategy_label = st.sidebar.selectbox("Strategy mode", ["Classic", "Pro", "Mix"], index=0)
strategy_mode = "classic" if strategy_label == "Classic" else "pro" if strategy_label == "Pro" else "mix"
block_weak_signals = st.sidebar.checkbox("Block weak signals", value=True)
cooldown_bars = st.sidebar.slider("Cooldown (bars)", 0, 12, 3, 1)
use_atr_risk = st.sidebar.checkbox("Use ATR-based TP/SL", value=True)
atr_sl_mult = st.sidebar.slider("ATR SL multiple", 0.5, 3.0, 1.5, 0.1)
atr_tp_mult = st.sidebar.slider("ATR TP multiple", 1.0, 6.0, 2.5, 0.1)
if st.sidebar.button("✅ Apply Filters", width='stretch'):
    _post_bot_command({
        "command": "set_filters",
        "strategy_mode": str(strategy_mode),
        "block_weak_signals": bool(block_weak_signals),
        "cooldown_bars": int(cooldown_bars),
        "use_atr_risk": bool(use_atr_risk),
        "atr_sl_mult": float(atr_sl_mult),
        "atr_tp_mult": float(atr_tp_mult),
        "time": str(datetime.now()),
    })
    st.sidebar.success("Filters applied")

# Start Trading Button in Sidebar
col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("🚀 Start All", width='stretch'):
    st.sidebar.success("All pairs started!")
    _post_bot_command({
        "command": "start_all",
        "tp": float(manual_tp),
        "sl": float(manual_sl),
        "leverage": int(leverage),
        "max_trades_per_day": int(max_trades_per_day),
        "daily_tp_target_percent": float(daily_tp_target_percent),
        "time": str(datetime.now())
    })

if col_btn2.button("🛑 Stop All", width='stretch'):
    st.sidebar.warning("All stopped!")
    _post_bot_command({"command": "stop_all", "time": str(datetime.now())})

st.sidebar.subheader("🧹 Maintenance")
reset_confirm = st.sidebar.checkbox("Confirm reset", value=False)
reset_balance = st.sidebar.checkbox("Reset balance to start", value=True)
if st.sidebar.button("🧹 Reset statistics", width='stretch'):
    if not reset_confirm:
        st.sidebar.error("Enable Confirm reset first")
    else:
        _post_bot_command({
            "command": "reset_stats",
            "scope": "all",
            "reset_balance": bool(reset_balance),
            "time": str(datetime.now()),
        })
        st.cache_data.clear()
        st.sidebar.success("Stats reset requested")
        st.rerun()

# ... existing code ...

# Timeframe
timeframe = st.sidebar.selectbox("Timeframe", ["1d", "1h", "15m"], index=1)

# Chart Type
chart_view = st.sidebar.radio("Chart View", ["Standard (Plotly)", "TradingView (Interactive)"], index=0)

# Update interval
st.sidebar.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ============================================
# MAIN TITLE
# ============================================
st.title("🤖 Multi-Asset AI Trading Bot with AI")
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

@st.cache_data(ttl=5)
def load_portfolio():
    try:
        api = _get_bot_json("/portfolio")
        if api is not None:
            return api
        
        p = os.path.join(DATA_DIR, "portfolio_state.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading portfolio: {e}")
    return None

@st.cache_data(ttl=5)
def load_trades():
    try:
        api = _get_bot_json("/trades")
        if api is not None:
            rows = (api.get("rows") or [])
            return pd.DataFrame(rows)

        p = os.path.join(DATA_DIR, "trade_history.csv")
        if os.path.exists(p):
            return pd.read_csv(p)
    except Exception as e:
        st.error(f"Error loading trades: {e}")
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
    start_capital = float(os.getenv("PAPER_START_CAPITAL", 10000))
    balance = float(portfolio.get('balance', start_capital))
    equity = float(portfolio.get('equity', balance))
    pnl_pct = ((equity - start_capital) / start_capital) * 100 if start_capital > 0 else 0.0
    positions = portfolio.get('positions', {})

    col1.metric("💰 Balance", f"${balance:.2f}")
    col2.metric("📈 Equity", f"${equity:.2f}")
    col3.metric("📊 Total PnL", f"{pnl_pct:+.2f}%")
    col4.metric("🎯 Active Positions", len(positions))
    col5.metric("💹 Total Trades", len(trades) if not trades.empty else 0)

    used_margin = portfolio.get('used_margin')
    unrealized_pnl = portfolio.get('unrealized_pnl')
    with st.expander("ℹ️ Balance vs Equity", expanded=False):
        st.write("Balance = free cash after margin is locked for open positions.")
        st.write("Equity = Balance + used margin + unrealized PnL (open positions at last price).")
        if used_margin is not None:
            st.write(f"Used margin: ${float(used_margin):.2f}")
        if unrealized_pnl is not None:
            st.write(f"Unrealized PnL: ${float(unrealized_pnl):+.2f}")

    # Detailed Portfolio View
    st.markdown("### 🎯 Active Positions")
    if not positions:
        st.info("No active positions.")
    else:
        for symbol, pos in positions.items():
            with st.expander(f"📍 {symbol} - {pos['side'].upper()}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.write(f"**Entry Price:** {pos['entry_price']:.5f}")
                c2.write(f"**Size:** {pos['size']:.4f}")
                c3.write(f"**TP / SL:** {pos.get('take_profit', 0):.5f} / {pos.get('stop_loss', 0):.5f}")
                c4.write(f"**Date:** {pos.get('entry_date', 'N/A')}")
else:
    col1.metric("💰 Balance", "$10,000")
    col2.metric("📈 Equity", "$10,000")
    col3.metric("📊 PnL", "0.00%")
    col4.metric("🎯 Positions", "0")
    col5.metric("💹 Trades", "0")
    st.info("Portfolio data not found. Start the bot to generate state.")

st.markdown("---")

# ============================================
# PRICE CHARTS
# ============================================
st.subheader("📈 Price Action & Signals")

if chart_view == "TradingView (Interactive)":
    for symbol in assets:
        st.markdown(f"### 📊 {symbol} Interactive Chart")
        
        # Mapping symbol for TradingView (e.g., EURUSD=X -> FX_IDC:EURUSD)
        tv_symbol = symbol
        if "=" in symbol:
            tv_symbol = symbol.replace("=X", "")
            if symbol.startswith("EUR") or symbol.startswith("GBP") or symbol.startswith("USD"):
                tv_symbol = f"FX_IDC:{tv_symbol}"
        elif "-USD" in symbol:
            tv_symbol = f"BINANCE:{symbol.replace('-', '')}"
            
        tradingview_html = f"""
        <div class=\"tradingview-widget-container\" style=\"height:900px;width:100%;\">
          <div id=\"tradingview_{symbol}\" style=\"height:900px;width:100%;\"></div>
          <script type=\"text/javascript\" src=\"https://s3.tradingview.com/tv.js\"></script>
          <script type=\"text/javascript\">
          new TradingView.widget({{
            \"autosize\": true,
            \"symbol\": \"{tv_symbol}\",
            \"interval\": \"60\",
            \"timezone\": \"Etc/UTC\",
            \"theme\": \"dark\",
            \"style\": \"1\",
            \"locale\": \"en\",
            \"toolbar_bg\": \"#0e1117\",
            \"enable_publishing\": false,
            \"withdateranges\": true,
            \"hide_side_toolbar\": false,
            \"allow_symbol_change\": true,
            \"details\": true,
            \"hotlist\": true,
            \"calendar\": true,
            \"container_id\": \"tradingview_{symbol}\"
          }});
          </script>
        </div>
        """
        components.html(tradingview_html, height=920)
else:
    for symbol in assets:
        df = fetch_asset_data(symbol, period="3mo", interval=timeframe)
        if df is None:
            st.warning(f"No data for {symbol}")
            continue

        df = df.copy()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{symbol} - Price", "RSI (14)")
        )

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=symbol
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', line=dict(color='orange', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50', line=dict(color='blue', width=1)),
            row=1, col=1
        )

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple', width=1)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(height=600, showlegend=True, xaxis_rangeslider_visible=False)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        st.plotly_chart(fig, width='stretch')


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
    st.plotly_chart(fig, width='stretch')

st.markdown("---")

# ============================================
# STRATEGY DIAGNOSTICS
# ============================================
st.subheader("📊 Strategy Diagnostics")

if trades is None or trades.empty or 'pnl' not in trades.columns:
    st.info("No trade statistics yet. Run the bot first.")
else:
    start_capital = float(os.getenv("PAPER_START_CAPITAL", 10000))

    t = trades.copy()
    if 'exit_date' in t.columns:
        t['exit_dt'] = pd.to_datetime(t['exit_date'], errors='coerce')
    elif 'entry_date' in t.columns:
        t['exit_dt'] = pd.to_datetime(t['entry_date'], errors='coerce')
    else:
        t['exit_dt'] = pd.NaT

    t = t.sort_values(by=['exit_dt'], na_position='last')

    def _profit_factor(pnl_series: pd.Series) -> float:
        wins = pnl_series[pnl_series > 0].sum()
        losses = pnl_series[pnl_series < 0].sum()
        if losses == 0:
            return float('inf') if wins > 0 else 0.0
        return float(wins / abs(losses))

    def _max_drawdown(equity: pd.Series) -> float:
        if equity.empty:
            return 0.0
        peak = equity.cummax()
        dd = (equity / peak) - 1.0
        return float(dd.min())

    pnl = pd.to_numeric(t['pnl'], errors='coerce').fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    equity = start_capital + pnl.cumsum()
    max_dd = _max_drawdown(equity)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trades", int(len(pnl)))
    c2.metric("Win Rate", f"{(len(wins)/len(pnl)*100):.1f}%" if len(pnl) else "0%")
    c3.metric("Expectancy", f"${pnl.mean():.2f}")
    c4.metric("Profit Factor", f"{_profit_factor(pnl):.2f}" if len(pnl) else "0")
    c5.metric("Max Drawdown", f"{max_dd*100:.2f}%")

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=t['exit_dt'].fillna(method='ffill'), y=equity, mode='lines', name='Equity'))
    fig_eq.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Time", yaxis_title="Equity")
    st.plotly_chart(fig_eq, width='stretch')

    meta = (portfolio or {}).get("meta") if isinstance(portfolio, dict) else None
    blocked = (meta or {}).get("blocked") if isinstance(meta, dict) else None
    if isinstance(blocked, dict):
        reasons = blocked.get("reasons") if isinstance(blocked.get("reasons"), dict) else {}
        total_blk = int(blocked.get("total") or 0)
        cooldown_blk = int(reasons.get("cooldown") or 0)
        weak_blk = int(reasons.get("weak") or 0)

        st.markdown("### Blocks (filtered entries)")
        b1, b2, b3 = st.columns(3)
        b1.metric("Total blocks", total_blk)
        b2.metric("Cooldown", cooldown_blk)
        b3.metric("Weak blocked", weak_blk)

        by_symbol = blocked.get("by_symbol")
        if isinstance(by_symbol, dict) and by_symbol:
            rows = []
            for sym, v in by_symbol.items():
                if not isinstance(v, dict):
                    continue
                rows.append({
                    "symbol": sym,
                    "total": int(v.get("total") or 0),
                    "cooldown": int(v.get("cooldown") or 0),
                    "weak": int(v.get("weak") or 0),
                })
            if rows:
                dfb = pd.DataFrame(rows).sort_values(by=["total"], ascending=False)
                st.dataframe(dfb, width='stretch')

    if 'strategy_mode' in t.columns or 'signal_strength' in t.columns:
        t['strategy_mode'] = (t.get('strategy_mode') if 'strategy_mode' in t.columns else None)
        if 'strategy_mode' in t.columns:
            t['strategy_mode'] = t['strategy_mode'].fillna('unknown').astype(str)
        t['signal_strength'] = (t.get('signal_strength') if 'signal_strength' in t.columns else None)
        if 'signal_strength' in t.columns:
            t['signal_strength'] = t['signal_strength'].fillna('unknown').astype(str)

        view = t.copy()
        view['pnl_num'] = pd.to_numeric(view['pnl'], errors='coerce').fillna(0.0)

        def _group_stats(g: pd.DataFrame):
            p = g['pnl_num']
            w = p[p > 0]
            return pd.Series({
                'trades': int(len(p)),
                'win_rate_pct': float((len(w)/len(p)*100) if len(p) else 0.0),
                'expectancy': float(p.mean() if len(p) else 0.0),
                'profit_factor': _profit_factor(p),
                'total_pnl': float(p.sum()),
            })

        if 'strategy_mode' in view.columns:
            st.markdown("### Classic vs Reinforse")
            by_mode = view.groupby('strategy_mode', dropna=False).apply(_group_stats).reset_index()
            st.dataframe(by_mode.sort_values('strategy_mode'), width='stretch')

        if 'signal_strength' in view.columns:
            st.markdown("### By Signal Strength")
            by_strength = view.groupby('signal_strength', dropna=False).apply(_group_stats).reset_index()
            st.dataframe(by_strength.sort_values('signal_strength'), width='stretch')

        if 'strategy_mode' in view.columns and 'signal_strength' in view.columns:
            st.markdown("### Mode × Strength")
            by_both = view.groupby(['strategy_mode', 'signal_strength'], dropna=False).apply(_group_stats).reset_index()
            st.dataframe(by_both.sort_values(['strategy_mode', 'signal_strength']), width='stretch')

st.markdown("---")

# ============================================
# TRADE HISTORY
# ============================================
st.subheader("📋 Trade History")

if trades is not None and not trades.empty:
    st.dataframe(trades.tail(50), width='stretch')
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
else:
    st.warning("Model not loaded. Train models first.")

# ============================================
# DEEPSEEK ANALYTICS & CONTROL
# ============================================
st.subheader("🧠 AI Control & Analysis")

# Get API Key
api_key = os.getenv("OPENROUTER_API_KEY")

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    if st.button("🚀 Start AI Trading (All)", width='stretch'):
        st.success("Command sent!")
        _post_bot_command({
            "command": "start_all",
            "tp": float(manual_tp),
            "sl": float(manual_sl),
            "leverage": int(leverage),
            "max_trades_per_day": int(max_trades_per_day),
            "daily_tp_target_percent": float(daily_tp_target_percent),
            "time": str(datetime.now())
        })

with col_ctrl2:
    if st.button("🛑 Stop All", width='stretch'):
        st.warning("Stopping...")
        _post_bot_command({"command": "stop_all", "time": str(datetime.now())})

st.markdown("---")

for symbol in assets:
    st.write(f"### 🔮 AI Analysis for {symbol}")
    
    # Show last known signal if exists
    if portfolio and symbol in portfolio.get('positions', {}):
        pos = portfolio['positions'][symbol]
        st.info(f"📍 Active Position: {pos['side'].upper()} | Entry: {pos['entry_price']:.5f}")

    if st.button(f"New Analysis for {symbol}", key=f"ai_{symbol}"):
        if not api_key:
            st.error("API Key not found!")
            continue
            
        with st.spinner(f"AI analyzing {symbol}..."):
            try:
                import openai
                client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
                
                df_analysis = fetch_asset_data(symbol, period="1mo", interval=timeframe)
                if df_analysis is not None:
                    latest = df_analysis.iloc[-1]
                    
                    prompt = f"""
                    Analyze {symbol}. Price: {latest['close']:.5f}.
                    Strategy: Alligator + Fractals.
                    TP: {manual_tp}%, SL: {manual_sl}%, Leverage: {leverage}x.
                    
                    RESPOND ONLY IN ENGLISH (max 40 words):
                    1. FORECAST: [UP/DOWN/FLAT]
                    2. PROBABILITY: [0-100]%
                    3. RECOMMENDATION: [TRADE/WAIT]
                    4. REASON: [1 short sentence referencing Alligator/Fractals]
                    """
                    
                    response = client.chat.completions.create(
                        model="deepseek/deepseek-chat",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=150
                    )
                    
                    st.success(f"Analysis for {symbol} ready!")
                    st.info(response.choices[0].message.content)
                else:
                    st.error(f"No data for {symbol}")
            except Exception as e:
                st.error(f"AI Error: {e}")
    
    if st.button(f"🚀 Trade {symbol}", key=f"trade_single_{symbol}"):
        _post_bot_command({
            "command": "start_single",
            "symbol": symbol,
            "tp": float(manual_tp),
            "sl": float(manual_sl),
            "leverage": int(leverage),
            "max_trades_per_day": int(max_trades_per_day),
            "daily_tp_target_percent": float(daily_tp_target_percent),
            "time": str(datetime.now())
        })
        st.success(f"Trade command for {symbol} sent!")
    st.markdown("---")
st.caption(f"© EURUSD AI Trading Bot | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")