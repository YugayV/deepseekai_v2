"""
Streamlit Dashboard for EURUSD AI Trading Bot
Simplified (no Alligator/Fractals UI)
"""

import os
import threading
import asyncio
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

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

load_dotenv()

st.set_page_config(page_title="Multi-Asset AI Trader", layout="wide")

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

    with open(CMD_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return {"ok": True}


def _get_bot_json(path: str):
    if not BOT_API_URL:
        return None
    import urllib.request

    with urllib.request.urlopen(BOT_API_URL + path, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


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
    except Exception:
        return None
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
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(ttl=60)
def fetch_asset_data(symbol: str, period: str, interval: str):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    return df


@st.cache_data(ttl=60)
def load_backtest_v3():
    out = Path("output")
    bt_path = out / "backtest_v3.csv"
    m_path = out / "metrics_v3.csv"
    if not bt_path.exists():
        return None, None
    bt = pd.read_csv(bt_path)
    metrics = pd.read_csv(m_path) if m_path.exists() else None
    return bt, metrics


def _to_bool(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _positions_to_frame(positions) -> pd.DataFrame:
    if not isinstance(positions, dict) or not positions:
        return pd.DataFrame()
    rows = []
    for sym, pos in positions.items():
        if not isinstance(pos, dict):
            continue
        row = {"symbol": sym}
        row.update(pos)
        rows.append(row)
    return pd.DataFrame(rows)


st.sidebar.title("⚙️ Config")

assets_universe = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "GC=F",
    "CL=F",
    "SI=F",
    "BTC-USD",
    "ETH-USD",
]

assets = st.sidebar.multiselect("Assets", assets_universe, default=["EURUSD=X"])

timeframe = st.sidebar.selectbox("Timeframe", ["1d", "1h", "15m"], index=1)
chart_symbol = st.sidebar.selectbox("Chart asset", assets if assets else ["EURUSD=X"], index=0)

manual_tp = st.sidebar.slider("Take Profit (%)", 0.5, 10.0, 4.0, 0.5)
manual_sl = st.sidebar.slider("Stop Loss (%)", 0.5, 5.0, 2.0, 0.5)
risk_per_trade_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 3.0, 1.0, 0.1)
paper_fee_bps = st.sidebar.slider("Paper fee (bps)", 0.0, 20.0, 2.0, 0.5)
paper_spread_bps = st.sidebar.slider("Paper spread (bps)", 0.0, 20.0, 1.0, 0.5)
leverage = st.sidebar.selectbox("Leverage", [1, 2, 5, 10, 20, 50, 100], index=2)
max_hold_bars = st.sidebar.slider("Max hold (bars)", 0, 48, 6, 1)

if st.sidebar.button("✅ Apply settings", width="stretch"):
    _post_bot_command(
        {
            "command": "set_filters",
            "risk_per_trade_pct": float(risk_per_trade_pct),
            "paper_fee_bps": float(paper_fee_bps),
            "paper_spread_bps": float(paper_spread_bps),
            "max_hold_bars": int(max_hold_bars),
            "time": str(datetime.now()),
        }
    )
    st.sidebar.success("Settings sent")

c1, c2 = st.sidebar.columns(2)
if c1.button("🚀 Start", width="stretch"):
    _post_bot_command(
        {
            "command": "start_all",
            "symbols": list(assets or []),
            "tp": float(manual_tp),
            "sl": float(manual_sl),
            "leverage": int(leverage),
            "risk_per_trade_pct": float(risk_per_trade_pct),
            "paper_fee_bps": float(paper_fee_bps),
            "paper_spread_bps": float(paper_spread_bps),
            "max_hold_bars": int(max_hold_bars),
            "time": str(datetime.now()),
        }
    )
    st.sidebar.success("Start command sent")

if c2.button("🛑 Stop", width="stretch"):
    _post_bot_command({"command": "stop_all", "time": str(datetime.now())})
    st.sidebar.warning("Stop command sent")

st.sidebar.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if st.sidebar.button("🔄 Refresh", width="stretch"):
    st.cache_data.clear()
    st.rerun()

st.title("Multi-Asset AI Trader")

portfolio = load_portfolio()
trades = load_trades()

st.subheader("📊 Portfolio")

start_capital = float(os.getenv("PAPER_START_CAPITAL", 10000))
balance = float(start_capital)
equity = float(start_capital)
positions = {}

if isinstance(portfolio, dict):
    balance = float(portfolio.get("balance", balance))
    equity = float(portfolio.get("equity", equity))
    positions = portfolio.get("positions", {}) if isinstance(portfolio.get("positions", {}), dict) else {}

pnl_pct = ((equity - start_capital) / start_capital) * 100.0 if start_capital > 0 else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Balance", f"${balance:,.2f}")
k2.metric("Equity", f"${equity:,.2f}")
k3.metric("PnL", f"{pnl_pct:+.2f}%")
k4.metric("Open positions", int(len(positions)))

pos_df = _positions_to_frame(positions)
if not pos_df.empty:
    st.dataframe(pos_df, use_container_width=True)
else:
    st.info("No open positions.")

st.subheader("📋 Trades")
if isinstance(trades, pd.DataFrame) and not trades.empty:
    st.dataframe(trades.tail(100), use_container_width=True)
else:
    st.info("No trades yet.")

st.subheader("📈 Price")
df = fetch_asset_data(chart_symbol, period="3mo", interval=timeframe)
if df is None:
    st.warning(f"No data for {chart_symbol}")
else:
    df = df.copy()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.72, 0.28],
        subplot_titles=(f"{chart_symbol}", "RSI (14)"),
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_20"], name="SMA 20", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], name="SMA 50", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI", line=dict(width=1)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="tomato", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="springgreen", row=2, col=1)
    fig.update_layout(height=820, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("🧪 Research v3 (new_3.py)")
bt, metrics = load_backtest_v3()
if bt is None or bt.empty:
    st.info("Файлы output\\backtest_v3.csv и output\\metrics_v3.csv появятся после запуска new_3.py.")
else:
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        st.dataframe(metrics, use_container_width=True)

    cols = [c for c in ["strategy_eq", "buyhold_eq", "drawdown"] if c in bt.columns]
    if "strategy_eq" in cols and "buyhold_eq" in cols:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(y=bt["strategy_eq"], name="Strategy"))
        fig_eq.add_trace(go.Scatter(y=bt["buyhold_eq"], name="Buy&Hold"))
        fig_eq.update_layout(height=360, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), title="Equity")
        st.plotly_chart(fig_eq, use_container_width=True)

    if "drawdown" in cols:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(y=bt["drawdown"], name="Drawdown"))
        fig_dd.update_layout(height=220, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), title="Drawdown")
        st.plotly_chart(fig_dd, use_container_width=True)

    st.dataframe(bt.tail(200), use_container_width=True)

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

