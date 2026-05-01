"""
Streamlit Dashboard for EURUSD AI Trading Bot
Simplified (no Alligator/Fractals UI)
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

try:
    from yfinance.exceptions import YFRateLimitError
except Exception:
    YFRateLimitError = Exception

# Load environment variables
load_dotenv()

DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
try:
    import psycopg2
except Exception:
    psycopg2 = None

@st.cache_data(ttl=60)
def load_trade_lessons(limit: int = 100) -> pd.DataFrame:
    if not DATABASE_URL or psycopg2 is None:
        return pd.DataFrame()
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, ts, asset, win, pnl, severity, tags, lesson
            FROM trade_lessons
            ORDER BY id DESC
            LIMIT %s
            """,
            (int(limit),),
        )
        rows = cur.fetchall() or []
        cols = ["id", "ts", "asset", "win", "pnl", "severity", "tags", "lesson"]
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return pd.DataFrame()
    finally:
        try:
            conn.close()
        except Exception:
            pass

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


@st.cache_data(ttl=300)
def autotune_atr(symbol: str, period: str, interval: str, fee_bps: float, spread_bps: float, risk_pct: float):
    import bot as bot_module

    try:
        raw = yf.Ticker(symbol).history(period=period, interval=interval)
    except YFRateLimitError:
        st.session_state["yf_rate_limited"] = True
        return None

    if raw is None or raw.empty:
        return None

    raw = raw[['Open', 'High', 'Low', 'Close', 'Volume']]
    raw.columns = ['open', 'high', 'low', 'close', 'volume']

    df = bot_module.calculate_indicators(raw).dropna()
    if df is None or df.empty or 'atr' not in df.columns:
        return None

    loader = bot_module.MLModelLoader()

    def _cost(notional: float) -> float:
        return float(notional) * (float(fee_bps) + float(spread_bps)) / 10_000.0

    atr_sl_grid = [1.2, 1.5, 1.8]
    atr_tp_grid = [2.0, 2.5, 3.0]

    rows = []

    for atr_sl in atr_sl_grid:
        for atr_tp in atr_tp_grid:
            bal = 10000.0
            pos = None
            pnls = []

            for i in range(60, len(df)):
                w = df.iloc[: i + 1]
                px = float(w['close'].iloc[-1])
                atr = float(w['atr'].iloc[-1])

                if pos is not None:
                    side = pos['side']
                    hit = (side == 'long' and (px <= pos['sl'] or px >= pos['tp'])) or (side == 'short' and (px >= pos['sl'] or px <= pos['tp']))
                    if hit:
                        pnl = (px - pos['entry']) * pos['size'] if side == 'long' else (pos['entry'] - px) * pos['size']
                        bal += pnl
                        bal -= _cost(abs(pos['entry'] * pos['size']))
                        bal -= _cost(abs(px * pos['size']))
                        pnls.append(float(pnl))
                        pos = None
                    continue

                ml = loader.predict(w)
                if not ml:
                    continue

                reg = str(ml.get('regime_name') or '').lower()
                try:
                    ema_fast = float(w.get('ema_8', pd.Series([0.0])).iloc[-1])
                    ema_slow = float(w.get('ema_21', pd.Series([0.0])).iloc[-1])
                    macd_h = float(w.get('macd_hist', pd.Series([0.0])).iloc[-1])
                    rsi_v = float(w.get('rsi', pd.Series([50.0])).iloc[-1])
                except Exception:
                    ema_fast, ema_slow, macd_h, rsi_v = 0.0, 0.0, 0.0, 50.0

                trend_up = ema_fast > ema_slow
                trend_dn = ema_fast < ema_slow

                long_ok = trend_up and macd_h > 0 and rsi_v <= 65
                short_ok = trend_dn and macd_h < 0 and rsi_v >= 35

                side = 'long' if (long_ok and reg != 'bearish') else 'short' if (short_ok and reg != 'bullish') else None
                if side is None or atr <= 0:
                    continue

                sl = px - atr * atr_sl if side == 'long' else px + atr * atr_sl
                tp = px + atr * atr_tp if side == 'long' else px - atr * atr_tp

                risk_amt = float(bal) * float(risk_pct)
                unit_risk = abs(px - sl)
                if unit_risk <= 0:
                    continue

                size = risk_amt / unit_risk
                pos = {'side': side, 'entry': px, 'sl': sl, 'tp': tp, 'size': size}

            if not pnls:
                rows.append({'atr_sl': atr_sl, 'atr_tp': atr_tp, 'trades': 0, 'win_rate_pct': 0.0, 'expectancy': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0})
                continue

            pnl = pd.Series(pnls)
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else (float('inf') if wins.sum() > 0 else 0.0)
            rows.append({
                'atr_sl': atr_sl,
                'atr_tp': atr_tp,
                'trades': int(len(pnl)),
                'win_rate_pct': float(len(wins) / len(pnl) * 100.0),
                'expectancy': float(pnl.mean()),
                'profit_factor': float(pf),
                'total_pnl': float(pnl.sum()),
            })

    out = pd.DataFrame(rows).sort_values(by=['profit_factor', 'expectancy'], ascending=False)
    best = out.iloc[0].to_dict() if not out.empty else None
    return {'best': best, 'table': out}


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
    [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X",
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD",
        "AVAX-USD", "LINK-USD", "DOT-USD", "LTC-USD", "TRX-USD",
    ],
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
risk_per_trade_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 3.0, 0.5 if is_real else 1.0, 0.1, key="risk_per_trade_pct")
paper_fee_bps = st.sidebar.slider("Paper fee (bps)", 0.0, 20.0, 2.0, 0.5)
paper_spread_bps = st.sidebar.slider("Paper spread (bps)", 0.0, 20.0, 1.0, 0.5)
leverage = st.sidebar.selectbox("Leverage", [1, 2, 5, 10, 20, 50, 100], index=2)
trade_alloc_pct = st.sidebar.slider("Trade allocation (%)", 1.0, 50.0, 10.0, 0.5, key="trade_alloc_pct")

st.sidebar.subheader("🎯 Partial Take")
use_partial_take = st.sidebar.checkbox("Partial take (TP1 then BE)", value=(not is_real))
partial_take_fraction = st.sidebar.slider("Partial take fraction", 0.1, 0.9, 0.5, 0.05, key="partial_take_fraction")
partial_take_r = st.sidebar.slider("TP1 at R multiple", 0.5, 3.0, 1.0, 0.1, key="partial_take_r")

st.sidebar.subheader("💸 Spread-Aware Allocation")
spread_alloc_reduce_threshold_bps = st.sidebar.slider("Spread reduce threshold (bps)", 0.0, 50.0, 5.0, 0.5, key="spread_alloc_reduce_threshold_bps")
spread_alloc_reduce_pct = st.sidebar.slider("Spread allocation reduce (%)", 0.0, 90.0, 40.0, 1.0, key="spread_alloc_reduce_pct")

st.sidebar.subheader("🕒 Demo Limits")
max_trades_per_day = st.sidebar.slider("Max trades per day", 1, 20, 2, 1)
daily_tp_target_percent = st.sidebar.slider("Daily TP target (%)", 1.0, 50.0, 5.0, 1.0)

st.sidebar.subheader("🧠 Strategy Filters")
strategy_mode = "classic"
st.sidebar.selectbox("Strategy mode", ["Classic"], index=0, disabled=True)

block_weak_signals = st.sidebar.checkbox("Block weak signals", value=False)
cooldown_bars = st.sidebar.slider("Cooldown (bars)", 0, 12, 0, 1)
use_atr_risk = st.sidebar.checkbox("Use ATR-based TP/SL", value=True)
atr_sl_mult = st.sidebar.slider("ATR SL multiple", 0.5, 3.0, 1.5, 0.1)
atr_tp_mult = st.sidebar.slider("ATR TP multiple", 1.0, 6.0, 2.5, 0.1)

st.sidebar.subheader("🎯 Quality (Fewer trades)")
quality_label = st.sidebar.selectbox("Quality mode", ["High (fewer)", "Balanced"], index=0 if is_real else 1)
quality_mode = "high" if quality_label.startswith("High") else "balanced"
min_setup_score = st.sidebar.slider("Min setup score", 0, 5, 4 if is_real else 3, 1)
use_session_filter = st.sidebar.checkbox("Session filter (forex)", value=is_real)
use_orb = st.sidebar.checkbox("Use ORB breakout (24h range)", value=False)
orb_min_range_pct = st.sidebar.slider("ORB min range (%)", 0.0, 1.0, 0.05, 0.01)
orb_max_range_pct = st.sidebar.slider("ORB max range (%)", 0.5, 5.0, 2.50, 0.05)
orb_alloc_reduce_pct = st.sidebar.slider("ORB allocation reduce (%)", 0.0, 90.0, 35.0, 1.0)
min_atr_pct = st.sidebar.slider("Min ATR%", 0.0, 5.0, 0.05, 0.01)
max_atr_pct = st.sidebar.slider("Max ATR%", 0.0, 5.0, 1.50, 0.05)
max_hold_bars = st.sidebar.slider("Max hold (bars)", 0, 48, 6, 1, key="max_hold_bars")

st.sidebar.subheader("🛑 Risk Guard")
risk_guard_enabled = st.sidebar.checkbox("Enable Risk Guard", value=False)
max_open_positions = st.sidebar.slider("Max open positions", 0, 10, 2, 1)
max_daily_drawdown_pct = st.sidebar.slider("Max daily drawdown (%)", 0.0, 20.0, 10.0, 0.1)
max_loss_streak = st.sidebar.slider("Max loss streak", 0, 20, 6, 1)
guard_pause_seconds = st.sidebar.slider("Pause after guard (sec)", 0, 7200, 300, 60)

use_wave_filter = st.sidebar.checkbox("Use wave filter", value=False)
wave_trend_block_pct = st.sidebar.slider("Wave block threshold (%)", 0.0, 2.0, 0.30, 0.01)

use_macro = st.sidebar.checkbox("Use macro/commodities context", value=False)
use_dxy_filter = st.sidebar.checkbox("Use DXY filter", value=False)
dxy_trend_block_pct = st.sidebar.slider("DXY block threshold (%)", 0.0, 2.0, 0.20, 0.01)

use_polymarket = st.sidebar.checkbox("Use Polymarket context", value=False)

st.sidebar.subheader("🌍 Macro Score")
use_macro_score = st.sidebar.checkbox("Use macro score", value=True)
macro_alloc_reduce_pct = st.sidebar.slider("Macro allocation reduce (%)", 0.0, 90.0, 35.0, 1.0)

st.sidebar.subheader("🔎 Auto-tune ATR")
_tune_assets = assets if assets else ["EURUSD=X"]
tune_symbol = st.sidebar.selectbox("Tune symbol", _tune_assets, index=0)
tune_period = st.sidebar.selectbox("Tune period", ["30d", "60d", "180d"], index=1)
tune_interval = "1h"
tune_fee_bps = st.sidebar.slider("Fee (bps)", 0.0, 20.0, 2.0, 0.5)
tune_spread_bps = st.sidebar.slider("Spread (bps)", 0.0, 20.0, 1.0, 0.5)
tune_risk_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 3.0, 1.0, 0.1, key="tune_risk_pct") / 100.0

if st.sidebar.button("🔎 Run auto-tune", width='stretch'):
    res = autotune_atr(tune_symbol, tune_period, tune_interval, float(tune_fee_bps), float(tune_spread_bps), float(tune_risk_pct))
    st.session_state['atr_tune_res'] = res

res = st.session_state.get('atr_tune_res')
if isinstance(res, dict) and isinstance(res.get('best'), dict):
    best = res['best']
    st.sidebar.info(f"Best ATR SL={best.get('atr_sl')} TP={best.get('atr_tp')} | PF={best.get('profit_factor')}")
    if st.sidebar.button("✅ Apply best ATR", width='stretch'):
        _post_bot_command({
            "command": "set_filters",
            "strategy_mode": str(strategy_mode),
            "block_weak_signals": bool(block_weak_signals),
            "cooldown_bars": int(cooldown_bars),
            "use_atr_risk": True,
            "atr_sl_mult": float(best.get('atr_sl')),
            "atr_tp_mult": float(best.get('atr_tp')),
            "quality_mode": str(quality_mode),
            "min_setup_score": int(min_setup_score),
            "use_session_filter": bool(use_session_filter),
            "use_orb": bool(use_orb),
            "orb_min_range_pct": float(orb_min_range_pct),
            "orb_max_range_pct": float(orb_max_range_pct),
            "orb_alloc_reduce_pct": float(orb_alloc_reduce_pct),
            "min_atr_pct": float(min_atr_pct),
            "max_atr_pct": float(max_atr_pct),
            "risk_per_trade_pct": float(risk_per_trade_pct),
            "paper_fee_bps": float(paper_fee_bps),
            "paper_spread_bps": float(paper_spread_bps),
            "max_trade_alloc_pct": float(trade_alloc_pct),
            "max_hold_bars": int(max_hold_bars),
            "use_partial_take": bool(use_partial_take),
            "partial_take_fraction": float(partial_take_fraction),
            "partial_take_r": float(partial_take_r),
            "spread_alloc_reduce_threshold_bps": float(spread_alloc_reduce_threshold_bps),
            "spread_alloc_reduce_pct": float(spread_alloc_reduce_pct),
            "risk_guard_enabled": bool(risk_guard_enabled),
            "max_open_positions": int(max_open_positions),
            "max_daily_drawdown_pct": float(max_daily_drawdown_pct),
            "max_loss_streak": int(max_loss_streak),
            "guard_pause_seconds": int(guard_pause_seconds),
            "use_wave_filter": bool(use_wave_filter),
            "wave_trend_block_pct": float(wave_trend_block_pct),
            "use_macro": bool(use_macro),
            "use_dxy_filter": bool(use_dxy_filter),
            "dxy_trend_block_pct": float(dxy_trend_block_pct),
            "use_polymarket": bool(use_polymarket),
            "use_macro_score": bool(use_macro_score),
            "macro_alloc_reduce_pct": float(macro_alloc_reduce_pct),
            "time": str(datetime.now()),
        })
        st.sidebar.success("Best ATR applied")

    with st.expander("Auto-tune table", expanded=False):
        tbl = res.get('table')
        if tbl is not None:
            st.dataframe(tbl, width='stretch')

if st.sidebar.button("✅ Apply Filters", width='stretch'):
    _post_bot_command({
        "command": "set_filters",
        "strategy_mode": str(strategy_mode),
        "block_weak_signals": bool(block_weak_signals),
        "cooldown_bars": int(cooldown_bars),
        "use_atr_risk": bool(use_atr_risk),
        "atr_sl_mult": float(atr_sl_mult),
        "atr_tp_mult": float(atr_tp_mult),
        "quality_mode": str(quality_mode),
        "min_setup_score": int(min_setup_score),
        "use_session_filter": bool(use_session_filter),
        "use_orb": bool(use_orb),
        "orb_min_range_pct": float(orb_min_range_pct),
        "orb_max_range_pct": float(orb_max_range_pct),
        "orb_alloc_reduce_pct": float(orb_alloc_reduce_pct),
        "min_atr_pct": float(min_atr_pct),
        "max_atr_pct": float(max_atr_pct),
        "risk_per_trade_pct": float(risk_per_trade_pct),
        "paper_fee_bps": float(paper_fee_bps),
        "paper_spread_bps": float(paper_spread_bps),
        "max_trade_alloc_pct": float(trade_alloc_pct),
        "max_hold_bars": int(max_hold_bars),
        "use_partial_take": bool(use_partial_take),
        "partial_take_fraction": float(partial_take_fraction),
        "partial_take_r": float(partial_take_r),
        "spread_alloc_reduce_threshold_bps": float(spread_alloc_reduce_threshold_bps),
        "spread_alloc_reduce_pct": float(spread_alloc_reduce_pct),
        "risk_guard_enabled": bool(risk_guard_enabled),
        "max_open_positions": int(max_open_positions),
        "max_daily_drawdown_pct": float(max_daily_drawdown_pct),
        "max_loss_streak": int(max_loss_streak),
        "guard_pause_seconds": int(guard_pause_seconds),
        "use_wave_filter": bool(use_wave_filter),
        "wave_trend_block_pct": float(wave_trend_block_pct),
        "use_macro": bool(use_macro),
        "use_dxy_filter": bool(use_dxy_filter),
        "dxy_trend_block_pct": float(dxy_trend_block_pct),
        "use_polymarket": bool(use_polymarket),
        "use_macro_score": bool(use_macro_score),
        "macro_alloc_reduce_pct": float(macro_alloc_reduce_pct),
        "time": str(datetime.now()),
    })
    st.sidebar.success("Filters applied")

# Start Trading Button in Sidebar
col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("🚀 Start All", width='stretch'):
    st.sidebar.success("All pairs started!")
    _post_bot_command({
        "command": "start_all",
        "symbols": list(assets or []),
        "tp": float(manual_tp),
        "sl": float(manual_sl),
        "leverage": int(leverage),
        "risk_per_trade_pct": float(risk_per_trade_pct),
        "paper_fee_bps": float(paper_fee_bps),
        "paper_spread_bps": float(paper_spread_bps),
        "max_trade_alloc_pct": float(trade_alloc_pct),
        "max_hold_bars": int(max_hold_bars),
        "use_partial_take": bool(use_partial_take),
        "partial_take_fraction": float(partial_take_fraction),
        "partial_take_r": float(partial_take_r),
        "spread_alloc_reduce_threshold_bps": float(spread_alloc_reduce_threshold_bps),
        "spread_alloc_reduce_pct": float(spread_alloc_reduce_pct),
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
chart_symbol = st.sidebar.selectbox("Chart asset", assets if assets else ["EURUSD=X"], index=0)

# Chart Type
chart_view = st.sidebar.radio(
    "Chart View",
    ["Dual (Plotly + TradingView)", "Standard (Plotly)", "TradingView (Interactive)"],
    index=0,
)

# Update interval
st.sidebar.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ============================================
# MAIN TITLE
# ============================================
st.title("🤖 Multi-Asset Trading Bot")
st.markdown("**Wave Multi-Scale | Real-time Signals**")
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

@st.cache_data(ttl=300)
def fetch_asset_data(symbol, period="3mo", interval="1d"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df is None or df.empty:
            return None

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']

        key = f"{symbol}|{period}|{interval}"
        cache = st.session_state.get("_last_yf_ok")
        if not isinstance(cache, dict):
            cache = {}
        cache[key] = df
        st.session_state["_last_yf_ok"] = cache
        st.session_state["yf_rate_limited"] = False
        return df
    except YFRateLimitError:
        st.session_state["yf_rate_limited"] = True
        key = f"{symbol}|{period}|{interval}"
        cache = st.session_state.get("_last_yf_ok")
        if isinstance(cache, dict) and key in cache:
            return cache[key]
        return None
    except Exception:
        key = f"{symbol}|{period}|{interval}"
        cache = st.session_state.get("_last_yf_ok")
        if isinstance(cache, dict) and key in cache:
            return cache[key]
        return None




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

    realized_pnl = 0.0
    realized_pnl_pct = 0.0
    win_rate = 0.0
    trades_today_pnl = 0.0

    if isinstance(trades, pd.DataFrame) and (not trades.empty) and ('pnl' in trades.columns):
        t = trades.copy()
        t['pnl'] = pd.to_numeric(t['pnl'], errors='coerce').fillna(0.0)
        realized_pnl = float(t['pnl'].sum())
        realized_pnl_pct = (realized_pnl / start_capital) * 100.0 if start_capital > 0 else 0.0

        if 'exit_date' in t.columns:
            dt = pd.to_datetime(t['exit_date'], errors='coerce', utc=False)
            today = datetime.utcnow().date()
            t_today = t[dt.dt.date == today]
            trades_today_pnl = float(pd.to_numeric(t_today['pnl'], errors='coerce').fillna(0.0).sum()) if (t_today is not None and not t_today.empty) else 0.0

        if 'win' in t.columns:
            w = pd.to_numeric(t['win'], errors='coerce')
            if w.notna().any():
                win_rate = float((w.fillna(0).astype(int) == 1).mean() * 100.0)
        else:
            win_rate = float((t['pnl'] > 0).mean() * 100.0) if len(t) > 0 else 0.0

    try:
        unreal = float(portfolio.get('unrealized_pnl', 0.0) or 0.0)
    except Exception:
        unreal = 0.0

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("✅ Realized PnL", f"${realized_pnl:+.2f}")
    p2.metric("🟡 Unrealized PnL", f"${unreal:+.2f}")
    p3.metric("📅 Today PnL", f"${trades_today_pnl:+.2f}")
    p4.metric("🏆 Win rate", f"{win_rate:.1f}%")

    with st.expander("📈 P&L chart", expanded=False):
        if isinstance(trades, pd.DataFrame) and (not trades.empty) and ('pnl' in trades.columns):
            t = trades.copy()
            t['pnl'] = pd.to_numeric(t['pnl'], errors='coerce').fillna(0.0)

            if 'exit_date' in t.columns:
                t['exit_dt'] = pd.to_datetime(t['exit_date'], errors='coerce')
                t = t.sort_values('exit_dt')
            else:
                t = t.reset_index(drop=True)

            t['cum_equity'] = float(start_capital) + t['pnl'].cumsum()

            figp = go.Figure()
            figp.add_trace(go.Scatter(x=list(range(len(t))), y=t['cum_equity'], mode='lines', name='Realized equity'))
            figp.add_hline(y=float(start_capital), line_dash='dash', line_color='gray')
            figp.update_layout(height=260, template='plotly_dark', margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(figp, use_container_width=True)

            cols = [c for c in ['exit_date', 'asset', 'symbol', 'side', 'exit_reason', 'pnl', 'pnl_percent'] if c in t.columns]
            if cols:
                st.dataframe(t[cols].tail(50), use_container_width=True)
        else:
            st.info("P&L появится после первых закрытых сделок (trade_history.csv).")

    used_margin = portfolio.get('used_margin')
    unrealized_pnl = portfolio.get('unrealized_pnl')
    with st.expander("ℹ️ Balance vs Equity", expanded=False):
        st.write("Balance = free cash after margin is locked for open positions.")
        st.write("Equity = Balance + used margin + unrealized PnL (open positions at last price).")
        if used_margin is not None:
            st.write(f"Used margin: ${float(used_margin):.2f}")
        if unrealized_pnl is not None:
            st.write(f"Unrealized PnL: ${float(unrealized_pnl):+.2f}")

    st.markdown("### 💰 P&L")

    realized_pnl = 0.0
    today_pnl = 0.0
    win_rate = 0.0

    t = None
    if isinstance(trades, pd.DataFrame) and (not trades.empty) and ('pnl' in trades.columns):
        t = trades.copy()
        t['pnl'] = pd.to_numeric(t['pnl'], errors='coerce').fillna(0.0)
        realized_pnl = float(t['pnl'].sum())

        if 'exit_date' in t.columns:
            dt = pd.to_datetime(t['exit_date'], errors='coerce')
            today = datetime.utcnow().date()
            m = dt.dt.date == today
            if m is not None:
                today_pnl = float(t.loc[m, 'pnl'].sum()) if m.any() else 0.0

        if 'win' in t.columns:
            w = pd.to_numeric(t['win'], errors='coerce')
            if w.notna().any():
                win_rate = float((w.fillna(0).astype(int) == 1).mean() * 100.0)
            else:
                win_rate = float((t['pnl'] > 0).mean() * 100.0) if len(t) else 0.0
        else:
            win_rate = float((t['pnl'] > 0).mean() * 100.0) if len(t) else 0.0

    try:
        unreal = float(unrealized_pnl or 0.0)
    except Exception:
        unreal = 0.0

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("✅ Realized", f"${realized_pnl:+.2f}")
    p2.metric("🟡 Unrealized", f"${unreal:+.2f}")
    p3.metric("📅 Today", f"${today_pnl:+.2f}")
    p4.metric("🏆 Win rate", f"{win_rate:.1f}%")

    with st.expander("📈 P&L analytics", expanded=False):
        if t is None or t.empty:
            st.info("P&L появится после первых закрытых сделок (trade_history.csv).")
        else:
            if 'exit_date' in t.columns:
                t['exit_dt'] = pd.to_datetime(t['exit_date'], errors='coerce')
                t = t.sort_values('exit_dt')
            else:
                t = t.reset_index(drop=True)

            t['cum_equity'] = float(start_capital) + t['pnl'].cumsum()

            eq = t['cum_equity'].astype(float)
            roll_max = eq.cummax()
            dd = (eq - roll_max)
            dd_pct = (dd / roll_max.replace(0, np.nan)) * 100.0
            max_dd = float(dd.min()) if len(dd) else 0.0
            max_dd_pct = float(dd_pct.min()) if len(dd_pct) else 0.0

            profits = t.loc[t['pnl'] > 0, 'pnl']
            losses = t.loc[t['pnl'] < 0, 'pnl']
            gross_profit = float(profits.sum()) if len(profits) else 0.0
            gross_loss = float(losses.sum()) if len(losses) else 0.0
            profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else (float('inf') if gross_profit > 0 else 0.0)

            avg_win = float(profits.mean()) if len(profits) else 0.0
            avg_loss = float(losses.mean()) if len(losses) else 0.0
            expectancy = float(t['pnl'].mean()) if len(t) else 0.0

            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Profit factor", f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "∞")
            a2.metric("Avg win", f"${avg_win:+.2f}")
            a3.metric("Avg loss", f"${avg_loss:+.2f}")
            a4.metric("Expectancy", f"${expectancy:+.2f}/trade")

            d1, d2 = st.columns(2)
            d1.metric("Max drawdown", f"${max_dd:.2f}")
            d2.metric("Max drawdown (%)", f"{max_dd_pct:.2f}%")

            figp = go.Figure()
            figp.add_trace(go.Scatter(x=list(range(len(t))), y=t['cum_equity'], mode='lines', name='Realized equity'))
            figp.add_hline(y=float(start_capital), line_dash='dash', line_color='gray')
            figp.update_layout(height=260, template='plotly_dark', margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(figp, use_container_width=True)

            figh = go.Figure()
            figh.add_trace(go.Histogram(x=t['pnl'], nbinsx=30, name='PnL distribution'))
            figh.update_layout(height=220, template='plotly_dark', margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(figh, use_container_width=True)

            if 'exit_reason' in t.columns:
                rc = t['exit_reason'].astype(str).value_counts().head(10).reset_index()
                rc.columns = ['reason', 'count']
                st.markdown("### Exit reasons")
                st.dataframe(rc, use_container_width=True)

            cols = [c for c in ['exit_date', 'asset', 'symbol', 'side', 'exit_reason', 'pnl', 'pnl_percent', 'is_partial'] if c in t.columns]
            if cols:
                st.dataframe(t[cols].tail(80), use_container_width=True)

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

    st.markdown("### 🕒 Session / Hour analysis (UTC)")
    if trades is None or trades.empty or 'pnl' not in trades.columns:
        st.info("No closed trades yet to analyze hours.")
    else:
        t = trades.copy()
        t['pnl'] = pd.to_numeric(t.get('pnl'), errors='coerce').fillna(0.0)

        dt_col = 'entry_date' if 'entry_date' in t.columns else ('exit_date' if 'exit_date' in t.columns else None)
        if dt_col is None:
            st.info("Trades are missing entry_date/exit_date.")
        else:
            dt = pd.to_datetime(t[dt_col], errors='coerce')
            t['hour_utc'] = dt.dt.hour
            t = t.dropna(subset=['hour_utc'])
            t['hour_utc'] = t['hour_utc'].astype(int)

            min_samples = st.slider("Min trades per hour", 1, 50, 5, 1)

            g = t.groupby('hour_utc').agg(
                trades=('pnl', 'size'),
                pnl_sum=('pnl', 'sum'),
                pnl_avg=('pnl', 'mean'),
            ).reset_index()

            wins = t.assign(win=(t['pnl'] > 0).astype(int)).groupby('hour_utc')['win'].sum().reset_index()
            g = g.merge(wins, on='hour_utc', how='left')
            g['win_rate'] = (g['win'] / g['trades'].replace(0, np.nan) * 100.0).fillna(0.0)

            losing_hours = g[(g['trades'] >= int(min_samples)) & (g['pnl_avg'] < 0)]['hour_utc'].tolist()
            excluded_hours = st.multiselect("Exclude hours (UTC)", list(range(24)), default=losing_hours)

            if st.button("🚫 Apply excluded hours", width='stretch'):
                _post_bot_command({
                    "command": "set_filters",
                    "excluded_hours": [int(h) for h in excluded_hours],
                    "time": str(datetime.now()),
                })
                st.success(f"Excluded hours applied: {sorted([int(h) for h in excluded_hours])}")

            g = g.sort_values('hour_utc')
            colors = ['tomato' if float(v) < 0 else 'springgreen' for v in g['pnl_sum'].tolist()]

            fig_h = go.Figure()
            fig_h.add_trace(go.Bar(x=g['hour_utc'], y=g['pnl_sum'], name='PnL sum', marker_color=colors))
            fig_h.update_layout(height=240, template='plotly_dark', margin=dict(l=10, r=10, t=10, b=10), xaxis_title='UTC hour', yaxis_title='PnL')
            st.plotly_chart(fig_h, use_container_width=True)

            g_show = g[['hour_utc', 'trades', 'win_rate', 'pnl_sum', 'pnl_avg']].copy()
            st.dataframe(g_show, use_container_width=True)

else:
    col1.metric("💰 Balance", "$10,000")
    col2.metric("📈 Equity", "$10,000")
    col3.metric("📊 PnL", "0.00%")
    col4.metric("🎯 Positions", "0")
    col5.metric("💹 Trades", "0")
    st.info("Portfolio data not found. Start the bot to generate state.")

st.markdown("---")

st.subheader("🌍 Macro (USD strength)")
if isinstance(portfolio, dict):
    meta = portfolio.get('meta') if isinstance(portfolio.get('meta'), dict) else {}
    last_decisions = meta.get('last_decisions') if isinstance(meta.get('last_decisions'), list) else []

    mrows = []
    for d in last_decisions[-80:]:
        if not isinstance(d, dict):
            continue
        sym = str(d.get('symbol') or d.get('asset') or '').strip()
        if not sym:
            continue
        try:
            usd24 = float(d.get('usd_strength_24h') or 0.0)
        except Exception:
            usd24 = 0.0
        try:
            align = float(d.get('macro_align') or 0.0)
        except Exception:
            align = 0.0
        mrows.append({'symbol': sym, 'usd_strength_24h': usd24, 'macro_align': align})

    if mrows:
        mdf = pd.DataFrame(mrows)
        c1, c2, c3 = st.columns(3)
        c1.metric("USD strength avg (24h)", f"{mdf['usd_strength_24h'].mean():+.2f}%")
        c2.metric("Macro align avg", f"{mdf['macro_align'].mean():+.2f}")
        c3.metric("Samples", int(len(mdf)))

        figm = go.Figure()
        figm.add_trace(go.Scatter(y=mdf['usd_strength_24h'], mode='lines', name='USD strength 24h (%)'))
        figm.add_hline(y=0, line_dash='dash', line_color='gray')
        figm.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(figm, use_container_width=True)

        st.dataframe(mdf.tail(20), width='stretch')
    else:
        st.info("Macro data will appear after new decisions. Run the bot for a few cycles.")
else:
    st.info("Start the bot to populate macro data.")

st.markdown("---")

st.subheader("🧠 Lessons")
lessons = load_trade_lessons(limit=100)
if lessons is None or lessons.empty:
    st.info("No lessons yet (DATABASE_URL missing or no closed trades).")
else:
    st.dataframe(lessons[["ts", "asset", "win", "pnl", "severity", "tags", "lesson"]].head(50), width='stretch')

    tags = []
    for v in lessons["tags"].dropna().astype(str).tolist():
        for t in [x.strip() for x in v.split(",") if x.strip()]:
            tags.append(t)
    if tags:
        top = pd.Series(tags).value_counts().head(15).reset_index()
        top.columns = ["tag", "count"]
        st.markdown("### Top tags")
        st.dataframe(top, width='stretch')

st.markdown("---")

# ============================================
# PRICE CHARTS
# ============================================
st.subheader("📈 Price Action & Signals")

if st.session_state.get("yf_rate_limited"):
    st.warning("Yahoo Finance rate limited. Showing last cached data where possible. Try again later.")

def _tv_symbol(sym: str) -> str:
    tv = sym
    if "=" in sym:
        tv = sym.replace("=X", "")
        if sym.startswith("EUR") or sym.startswith("GBP") or sym.startswith("USD") or sym.startswith("JPY"):
            tv = f"FX_IDC:{tv}"
    elif "-USD" in sym:
        tv = f"BINANCE:{sym.replace('-', '')}"
    return tv

_tv_interval = "60" if timeframe == "1h" else "15" if timeframe == "15m" else "D"

if chart_view == "Dual (Plotly + TradingView)":
    df = fetch_asset_data(chart_symbol, period="3mo", interval=timeframe)
    if df is None:
        st.warning(f"No data for {chart_symbol}")
    else:
        df = df.copy()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['or_high_24'] = df['high'].rolling(24).max().shift(1)
        df['or_low_24'] = df['low'].rolling(24).min().shift(1)
        df['or_range_pct'] = ((df['or_high_24'] - df['or_low_24']) / df['close']) * 100.0
        df['breakout_up'] = (df['close'] > df['or_high_24']).astype(int)
        df['breakout_down'] = (df['close'] < df['or_low_24']).astype(int)

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.72, 0.28],
            subplot_titles=(f"{chart_symbol} - Price", "RSI (14)")
        )

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=chart_symbol
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', line=dict(color='orange', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50', line=dict(color='deepskyblue', width=1)),
            row=1, col=1
        )

        if use_orb:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['or_high_24'], name='ORB 24h High', line=dict(color='springgreen', width=1, dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['or_low_24'], name='ORB 24h Low', line=dict(color='tomato', width=1, dash='dash')),
                row=1, col=1
            )

            bu = df[df['breakout_up'] == 1]
            bd = df[df['breakout_down'] == 1]
            if not bu.empty:
                fig.add_trace(
                    go.Scatter(x=bu.index, y=bu['close'], mode='markers', name='ORB Breakout Up', marker=dict(color='springgreen', size=7, symbol='triangle-up')),
                    row=1, col=1
                )
            if not bd.empty:
                fig.add_trace(
                    go.Scatter(x=bd.index, y=bd['close'], mode='markers', name='ORB Breakout Down', marker=dict(color='tomato', size=7, symbol='triangle-down')),
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
            go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='violet', width=1)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="tomato", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="springgreen", row=2, col=1)

        fig.update_layout(
            height=820,
            template="plotly_dark",
            showlegend=True,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(title_text="Date", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        if use_orb:
            f_or = go.Figure()
            f_or.add_trace(go.Scatter(x=df.index, y=df['or_range_pct'], mode='lines', name='ORB range (%)'))
            f_or.add_hline(y=float(orb_min_range_pct), line_dash='dash', line_color='gray')
            f_or.add_hline(y=float(orb_max_range_pct), line_dash='dash', line_color='gray')
            f_or.update_layout(height=180, template='plotly_dark', margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(f_or, use_container_width=True)

    st.markdown("---")

    tv = _tv_symbol(chart_symbol)
    tradingview_html = f"""
    <div class=\"tradingview-widget-container\" style=\"height:860px;width:100%;\">
      <div id=\"tradingview_dual\" style=\"height:860px;width:100%;\"></div>
      <script type=\"text/javascript\" src=\"https://s3.tradingview.com/tv.js\"></script>
      <script type=\"text/javascript\">
      new TradingView.widget({{
        \"autosize\": true,
        \"symbol\": \"{tv}\",
        \"interval\": \"{_tv_interval}\",
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
        \"container_id\": \"tradingview_dual\"
      }});
      </script>
    </div>
    """
    components.html(tradingview_html, height=900)

elif chart_view == "TradingView (Interactive)":
    for symbol in assets:
        st.markdown(f"### 📊 {symbol} Interactive Chart")

        tv = _tv_symbol(symbol)
        tradingview_html = f"""
        <div class=\"tradingview-widget-container\" style=\"height:900px;width:100%;\">
          <div id=\"tradingview_{symbol}\" style=\"height:900px;width:100%;\"></div>
          <script type=\"text/javascript\" src=\"https://s3.tradingview.com/tv.js\"></script>
          <script type=\"text/javascript\">
          new TradingView.widget({{
            \"autosize\": true,
            \"symbol\": \"{tv}\",
            \"interval\": \"{_tv_interval}\",
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
            row_heights=[0.72, 0.28],
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
            go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50', line=dict(color='deepskyblue', width=1)),
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
            go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='violet', width=1)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="tomato", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="springgreen", row=2, col=1)

        fig.update_layout(
            height=780,
            template="plotly_dark",
            showlegend=True,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(title_text="Date", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)


st.markdown("---")

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
    st.plotly_chart(fig_eq, use_container_width=True)

    meta = (portfolio or {}).get("meta") if isinstance(portfolio, dict) else None
    blocked = (meta or {}).get("blocked") if isinstance(meta, dict) else None
    if isinstance(blocked, dict):
        reasons = blocked.get("reasons") if isinstance(blocked.get("reasons"), dict) else {}
        total_blk = int(blocked.get("total") or 0)
        cooldown_blk = int(reasons.get("cooldown") or 0)
        weak_blk = int(reasons.get("weak") or 0)

        st.markdown("### Blocks (filtered entries)")
        b1, b2 = st.columns(2)
        b1.metric("Total blocks", total_blk)
        b2.metric("Symbols blocked", int(len((blocked or {}).get('by_symbol') or {})))

        reasons = blocked.get('reasons') if isinstance(blocked, dict) else None
        if isinstance(reasons, dict) and reasons:
            rrows = []
            for k, v in reasons.items():
                try:
                    rrows.append({'reason': str(k), 'count': int(v)})
                except Exception:
                    continue
            if rrows:
                dfr = pd.DataFrame(rrows).sort_values(by=['count'], ascending=False)
                st.dataframe(dfr, width='stretch')

        by_symbol = blocked.get("by_symbol")
        if isinstance(by_symbol, dict) and by_symbol:
            keys = set()
            for _, v in by_symbol.items():
                if isinstance(v, dict):
                    keys |= set(v.keys())
            keys.discard('symbol')
            cols = ['symbol'] + [k for k in sorted(keys) if k != 'symbol']

            rows = []
            for sym, v in by_symbol.items():
                if not isinstance(v, dict):
                    continue
                row = {'symbol': sym}
                for k in keys:
                    if k == 'symbol':
                        continue
                    try:
                        row[k] = int(v.get(k) or 0)
                    except Exception:
                        row[k] = 0
                rows.append(row)

            if rows:
                dfb = pd.DataFrame(rows)
                if 'total' in dfb.columns:
                    dfb = dfb.sort_values(by=['total'], ascending=False)
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