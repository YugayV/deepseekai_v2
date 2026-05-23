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
OPENROUTER_API_KEY = (os.getenv("OPENROUTER_API_KEY") or "").strip()
OPENROUTER_SITE_URL = (os.getenv("OPENROUTER_SITE_URL") or "").strip()
OPENROUTER_APP_NAME = (os.getenv("OPENROUTER_APP_NAME") or "").strip()
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

st.set_page_config(page_title="Multi-Asset AI Trader", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 1.6rem; max-width: 1280px; }
      [data-testid="stSidebar"] { width: 360px; }
      header { visibility: hidden; height: 0px; }
      footer { visibility: hidden; height: 0px; }
      #MainMenu { visibility: hidden; }
      div[data-testid="stMetric"] { background: rgba(255,255,255,0.03); padding: 12px 12px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.06); }
      div[data-testid="stMetricLabel"] > div { font-size: 0.95rem; opacity: 0.85; }
      div[data-testid="stMetricValue"] > div { font-size: 1.45rem; }
      .stTabs [data-baseweb="tab-list"] { gap: 6px; }
      .stTabs [data-baseweb="tab"] { padding-top: 10px; padding-bottom: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

def _get_openrouter_api_key() -> str:
    v = st.session_state.get("openrouter_api_key_override")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return OPENROUTER_API_KEY

def _openrouter_client(api_key: str):
    import openai
    headers = {}
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_APP_NAME:
        headers["X-Title"] = OPENROUTER_APP_NAME
    return openai.OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers=(headers or None),
    )

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

with st.sidebar.expander("🔑 OpenRouter", expanded=False):
    st.text_input("API key (session)", type="password", key="openrouter_api_key_override")
    k = _get_openrouter_api_key()
    if not k:
        st.error("OPENROUTER_API_KEY is missing.")
    else:
        st.success("API key is set.")
    if st.button("Clear session key", width='stretch'):
        st.session_state["openrouter_api_key_override"] = ""
        st.rerun()

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
strategy_mode_label = st.sidebar.selectbox("Strategy mode", ["Classic", "NY SMC (Pine)", "COMBO (Pine gate + Classic)"], index=2)
if strategy_mode_label == "Classic":
    strategy_mode = "classic"
elif strategy_mode_label == "NY SMC (Pine)":
    strategy_mode = "ny_smc"
else:
    strategy_mode = "combo"

if st.sidebar.button("✅ Применить улучшения (качество)", width='stretch'):
    _post_bot_command({
        "command": "set_filters",
        "strategy_mode": "combo",
        "use_ml": True,
        "pine_trend_filter": True,
        "pine_allow_countertrend": False,
        "pine_ml_confirm": True,
        "pine_ml_min_conf": 0.58,
        "pine_ny_session_only": True,
        "pine_ny_first_2h_only": True,
        "pine_require_london_sweep": True,
        "pine_s1_enabled": True,
        "pine_s2_enabled": False,
        "pine_pivot_len": 5,
        "pine_max_bars_after_sweep": 500,
        "pine_max_bars_after_imbalance": 500,
        "use_atr_risk": True,
        "atr_sl_mult": 1.5,
        "atr_tp_mult": 2.5,
        "block_weak_signals": True,
        "cooldown_bars": 2,
        "quality_mode": "high",
        "min_setup_score": 3,
        "time": str(datetime.now()),
    })
    st.sidebar.success("Улучшения применены")

with st.sidebar.expander("🧠 Strategy & Filters (Advanced)", expanded=False):
    use_ml = st.checkbox("Use ML regime filter (unstable)", value=False)
    block_weak_signals = st.checkbox("Block weak signals", value=False)
    cooldown_bars = st.slider("Cooldown (bars)", 0, 12, 0, 1)
    use_atr_risk = st.checkbox("Use ATR-based TP/SL", value=True)
    atr_sl_mult = st.slider("ATR SL multiple", 0.5, 3.0, 1.5, 0.1)
    atr_tp_mult = st.slider("ATR TP multiple", 1.0, 6.0, 2.5, 0.1)

    st.subheader("🗽 Pine (NY_AllTimezones)")
    pine_ny_session_only = st.checkbox("NY session only (UTC 12:00-21:00)", value=True)
    pine_ny_first_2h_only = st.checkbox("NY first 2 hours only (UTC 12:00-14:00)", value=True)
    pine_require_london_sweep = st.checkbox("Require London sweep first (UTC 07:00-16:00)", value=True)
    pine_s1_enabled = st.checkbox("Enable S1 (H1 sweep + M15 imbalance/inversion)", value=True)
    pine_s2_enabled = st.checkbox("Enable S2 (M15 sweep + M5 imbalance/inversion)", value=False)
    pine_pivot_len = st.slider("Pine pivot len", 1, 50, 5, 1)
    pine_max_bars_after_sweep = st.slider("Max bars after sweep", 1, 5000, 500, 10)
    pine_max_bars_after_imbalance = st.slider("Max bars after imbalance", 1, 5000, 500, 10)
    pine_trend_filter = st.checkbox("Trend filter (H1 EMA21/50)", value=True)
    pine_allow_countertrend = st.checkbox("Allow countertrend", value=False)
    pine_ml_confirm = st.checkbox("ML confirm (if ML enabled)", value=True)
    pine_ml_min_conf = st.slider("ML min confidence", 0.0, 1.0, 0.58, 0.01)

    st.subheader("🎯 Quality (Fewer trades)")
    quality_label = st.selectbox("Quality mode", ["High (fewer)", "Balanced"], index=0 if is_real else 1)
    quality_mode = "high" if quality_label.startswith("High") else "balanced"
    min_setup_score = st.slider("Min setup score", 0, 5, 4 if is_real else 3, 1)
    use_session_filter = st.checkbox("Session filter (forex)", value=is_real)
    use_orb = st.checkbox("Use ORB breakout (24h range)", value=False)
    orb_min_range_pct = st.slider("ORB min range (%)", 0.0, 1.0, 0.05, 0.01)
    orb_max_range_pct = st.slider("ORB max range (%)", 0.5, 5.0, 2.50, 0.05)
    orb_alloc_reduce_pct = st.slider("ORB allocation reduce (%)", 0.0, 90.0, 35.0, 1.0)
    min_atr_pct = st.slider("Min ATR%", 0.0, 5.0, 0.05, 0.01)
    max_atr_pct = st.slider("Max ATR%", 0.0, 5.0, 1.50, 0.05)
    max_hold_bars = st.slider("Max hold (bars)", 0, 48, 6, 1, key="max_hold_bars")

    st.subheader("🛑 Risk Guard")
    risk_guard_enabled = st.checkbox("Enable Risk Guard", value=False)
    max_open_positions = st.slider("Max open positions", 0, 10, 2, 1)
    max_daily_drawdown_pct = st.slider("Max daily drawdown (%)", 0.0, 20.0, 10.0, 0.1)
    max_loss_streak = st.slider("Max loss streak", 0, 20, 6, 1)
    guard_pause_seconds = st.slider("Pause after guard (sec)", 0, 7200, 300, 60)

    use_wave_filter = st.checkbox("Use wave filter", value=False)
    wave_trend_block_pct = st.slider("Wave block threshold (%)", 0.0, 2.0, 0.30, 0.01)

    use_macro = st.checkbox("Use macro/commodities context", value=False)
    use_dxy_filter = st.checkbox("Use DXY filter", value=False)
    dxy_trend_block_pct = st.slider("DXY block threshold (%)", 0.0, 2.0, 0.20, 0.01)

    use_polymarket = st.checkbox("Use Polymarket context", value=False)

    st.subheader("🌍 Macro Score")
    use_macro_score = st.checkbox("Use macro score", value=True)
    macro_alloc_reduce_pct = st.slider("Macro allocation reduce (%)", 0.0, 90.0, 35.0, 1.0)

    st.subheader("🔎 Auto-tune ATR")
    _tune_assets = assets if assets else ["EURUSD=X"]
    tune_symbol = st.selectbox("Tune symbol", _tune_assets, index=0)
    tune_period = st.selectbox("Tune period", ["30d", "60d", "180d"], index=1)
    tune_interval = "1h"
    tune_fee_bps = st.slider("Fee (bps)", 0.0, 20.0, 2.0, 0.5)
    tune_spread_bps = st.slider("Spread (bps)", 0.0, 20.0, 1.0, 0.5)
    tune_risk_pct = st.slider("Risk per trade (%)", 0.1, 3.0, 1.0, 0.1, key="tune_risk_pct") / 100.0

    if st.button("🔎 Run auto-tune", width='stretch'):
        res = autotune_atr(tune_symbol, tune_period, tune_interval, float(tune_fee_bps), float(tune_spread_bps), float(tune_risk_pct))
        st.session_state['atr_tune_res'] = res

    res = st.session_state.get('atr_tune_res')
    if isinstance(res, dict) and isinstance(res.get('best'), dict):
        best = res['best']
        st.info(f"Best ATR SL={best.get('atr_sl')} TP={best.get('atr_tp')} | PF={best.get('profit_factor')}")
        if st.button("✅ Apply best ATR", width='stretch'):
            _post_bot_command({
                "command": "set_filters",
                "strategy_mode": str(strategy_mode),
                "use_ml": bool(use_ml),
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
                "pine_ny_session_only": bool(pine_ny_session_only),
                "pine_ny_first_2h_only": bool(pine_ny_first_2h_only),
                "pine_require_london_sweep": bool(pine_require_london_sweep),
                "pine_s1_enabled": bool(pine_s1_enabled),
                "pine_s2_enabled": bool(pine_s2_enabled),
                "pine_pivot_len": int(pine_pivot_len),
                "pine_max_bars_after_sweep": int(pine_max_bars_after_sweep),
                "pine_max_bars_after_imbalance": int(pine_max_bars_after_imbalance),
                "pine_trend_filter": bool(pine_trend_filter),
                "pine_allow_countertrend": bool(pine_allow_countertrend),
                "pine_ml_confirm": bool(pine_ml_confirm),
                "pine_ml_min_conf": float(pine_ml_min_conf),
                "time": str(datetime.now()),
            })
            st.success("Best ATR applied")

        with st.expander("Auto-tune table", expanded=False):
            tbl = res.get('table')
            if tbl is not None:
                st.dataframe(tbl, width='stretch')

    if st.button("⚡ More trades preset", width='stretch'):
        _post_bot_command({
            "command": "set_filters",
            "strategy_mode": str(strategy_mode),
            "use_ml": False,
            "block_weak_signals": False,
            "cooldown_bars": 0,
            "use_atr_risk": bool(use_atr_risk),
            "atr_sl_mult": float(atr_sl_mult),
            "atr_tp_mult": float(atr_tp_mult),
            "quality_mode": "balanced",
            "min_setup_score": 1,
            "use_session_filter": False,
            "use_orb": False,
            "orb_min_range_pct": float(orb_min_range_pct),
            "orb_max_range_pct": float(orb_max_range_pct),
            "orb_alloc_reduce_pct": float(orb_alloc_reduce_pct),
            "min_atr_pct": 0.0,
            "max_atr_pct": 5.0,
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
            "risk_guard_enabled": False,
            "max_open_positions": int(max_open_positions),
            "max_daily_drawdown_pct": float(max_daily_drawdown_pct),
            "max_loss_streak": int(max_loss_streak),
            "guard_pause_seconds": int(guard_pause_seconds),
            "use_wave_filter": False,
            "wave_trend_block_pct": float(wave_trend_block_pct),
            "use_macro": bool(use_macro),
            "use_dxy_filter": False,
            "dxy_trend_block_pct": float(dxy_trend_block_pct),
            "use_polymarket": bool(use_polymarket),
            "use_macro_score": bool(use_macro_score),
            "macro_alloc_reduce_pct": float(macro_alloc_reduce_pct),
            "pine_ny_session_only": bool(pine_ny_session_only),
            "pine_ny_first_2h_only": bool(pine_ny_first_2h_only),
            "pine_require_london_sweep": bool(pine_require_london_sweep),
            "pine_s1_enabled": bool(pine_s1_enabled),
            "pine_s2_enabled": bool(pine_s2_enabled),
            "pine_pivot_len": int(pine_pivot_len),
            "pine_max_bars_after_sweep": int(pine_max_bars_after_sweep),
            "pine_max_bars_after_imbalance": int(pine_max_bars_after_imbalance),
            "pine_trend_filter": bool(pine_trend_filter),
            "pine_allow_countertrend": bool(pine_allow_countertrend),
            "pine_ml_confirm": bool(pine_ml_confirm),
            "pine_ml_min_conf": float(pine_ml_min_conf),
            "time": str(datetime.now()),
        })
        st.success("More-trades preset applied")

    if st.button("✅ Apply Filters", width='stretch'):
        _post_bot_command({
            "command": "set_filters",
            "strategy_mode": str(strategy_mode),
            "use_ml": bool(use_ml),
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
            "pine_ny_session_only": bool(pine_ny_session_only),
            "pine_ny_first_2h_only": bool(pine_ny_first_2h_only),
            "pine_require_london_sweep": bool(pine_require_london_sweep),
            "pine_s1_enabled": bool(pine_s1_enabled),
            "pine_s2_enabled": bool(pine_s2_enabled),
            "pine_pivot_len": int(pine_pivot_len),
            "pine_max_bars_after_sweep": int(pine_max_bars_after_sweep),
            "pine_max_bars_after_imbalance": int(pine_max_bars_after_imbalance),
            "pine_trend_filter": bool(pine_trend_filter),
            "pine_allow_countertrend": bool(pine_allow_countertrend),
            "pine_ml_confirm": bool(pine_ml_confirm),
            "pine_ml_min_conf": float(pine_ml_min_conf),
            "time": str(datetime.now()),
        })
        st.success("Filters applied")

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
    ["NY SMC (Indicator)", "TradingView (Interactive)"],
    index=0,
)

with st.sidebar.expander("🎛 Индикатор NY SMC", expanded=False):
    show_sessions_asia = st.checkbox("Сессия Азия (UTC 00:00-09:00)", value=True)
    show_sessions_london = st.checkbox("Сессия Лондон (UTC 07:00-16:00)", value=True)
    show_sessions_ny = st.checkbox("Сессия Нью-Йорк (UTC 12:00-21:00)", value=True)
    show_imbalance_zones = st.checkbox("Имбаланс-зоны", value=True)
    show_zone_levels = st.checkbox("Уровни зоны (верх/низ)", value=True)
    show_zone_labels = st.checkbox("Подписи уровней", value=False)
    show_sweeps = st.checkbox("Свипы (liquidity sweeps)", value=True)
    show_sweep_level_labels = st.checkbox("Подписи уровней свипа", value=False)
    show_s1 = st.checkbox("Сигналы S1", value=True)
    show_s2 = st.checkbox("Сигналы S2", value=True)
    session_days = st.slider("Сколько дней подсветки сессий", 1, 30, 7, 1)
    zones_per_side = st.slider("Сколько зон на сторону", 1, 10, 3, 1)

# Update interval
st.sidebar.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ============================================
# MAIN TITLE
# ============================================
st.title("Multi-Asset AI Trader")
st.markdown("Wave multi-scale • Real-time signals")
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


def _to_utc_index_df(df: pd.DataFrame) -> pd.DataFrame:
    try:
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        out = df.copy()
        out.index = idx
        return out
    except Exception:
        return df


def _pivot_points(series: pd.Series, length: int, kind: str) -> pd.Series:
    n = int(length)
    if n <= 0:
        return pd.Series(index=series.index, dtype=float)
    win = int(2 * n + 1)
    if win <= 1:
        return pd.Series(index=series.index, dtype=float)
    if kind == "high":
        r = series.rolling(win, center=True).max()
    else:
        r = series.rolling(win, center=True).min()
    return series.where(series == r)


def _engulf_events(df: pd.DataFrame):
    o = df["open"]
    c = df["close"]
    bull = (c > o) & (c.shift(1) < o.shift(1)) & (c >= o.shift(1)) & (o <= c.shift(1))
    bear = (c < o) & (c.shift(1) > o.shift(1)) & (c <= o.shift(1)) & (o >= c.shift(1))
    bull_ev = bull & (~bull.shift(1).fillna(False))
    bear_ev = bear & (~bear.shift(1).fillna(False))
    return bull_ev, bear_ev


def _imbalance_events(df: pd.DataFrame):
    low = df["low"]
    high = df["high"]
    bull = low > high.shift(2)
    bear = high < low.shift(2)
    bull_idx = bull[bull].index
    bear_idx = bear[bear].index
    bull_rows = []
    bear_rows = []
    for i in bull_idx:
        top = float(low.loc[i])
        bot = float(high.shift(2).loc[i])
        bull_rows.append({"ts": i, "upper": max(top, bot), "lower": min(top, bot), "side": "bull"})
    for i in bear_idx:
        top = float(low.shift(2).loc[i])
        bot = float(high.loc[i])
        bear_rows.append({"ts": i, "upper": max(top, bot), "lower": min(top, bot), "side": "bear"})
    bull_df = pd.DataFrame(bull_rows)
    bear_df = pd.DataFrame(bear_rows)
    return bull_df, bear_df


def _minutes_utc(ts) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC")
    return int(t.hour) * 60 + int(t.minute)


def _in_range_utc(ts, start_min: int, end_min: int) -> bool:
    try:
        m = _minutes_utc(ts)
        return int(start_min) <= m <= int(end_min)
    except Exception:
        return False


def _session_vrects(fig, idx, start_min: int, end_min: int, fillcolor: str, opacity: float, limit_days: int = 7):
    try:
        t0 = pd.Timestamp(idx.min())
        t1 = pd.Timestamp(idx.max())
        dates = pd.date_range(t0.normalize(), t1.normalize(), freq="D")
        if len(dates) > int(limit_days):
            dates = dates[-int(limit_days):]
        for d in dates:
            x0 = d + pd.Timedelta(minutes=int(start_min))
            x1 = d + pd.Timedelta(minutes=int(end_min))
            if x1 < t0 or x0 > t1:
                continue
            fig.add_vrect(x0=x0, x1=x1, fillcolor=fillcolor, opacity=float(opacity), line_width=0)
    except Exception:
        return


def _sweep_events(df: pd.DataFrame, pivot_len: int):
    ph = _pivot_points(df["high"], pivot_len, "high")
    pl = _pivot_points(df["low"], pivot_len, "low")
    last_ph = ph.ffill()
    last_pl = pl.ffill()
    sweep_up = (df["high"] > last_ph) & (df["close"] < last_ph)
    sweep_dn = (df["low"] < last_pl) & (df["close"] > last_pl)
    sweep_up_ev = sweep_up & (~sweep_up.shift(1).fillna(False))
    sweep_dn_ev = sweep_dn & (~sweep_dn.shift(1).fillna(False))
    up_ts = sweep_up_ev[sweep_up_ev].index
    dn_ts = sweep_dn_ev[sweep_dn_ev].index
    rows = []
    for t in up_ts:
        rows.append({"ts": t, "dir": "up", "level": float(last_ph.loc[t]) if pd.notna(last_ph.loc[t]) else np.nan})
    for t in dn_ts:
        rows.append({"ts": t, "dir": "down", "level": float(last_pl.loc[t]) if pd.notna(last_pl.loc[t]) else np.nan})
    ev = pd.DataFrame(rows)
    if ev.empty:
        return ev
    ev = ev.sort_values("ts").reset_index(drop=True)
    return ev


def _signals_from_smc(df_low: pd.DataFrame, sweep_high: pd.DataFrame, imb: pd.DataFrame, engulf_bull: pd.Series, engulf_bear: pd.Series, *, side: str, max_bars_after_sweep: int, max_bars_after_imbalance: int):
    if df_low is None or df_low.empty:
        return pd.Series(False, index=pd.Index([]))
    try:
        df_low = df_low.sort_index()
    except Exception:
        pass
    out = pd.Series(False, index=df_low.index)
    if sweep_high is None or sweep_high.empty or imb is None or imb.empty:
        return out

    dfp = pd.DataFrame({"ts": pd.to_datetime(df_low.index)})
    dfp["pos"] = np.arange(len(df_low), dtype=int)
    if "ts" not in dfp.columns:
        dfp["ts"] = pd.to_datetime(df_low.index)
    if "pos" not in dfp.columns:
        dfp["pos"] = np.arange(len(df_low), dtype=int)
    try:
        dfp = dfp.sort_values("ts").reset_index(drop=True)
    except Exception:
        pass

    if side == "long":
        sweeps = sweep_high[sweep_high["dir"] == "down"][["ts"]].copy()
        zones = imb[imb["side"] == "bull"][["ts", "upper", "lower"]].copy()
        inv = engulf_bull.reindex(df_low.index).fillna(False).values
    else:
        sweeps = sweep_high[sweep_high["dir"] == "up"][["ts"]].copy()
        zones = imb[imb["side"] == "bear"][["ts", "upper", "lower"]].copy()
        inv = engulf_bear.reindex(df_low.index).fillna(False).values

    if sweeps.empty or zones.empty:
        return out

    sweeps = sweeps.sort_values("ts").reset_index(drop=True).rename(columns={"ts": "sweep_ts"})
    zones = zones.sort_values("ts").reset_index(drop=True).rename(columns={"ts": "zone_ts"})

    sw = pd.merge_asof(dfp[["ts", "pos"]], sweeps, left_on="ts", right_on="sweep_ts", direction="backward")
    zn = pd.merge_asof(dfp[["ts", "pos"]], zones, left_on="ts", right_on="zone_ts", direction="backward")

    pos_map = pd.Series(dfp["pos"].values, index=dfp["ts"].values)
    sw_pos = sw["sweep_ts"].map(pos_map).astype("float")
    zn_pos = zn["zone_ts"].map(pos_map).astype("float")

    bars_since_sw = (dfp["pos"] - sw_pos).fillna(10**9).astype(int)
    bars_since_zn = (dfp["pos"] - zn_pos).fillna(10**9).astype(int)

    upper = zn["upper"].astype(float).values
    lower = zn["lower"].astype(float).values
    lo = df_low["low"].astype(float).values
    hi = df_low["high"].astype(float).values
    in_zone = (lo <= upper) & (hi >= lower)

    ok = (bars_since_sw <= int(max_bars_after_sweep)) & (bars_since_zn <= int(max_bars_after_imbalance)) & in_zone & inv
    out.loc[df_low.index] = ok
    return out




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
            st.plotly_chart(figp, use_container_width=True, key="pnl_equity_analytics")

            figh = go.Figure()
            figh.add_trace(go.Histogram(x=t['pnl'], nbinsx=30, name='PnL distribution'))
            figh.update_layout(height=220, template='plotly_dark', margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(figh, use_container_width=True, key="pnl_hist")

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
            with st.expander(f"📍 {symbol} - {pos['side'].upper()}", expanded=False):
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
            st.plotly_chart(fig_h, use_container_width=True, key="hour_pnl_bar")

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

st.subheader("🧾 Why no trades?")
if isinstance(portfolio, dict):
    meta = portfolio.get('meta') if isinstance(portfolio.get('meta'), dict) else {}
    blocked = meta.get("blocked") if isinstance(meta.get("blocked"), dict) else {}
    reasons = blocked.get("reasons") if isinstance(blocked.get("reasons"), dict) else {}

    c1, c2, c3 = st.columns(3)
    c1.metric("Total blocks", int(blocked.get("total") or 0))
    c2.metric("Cooldown blocks", int(reasons.get("cooldown") or 0))
    c3.metric("Weak blocks", int(reasons.get("weak") or 0))

    if reasons:
        r = pd.DataFrame([{"reason": str(k), "count": int(v)} for k, v in reasons.items()]).sort_values("count", ascending=False)
        st.dataframe(r, use_container_width=True)

    last_decisions = meta.get('last_decisions') if isinstance(meta.get('last_decisions'), list) else []
    if last_decisions:
        df_d = pd.DataFrame([d for d in last_decisions if isinstance(d, dict)])
        if not df_d.empty:
            cols = [c for c in ["ts", "symbol", "price", "trade_decision", "side", "signal_strength", "reasoning_short", "setup_score", "min_setup_score", "quality_mode", "atr_pct", "hour_utc"] if c in df_d.columns]
            st.dataframe(df_d[cols].tail(80), use_container_width=True)
        else:
            st.info("No decision rows yet.")
    else:
        st.info("No decisions yet. Press Start All (or wait one demo cycle) to generate decision logs.")
else:
    st.info("Portfolio data not found. Start the bot and press Start All to generate state.")

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

if chart_view == "TradingView (Interactive)":
    symbol = chart_symbol
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
    symbol = chart_symbol
    df15 = fetch_asset_data(symbol, period="60d", interval="15m")
    df1h = fetch_asset_data(symbol, period="6mo", interval="1h")
    df5 = fetch_asset_data(symbol, period="30d", interval="5m")

    if df15 is None or df15.empty:
        st.warning(f"No 15m data for {symbol}")
    else:
        df15 = _to_utc_index_df(df15)
        if df1h is not None and not df1h.empty:
            df1h = _to_utc_index_df(df1h)
        else:
            df1h = None
        if df5 is not None and not df5.empty:
            df5 = _to_utc_index_df(df5)
        else:
            df5 = None

        sweeps_15 = _sweep_events(df15, int(pine_pivot_len))
        sweeps_1h = _sweep_events(df1h, int(pine_pivot_len)) if df1h is not None else pd.DataFrame()

        bull15, bear15 = _imbalance_events(df15)
        imb15 = pd.concat([bull15, bear15], ignore_index=True) if (bull15 is not None and bear15 is not None) else pd.DataFrame()
        bull_ev_15, bear_ev_15 = _engulf_events(df15)

        s1_long = pd.Series(False, index=df15.index)
        s1_short = pd.Series(False, index=df15.index)
        if pine_s1_enabled and (df1h is not None) and (not sweeps_1h.empty) and (not imb15.empty):
            s1_long = _signals_from_smc(
                df15, sweeps_1h, imb15, bull_ev_15, bear_ev_15,
                side="long",
                max_bars_after_sweep=int(pine_max_bars_after_sweep),
                max_bars_after_imbalance=int(pine_max_bars_after_imbalance),
            )
            s1_short = _signals_from_smc(
                df15, sweeps_1h, imb15, bull_ev_15, bear_ev_15,
                side="short",
                max_bars_after_sweep=int(pine_max_bars_after_sweep),
                max_bars_after_imbalance=int(pine_max_bars_after_imbalance),
            )

        s2_long_15 = pd.Series(False, index=df15.index)
        s2_short_15 = pd.Series(False, index=df15.index)
        if pine_s2_enabled and (df5 is not None) and (not df5.empty) and (not sweeps_15.empty):
            bull5, bear5 = _imbalance_events(df5)
            imb5 = pd.concat([bull5, bear5], ignore_index=True) if (bull5 is not None and bear5 is not None) else pd.DataFrame()
            bull_ev_5, bear_ev_5 = _engulf_events(df5)
            s2_long = _signals_from_smc(
                df5, sweeps_15, imb5, bull_ev_5, bear_ev_5,
                side="long",
                max_bars_after_sweep=int(pine_max_bars_after_sweep),
                max_bars_after_imbalance=int(pine_max_bars_after_imbalance),
            )
            s2_short = _signals_from_smc(
                df5, sweeps_15, imb5, bull_ev_5, bear_ev_5,
                side="short",
                max_bars_after_sweep=int(pine_max_bars_after_sweep),
                max_bars_after_imbalance=int(pine_max_bars_after_imbalance),
            )
            s2_long_15 = s2_long.groupby(s2_long.index.floor("15min")).max().reindex(df15.index, fill_value=False)
            s2_short_15 = s2_short.groupby(s2_short.index.floor("15min")).max().reindex(df15.index, fill_value=False)

        ny_mask = pd.Series(True, index=df15.index)
        if pine_ny_session_only:
            ny_mask = df15.index.map(lambda x: _in_range_utc(x, 12 * 60, 21 * 60))
            ny_mask = pd.Series(ny_mask, index=df15.index)

        ny_2h_mask = pd.Series(True, index=df15.index)
        if pine_ny_first_2h_only:
            ny_2h_mask = df15.index.map(lambda x: _in_range_utc(x, 12 * 60, 14 * 60))
            ny_2h_mask = pd.Series(ny_2h_mask, index=df15.index)

        ldn_ok = pd.Series(True, index=df15.index)
        last_london_sweep_by_day = {}
        last_london_sweep_ts_today = None
        if pine_require_london_sweep and (not sweeps_15.empty):
            in_ldn = sweeps_15["ts"].map(lambda x: _in_range_utc(x, 7 * 60, 16 * 60))
            ldn_sweeps = sweeps_15.loc[in_ldn]
            for t in ldn_sweeps["ts"].tolist():
                d = pd.Timestamp(t).date()
                prev = last_london_sweep_by_day.get(d)
                if prev is None or pd.Timestamp(t) > pd.Timestamp(prev):
                    last_london_sweep_by_day[d] = t
            try:
                last_london_sweep_ts_today = last_london_sweep_by_day.get(df15.index[-1].date())
            except Exception:
                last_london_sweep_ts_today = None

            ldn_ok = df15.index.map(lambda x: (x.date() in last_london_sweep_by_day) and (pd.Timestamp(x) >= pd.Timestamp(last_london_sweep_by_day.get(x.date()))))
            ldn_ok = pd.Series(ldn_ok, index=df15.index)

        allow = ny_mask & ny_2h_mask & ldn_ok
        s1_long_raw = s1_long.copy()
        s1_short_raw = s1_short.copy()
        s2_long_raw = s2_long_15.copy()
        s2_short_raw = s2_short_15.copy()
        s1_long = s1_long & allow
        s1_short = s1_short & allow
        s2_long_15 = s2_long_15 & allow
        s2_short_15 = s2_short_15 & allow

        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df15.index,
                open=df15["open"],
                high=df15["high"],
                low=df15["low"],
                close=df15["close"],
                name=f"{symbol} 15m",
            )
        )

        if show_sessions_asia:
            _session_vrects(fig, df15.index, 0, 9 * 60, fillcolor="#5b3fd6", opacity=0.06, limit_days=int(session_days))
        if show_sessions_london:
            _session_vrects(fig, df15.index, 7 * 60, 16 * 60, fillcolor="#1f6feb", opacity=0.06, limit_days=int(session_days))
        if show_sessions_ny:
            _session_vrects(fig, df15.index, 12 * 60, 21 * 60, fillcolor="#ffb100", opacity=0.06, limit_days=int(session_days))

        zones_to_draw = []
        if show_imbalance_zones and (not imb15.empty):
            bull_z = imb15[imb15["side"] == "bull"].tail(int(zones_per_side)).to_dict("records")
            bear_z = imb15[imb15["side"] == "bear"].tail(int(zones_per_side)).to_dict("records")
            zones_to_draw = bull_z + bear_z

        x_end = df15.index[-1]
        for z in zones_to_draw:
            x0 = pd.Timestamp(z["ts"])
            if x0 > x_end:
                continue
            is_bull = (z.get("side") == "bull")
            color_fill = "rgba(46,204,113,0.10)" if is_bull else "rgba(231,76,60,0.10)"
            color_line = "rgba(46,204,113,0.35)" if is_bull else "rgba(231,76,60,0.35)"
            y_low = float(z["lower"])
            y_up = float(z["upper"])
            fig.add_shape(
                type="rect",
                xref="x",
                yref="y",
                x0=x0,
                x1=x_end,
                y0=y_low,
                y1=y_up,
                fillcolor=color_fill,
                line=dict(width=0),
                layer="below",
            )
            if show_zone_levels:
                fig.add_shape(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=x0,
                    x1=x_end,
                    y0=y_up,
                    y1=y_up,
                    line=dict(color=color_line, width=1, dash="dot"),
                    layer="below",
                )
                fig.add_shape(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=x0,
                    x1=x_end,
                    y0=y_low,
                    y1=y_low,
                    line=dict(color=color_line, width=1, dash="dot"),
                    layer="below",
                )
            if show_zone_labels:
                tag = "BULL" if is_bull else "BEAR"
                fig.add_annotation(x=x_end, y=y_up, text=f"{tag} U", showarrow=False, xanchor="left", font=dict(size=10, color=color_line))
                fig.add_annotation(x=x_end, y=y_low, text=f"{tag} L", showarrow=False, xanchor="left", font=dict(size=10, color=color_line))

        if show_sweeps and (not sweeps_15.empty):
            up = sweeps_15[sweeps_15["dir"] == "up"]
            dn = sweeps_15[sweeps_15["dir"] == "down"]
            if not up.empty:
                y_up = df15.reindex(pd.to_datetime(up["ts"]).values, method="ffill")["high"].values
                fig.add_trace(
                    go.Scatter(
                        x=up["ts"],
                        y=y_up,
                        mode="markers",
                        name="Sweep Up (15m)",
                        text=(up["level"].round(5).astype(str).tolist() if (show_sweep_level_labels and ("level" in up.columns)) else None),
                        marker=dict(color="tomato", size=9, symbol="triangle-up"),
                    )
                )
            if not dn.empty:
                y_dn = df15.reindex(pd.to_datetime(dn["ts"]).values, method="ffill")["low"].values
                fig.add_trace(
                    go.Scatter(
                        x=dn["ts"],
                        y=y_dn,
                        mode="markers",
                        name="Sweep Down (15m)",
                        text=(dn["level"].round(5).astype(str).tolist() if (show_sweep_level_labels and ("level" in dn.columns)) else None),
                        marker=dict(color="springgreen", size=9, symbol="triangle-down"),
                    )
                )

        if show_s1 and (s1_long is not None) and s1_long.any():
            pts = df15.loc[s1_long]
            fig.add_trace(go.Scatter(x=pts.index, y=pts["close"], mode="markers", name="S1 Long", marker=dict(color="#2ecc71", size=9, symbol="circle")))
        if show_s1 and (s1_short is not None) and s1_short.any():
            pts = df15.loc[s1_short]
            fig.add_trace(go.Scatter(x=pts.index, y=pts["close"], mode="markers", name="S1 Short", marker=dict(color="#e74c3c", size=9, symbol="circle")))
        if show_s2 and (s2_long_15 is not None) and s2_long_15.any():
            pts = df15.loc[s2_long_15]
            fig.add_trace(go.Scatter(x=pts.index, y=pts["close"], mode="markers", name="S2 Long", marker=dict(color="#27ae60", size=7, symbol="diamond")))
        if show_s2 and (s2_short_15 is not None) and s2_short_15.any():
            pts = df15.loc[s2_short_15]
            fig.add_trace(go.Scatter(x=pts.index, y=pts["close"], mode="markers", name="S2 Short", marker=dict(color="#c0392b", size=7, symbol="diamond")))

        fig.update_layout(
            height=820,
            template="plotly_dark",
            showlegend=True,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🧪 Проверка стратегии NY SMC (Pine)", expanded=False):
            now_ts = df15.index[-1]
            now_utc = pd.Timestamp(now_ts)
            if now_utc.tzinfo is not None:
                now_utc = now_utc.tz_convert("UTC").tz_localize(None)

            g_ny = bool(allow.iloc[-1]) if len(allow) else False
            g_ny_sess = bool(ny_mask.iloc[-1]) if len(ny_mask) else True
            g_ny_2h = bool(ny_2h_mask.iloc[-1]) if len(ny_2h_mask) else True
            g_ldn = bool(ldn_ok.iloc[-1]) if len(ldn_ok) else True

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Текущее время (UTC)", now_utc.strftime("%Y-%m-%d %H:%M"))
            c2.metric("NY-сессия", "OK" if g_ny_sess else "NO")
            c3.metric("Первые 2 часа NY", "OK" if g_ny_2h else "NO")
            c4.metric("Свип Лондона", "OK" if g_ldn else "NO")
            c5.metric("Вход сейчас", "YES" if g_ny else "NO")

            gate_from = None
            day0 = now_utc.normalize()
            candidates = []
            if pine_ny_session_only or pine_ny_first_2h_only:
                candidates.append(day0 + pd.Timedelta(hours=12))
            if pine_require_london_sweep:
                if last_london_sweep_ts_today is not None:
                    lt = pd.Timestamp(last_london_sweep_ts_today)
                    if lt.tzinfo is not None:
                        lt = lt.tz_convert("UTC").tz_localize(None)
                    candidates.append(lt)
                else:
                    candidates = []

            if candidates:
                try:
                    gate_from = max(candidates)
                except Exception:
                    gate_from = None

            if last_london_sweep_ts_today is not None:
                st.write(f"Последний свип в Лондоне (сегодня): {pd.Timestamp(last_london_sweep_ts_today)}")
            elif pine_require_london_sweep:
                st.write("Последний свип в Лондоне (сегодня): нет")

            if gate_from is not None:
                st.write(f"Гейт активен с (UTC): {pd.Timestamp(gate_from)}")
            else:
                st.write("Гейт активен с (UTC): —")

            st.write(f"Окно NY (UTC): 12:00–21:00, первые 2 часа: 12:00–14:00")

            def _last_time(mask: pd.Series):
                try:
                    if mask is None or mask.empty or (not bool(mask.any())):
                        return None
                    return pd.Timestamp(mask[mask].index[-1])
                except Exception:
                    return None

            rows = []
            t_s1l = _last_time(s1_long_raw)
            t_s1s = _last_time(s1_short_raw)
            t_s2l = _last_time(s2_long_raw)
            t_s2s = _last_time(s2_short_raw)
            rows.append({"сигнал": "S1 Long (raw)", "время": str(t_s1l) if t_s1l is not None else ""})
            rows.append({"сигнал": "S1 Short (raw)", "время": str(t_s1s) if t_s1s is not None else ""})
            rows.append({"сигнал": "S2 Long (raw)", "время": str(t_s2l) if t_s2l is not None else ""})
            rows.append({"сигнал": "S2 Short (raw)", "время": str(t_s2s) if t_s2s is not None else ""})

            t_s1l_a = _last_time(s1_long)
            t_s1s_a = _last_time(s1_short)
            t_s2l_a = _last_time(s2_long_15)
            t_s2s_a = _last_time(s2_short_15)
            rows.append({"сигнал": "S1 Long (после фильтров)", "время": str(t_s1l_a) if t_s1l_a is not None else ""})
            rows.append({"сигнал": "S1 Short (после фильтров)", "время": str(t_s1s_a) if t_s1s_a is not None else ""})
            rows.append({"сигнал": "S2 Long (после фильтров)", "время": str(t_s2l_a) if t_s2l_a is not None else ""})
            rows.append({"сигнал": "S2 Short (после фильтров)", "время": str(t_s2s_a) if t_s2s_a is not None else ""})

            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            if not g_ny:
                parts = []
                if pine_ny_session_only and (not g_ny_sess):
                    parts.append("вне NY-сессии")
                if pine_ny_first_2h_only and (not g_ny_2h):
                    parts.append("не первые 2 часа NY")
                if pine_require_london_sweep and (not g_ldn):
                    parts.append("нет свипа Лондона сегодня / ещё не наступил")
                if parts:
                    st.info("Фильтр блокирует вход: " + ", ".join(parts))

        last = []
        if s1_long.any():
            last.append({"сигнал": "S1 Long", "время": str(s1_long[s1_long].index[-1])})
        if s1_short.any():
            last.append({"сигнал": "S1 Short", "время": str(s1_short[s1_short].index[-1])})
        if s2_long_15.any():
            last.append({"сигнал": "S2 Long", "время": str(s2_long_15[s2_long_15].index[-1])})
        if s2_short_15.any():
            last.append({"сигнал": "S2 Short", "время": str(s2_short_15[s2_short_15].index[-1])})
        if last:
            st.dataframe(pd.DataFrame(last), width='stretch')
        else:
            st.info("Сигналов по NY SMC (Pine) на текущем отрезке не найдено (с учётом фильтров).")


st.markdown("---")

st.subheader("📷 CVision (Chart Screenshot)")
api_key = _get_openrouter_api_key()
uploaded = st.file_uploader("Upload a chart screenshot (PNG/JPG)", type=["png", "jpg", "jpeg"])
vision_model = st.text_input("Vision model (OpenRouter)", value=os.getenv("OPENROUTER_VISION_MODEL", "openai/gpt-4o-mini"))
vision_prompt = st.text_area(
    "Question / task",
    value=f"Analyze this chart for {chart_symbol} ({timeframe}). Identify trend, key levels, and a trade idea with TP/SL.",
    height=110,
)

if uploaded is not None:
    st.image(uploaded, use_container_width=True)
    if st.button("🔎 Run CVision", width='stretch'):
        if not api_key:
            st.error("OPENROUTER_API_KEY not found.")
        else:
            try:
                import base64
                b = uploaded.getvalue()
                mime = uploaded.type or "image/png"
                data_url = f"data:{mime};base64,{base64.b64encode(b).decode('utf-8')}"
                client = _openrouter_client(api_key)
                resp = client.chat.completions.create(
                    model=str(vision_model).strip(),
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": str(vision_prompt)},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        }
                    ],
                    temperature=0.2,
                    max_tokens=350,
                )
                st.success("CVision result:")
                st.write(resp.choices[0].message.content)
            except Exception as e:
                msg = str(e)
                if "401" in msg:
                    st.error("CVision error: 401 Unauthorized. Check that OPENROUTER_API_KEY is valid (or paste it in sidebar → OpenRouter).")
                else:
                    st.error(f"CVision error: {e}")
else:
    st.caption("Tip: take a screenshot from TradingView and upload it here.")

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

ai_symbol = chart_symbol
st.write(f"### 🔮 AI Analysis for {ai_symbol}")

if portfolio and ai_symbol in portfolio.get('positions', {}):
    pos = portfolio['positions'][ai_symbol]
    st.info(f"📍 Active Position: {pos['side'].upper()} | Entry: {pos['entry_price']:.5f}")

if st.button("New Analysis", key="ai_single"):
    api_key = _get_openrouter_api_key()
    if not api_key:
        st.error("API Key not found!")
    else:
        with st.spinner(f"AI analyzing {ai_symbol}..."):
            try:
                client = _openrouter_client(api_key)

                df_analysis = fetch_asset_data(ai_symbol, period="1mo", interval=timeframe)
                if df_analysis is not None:
                    latest = df_analysis.iloc[-1]

                    prompt = f"""
                    Analyze {ai_symbol}. Price: {latest['close']:.5f}.
                    TP: {manual_tp}%, SL: {manual_sl}%, Leverage: {leverage}x.

                    RESPOND ONLY IN ENGLISH (max 60 words):
                    1. FORECAST: [UP/DOWN/FLAT]
                    2. PROBABILITY: [0-100]%
                    3. RECOMMENDATION: [TRADE/WAIT]
                    4. LEVELS: [support/resistance or invalidation]
                    """

                    response = client.chat.completions.create(
                        model=os.getenv("OPENROUTER_TEXT_MODEL", "deepseek/deepseek-chat"),
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=220
                    )

                    st.success("Analysis ready!")
                    st.info(response.choices[0].message.content)
                else:
                    st.error(f"No data for {ai_symbol}")
            except Exception as e:
                msg = str(e)
                if "401" in msg:
                    st.error("AI Error: 401 Unauthorized. Check that OPENROUTER_API_KEY is valid (or paste it in sidebar → OpenRouter).")
                else:
                    st.error(f"AI Error: {e}")

if st.button("🚀 Trade This Asset", key="trade_single"):
    _post_bot_command({
        "command": "start_single",
        "symbol": ai_symbol,
        "tp": float(manual_tp),
        "sl": float(manual_sl),
        "leverage": int(leverage),
        "max_trades_per_day": int(max_trades_per_day),
        "daily_tp_target_percent": float(daily_tp_target_percent),
        "time": str(datetime.now())
    })
    st.success(f"Trade command for {ai_symbol} sent!")

st.markdown("---")
st.caption(f"© EURUSD AI Trading Bot | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
