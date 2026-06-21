
"""
Streamlit Dashboard for Computer Vision Trading Assistant
Focused exclusively on computer vision analysis with DeepSeek
"""

import os
import json

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
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or ""
OPENROUTER_SITE_URL = (os.getenv("OPENROUTER_SITE_URL") or "").strip()
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME") or ""

st.set_page_config(page_title="🤖 Компьютерное Зрение - Торговый Ассистент", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 1.6rem; max-width: 1280px; }
      [data-testid="stSidebar"] { width: 360px; }
      header { visibility: hidden; height: 0px; }
      footer { visibility: hidden; height: 0px; }
      #MainMenu { visibility: hidden; }
      div[data-testid="stMetric"] { background: rgba(255,255,255,0.03); padding: 12px 12px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.06); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================
# SIDEBAR
# ============================================
st.sidebar.title("⚙️ Конфигурация")

with st.sidebar.expander("🔑 OpenRouter", expanded=True):
    api_key_input = st.text_input("API key (сессия)", type="password", value=OPENROUTER_API_KEY, key="openrouter_api_key")
    if not api_key_input:
        st.error("OPENROUTER_API_KEY отсутствует.")
    else:
        st.success("API key установлен.")

# Assets
st.sidebar.subheader("📊 Активы")
assets = st.sidebar.multiselect(
    "Выберите активы для анализа",
    [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "XAUUSD=X", "GLD",
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD",
        "AVAX-USD", "LINK-USD", "DOT-USD", "LTC-USD", "TRX-USD",
    ],
    default=["EURUSD=X"]
)

st.sidebar.subheader("👁️ Компьютерное Зрение")
st.sidebar.info("Компьютерное зрение используется для анализа всех графиков на 5 таймфреймах: Неделя → 4ч → 1ч → 15м → 5м")

st.sidebar.caption(f"Последнее обновление: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================
# MAIN TITLE
# ============================================
st.title("🤖 Торговый Ассистент с Компьютерным Зрением")
st.markdown("SMC • Multi-Timeframe Analysis • DeepSeek AI")
st.markdown("---")

st.subheader("👁️ Анализ с помощью Компьютерного Зрения")

assistant_symbol = st.selectbox(
    "Выберите актив для анализа",
    assets if assets else ["EURUSD=X"],
    index=0,
    key="assistant_symbol"
)

def _tv_symbol(sym: str) -> str:
    s = str(sym or "").strip()
    if not s:
        return "FX_IDC:EURUSD"
    if s.endswith("=X"):
        raw = s.replace("=X", "")
        return f"FX_IDC:{raw}"
    if s.endswith("-USD"):
        raw = s.replace("-", "")
        return f"BINANCE:{raw}T"
    return s

with st.expander("📺 TradingView (просмотр)", expanded=True):
    tv = _tv_symbol(assistant_symbol)
    tv_interval = st.selectbox("Таймфрейм отображения", ["1W", "240", "60", "15", "5"], index=2, help="Это только просмотр. Для CV используем авто-режим или 5 скриншотов.")
    tradingview_html = f"""
    <div class="tradingview-widget-container" style="height:680px;width:100%;">
      <div id="tradingview_{tv.replace(':','_')}" style="height:680px;width:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{tv}",
        "interval": "{tv_interval}",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "ru",
        "toolbar_bg": "#0e1117",
        "enable_publishing": false,
        "withdateranges": true,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "details": true,
        "hotlist": false,
        "calendar": false,
        "container_id": "tradingview_{tv.replace(':','_')}"
      }});
      </script>
    </div>
    """
    components.html(tradingview_html, height=710)

user_prompt_ru = st.text_area(
    "Контекст/задание (опционально)",
    value="Сделай SMC+SNR анализ и дай конкретный план входа. Учитывай структуру: Weekly -> 4H -> 1H -> 15m -> 5m.",
    height=80,
    key="user_prompt_ru",
)

tf_labels = {
    "1wk": "Неделя (1W)",
    "4h": "4 часа (4H)",
    "1h": "1 час (1H)",
    "15m": "15 минут (15m)",
    "5m": "5 минут (5m)",
}

def _to_conf01(v):
    try:
        x = float(v)
    except Exception:
        return 0.0
    if not (x == x):
        return 0.0
    if x > 1.0:
        x = x / 100.0
    return float(max(0.0, min(1.0, x)))

def _pct_from_prices(entry: float, target: float) -> float:
    try:
        e = float(entry)
        t = float(target)
    except Exception:
        return 0.0
    if e <= 0:
        return 0.0
    return float(abs(t - e) / e * 100.0)

def _render_analysis_result(result: dict, images: dict, image_title: str, symbol: str) -> None:
    analysis = result.get("final_recommendation", {}) if isinstance(result, dict) else {}
    if isinstance(analysis, dict) and ("error" in analysis):
        st.error(f"Ошибка итогового объединения: {analysis.get('error')}")
        st.stop()

    st.session_state["last_final_reco"] = analysis
    st.session_state["last_symbol"] = symbol

    trend = (analysis.get("overall_trend") or "neutral").lower()
    trend_emoji = "🟢" if trend == "bullish" else "🔴" if trend == "bearish" else "🟡"
    st.metric("Общий тренд", f"{trend_emoji} {trend.upper()}")

    score = int(float(analysis.get("setup_score") or 0))
    align = int(float(analysis.get("alignment_score") or 0))
    trade_allowed = bool(analysis.get("trade_allowed"))
    q1, q2, q3 = st.columns(3)
    q1.metric("Setup Score", str(score))
    q2.metric("MTF Alignment", f"{align}%")
    q3.metric("Trade Allowed", "YES" if trade_allowed else "NO")

    entry = analysis.get("entry_recommendation", {}) if isinstance(analysis, dict) else {}
    if isinstance(entry, dict) and entry:
        st.subheader("🎯 Рекомендация по входу")
        e1, e2, e3 = st.columns(3)
        e1.metric("Направление", str(entry.get("direction") or "wait").upper())
        e2.metric("Цена входа", f"{float(entry.get('entry_price') or 0.0):.5f}")
        e3.metric("Риск/доход", str(analysis.get("risk_reward_ratio") or "N/A"))

        t1, t2, t3 = st.columns(3)
        t1.metric("TP1", f"{float(entry.get('take_profit_1') or 0.0):.5f}")
        t2.metric("TP2", f"{float(entry.get('take_profit_2') or 0.0):.5f}")
        t3.metric("TP3", f"{float(entry.get('take_profit_3') or 0.0):.5f}")
        st.metric("SL", f"{float(entry.get('stop_loss') or 0.0):.5f}")

    smc_text = analysis.get("smart_money_analysis") if isinstance(analysis, dict) else ""
    if isinstance(smc_text, str) and smc_text.strip():
        st.subheader("🧠 Итоговый разбор (SMC + компьютерное зрение)")
        st.write(smc_text)

    conf = analysis.get("confidence") if isinstance(analysis, dict) else None
    if conf is not None:
        try:
            st.metric("Уверенность", f"{int(float(conf))}%")
        except Exception:
            st.metric("Уверенность", str(conf))

    guardrails = analysis.get("guardrails") if isinstance(analysis, dict) else None
    if isinstance(guardrails, list) and guardrails:
        st.subheader("🛡️ Guardrails")
        for g in guardrails:
            st.write(f"- {g}")

    consensus = analysis.get("consensus") if isinstance(analysis, dict) else None
    if isinstance(consensus, dict) and consensus:
        rows = []
        for tf in ["1wk", "4h", "1h", "15m", "5m"]:
            item = consensus.get(tf)
            if isinstance(item, dict):
                rows.append({
                    "timeframe": tf_labels.get(tf, tf),
                    "trend": item.get("trend"),
                    "direction": item.get("direction"),
                    "confidence": item.get("confidence"),
                    "score": item.get("score"),
                })
        if rows:
            st.subheader("📊 Consensus Engine")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader(image_title)
    i1, i2, i3 = st.columns(3)
    cols = [i1, i2, i3]
    for idx, tf in enumerate(["1wk", "4h", "1h", "15m", "5m"]):
        item = images.get(tf) if isinstance(images, dict) else None
        if not item:
            continue
        with cols[idx % 3]:
            if isinstance(item, dict):
                if "bytes" in item:
                    st.image(item["bytes"], caption=tf_labels.get(tf, tf), use_container_width=True)
            else:
                try:
                    st.image(item.getvalue(), caption=tf_labels.get(tf, tf), use_container_width=True)
                except Exception:
                    pass

    vision_analyses = result.get("vision_analyses", {}) if isinstance(result, dict) else {}
    if isinstance(vision_analyses, dict) and vision_analyses:
        st.subheader("👁️ Анализ зрения по таймфреймам")
        for tf in ["1wk", "4h", "1h", "15m", "5m"]:
            vision_data = vision_analyses.get(tf) if isinstance(vision_analyses, dict) else None
            tf_name = tf_labels.get(tf, tf)
            with st.expander(tf_name, expanded=False):
                if not isinstance(vision_data, dict):
                    st.write("Нет данных.")
                    continue
                if "error" in vision_data:
                    st.error(str(vision_data.get("error")))
                    continue
                st.write(f"Тренд: {vision_data.get('trend')}")
                sup = vision_data.get("support_levels") or []
                res = vision_data.get("resistance_levels") or []
                if sup:
                    st.write("Поддержки: " + ", ".join([f"{float(x):.5f}" for x in sup if x is not None]))
                if res:
                    st.write("Сопротивления: " + ", ".join([f"{float(x):.5f}" for x in res if x is not None]))
                pe = vision_data.get("potential_entry") or {}
                if isinstance(pe, dict) and pe:
                    st.write(
                        f"Сценарий: {str(pe.get('direction') or 'none').upper()} | "
                        f"Entry={float(pe.get('entry_price') or 0.0):.5f} | "
                        f"SL={float(pe.get('stop_loss') or 0.0):.5f} | "
                        f"TP1={float(pe.get('take_profit_1') or 0.0):.5f} | "
                        f"TP2={float(pe.get('take_profit_2') or 0.0):.5f} | "
                        f"Conf={pe.get('confidence')}"
                    )
                inv = vision_data.get("invalidation")
                if isinstance(inv, str) and inv.strip():
                    st.write("Отмена: " + inv)
                notes = vision_data.get("analysis_notes")
                if isinstance(notes, str) and notes.strip():
                    st.write(notes)

    tf_analysis_list = analysis.get("timeframe_analysis", []) if isinstance(analysis, dict) else []
    if isinstance(tf_analysis_list, list) and tf_analysis_list:
        st.subheader("📅 Разбор по таймфреймам (итог)")
        for row in tf_analysis_list:
            if not isinstance(row, dict):
                continue
            with st.expander(str(row.get("timeframe") or "N/A"), expanded=False):
                st.write("Тренд: " + str(row.get("trend") or "neutral"))
                levels = row.get("key_levels") if isinstance(row.get("key_levels"), dict) else {}
                if levels:
                    try:
                        st.write(f"Поддержка: {float(levels.get('support') or 0.0):.5f}")
                        st.write(f"Сопротивление: {float(levels.get('resistance') or 0.0):.5f}")
                    except Exception:
                        st.write(str(levels))
                notes = row.get("notes")
                if isinstance(notes, str) and notes.strip():
                    st.write(notes)

c1, c2, c3 = st.columns(3)
with c1:
    up_1wk = st.file_uploader("Скриншот графика: Неделя (1W)", type=["png", "jpg", "jpeg"], key="up_1wk")
    up_4h = st.file_uploader("Скриншот графика: 4H", type=["png", "jpg", "jpeg"], key="up_4h")
with c2:
    up_1h = st.file_uploader("Скриншот графика: 1H", type=["png", "jpg", "jpeg"], key="up_1h")
    up_15m = st.file_uploader("Скриншот графика: 15m", type=["png", "jpg", "jpeg"], key="up_15m")
with c3:
    up_5m = st.file_uploader("Скриншот графика: 5m", type=["png", "jpg", "jpeg"], key="up_5m")

uploads = {"1wk": up_1wk, "4h": up_4h, "1h": up_1h, "15m": up_15m, "5m": up_5m}

st.markdown("### ₿ Аналитика BTC/USD")
if st.button("Обновить market snapshot BTC/USD", key="btc_snapshot_btn"):
    st.session_state["btc_snapshot_refresh"] = datetime.now().isoformat()

try:
    from trading_assistant import TradingAssistant
    _btc_assistant = TradingAssistant()
    btc_snapshot = _btc_assistant.build_market_snapshot("BTC-USD")
except Exception as e:
    btc_snapshot = {"error": str(e)}

if isinstance(btc_snapshot, dict) and ("error" not in btc_snapshot):
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("BTC/USD", f"{float(btc_snapshot.get('current_price') or 0.0):.2f}")
    b2.metric("24ч", f"{float(btc_snapshot.get('day_change_pct') or 0.0):+.2f}%")
    b3.metric("7д", f"{float(btc_snapshot.get('week_change_pct') or 0.0):+.2f}%")
    b4.metric("30д", f"{float(btc_snapshot.get('month_change_pct') or 0.0):+.2f}%")
    b5, b6, b7 = st.columns(3)
    b5.metric("Trend D1", str(btc_snapshot.get("trend_daily") or "neutral").upper())
    b6.metric("Vol 30d", f"{float(btc_snapshot.get('volatility_30d_pct') or 0.0):.2f}%")
    b7.metric("VolRatio 24h/7d", f"{float(btc_snapshot.get('volume_ratio_24h_vs_7d') or 0.0):.2f}")
    with st.expander("Уровни BTC/USD", expanded=False):
        sup = btc_snapshot.get("support_levels") or []
        res = btc_snapshot.get("resistance_levels") or []
        if sup:
            st.write("Поддержки: " + ", ".join([f"{float(x):.2f}" for x in sup]))
        if res:
            st.write("Сопротивления: " + ", ".join([f"{float(x):.2f}" for x in res]))
else:
    st.warning("BTC snapshot пока недоступен.")

if st.button("₿ Запустить авто-анализ BTC/USD", key="run_btc_auto_vision"):
    with st.spinner("Строим MTF-аналитику для BTC/USD..."):
        try:
            if not api_key_input:
                st.error("Нужен OPENROUTER_API_KEY (в .env или в поле слева).")
                st.stop()

            from trading_assistant import TradingAssistant
            assistant = TradingAssistant()
            assistant.client = None
            assistant.ensure_client(api_key_input)

            btc_images = assistant.build_images_from_market_data("BTC-USD")
            required = ["1wk", "4h", "1h", "15m", "5m"]
            missing = [tf for tf in required if tf not in btc_images]
            if missing:
                st.error("Не удалось построить автографики BTC/USD для: " + ", ".join([tf_labels.get(x, x) for x in missing]))
                st.stop()

            btc_result = assistant.full_vision_assessment(symbol="BTC-USD", images=btc_images, user_prompt_ru=user_prompt_ru)
            if "error" in btc_result:
                st.error(f"Ошибка BTC/USD: {btc_result['error']}")
                st.stop()

            _render_analysis_result(btc_result, btc_images, "🖼️ BTC/USD Автографики", "BTC-USD")
        except Exception as e:
            st.error(f"Ошибка при запуске BTC/USD анализа: {e}")
            import traceback
            st.code(traceback.format_exc())

st.markdown("###  Авто-анализ без скриншотов (данные -> автографики -> computer vision)")
if st.button("⚡ Авто-анализ без скриншотов", key="run_auto_vision_analysis"):
    with st.spinner("Тянем данные, строим 5 автографиков и запускаем computer vision..."):
        try:
            if not api_key_input:
                st.error("Нужен OPENROUTER_API_KEY (в .env или в поле слева).")
                st.stop()

            from trading_assistant import TradingAssistant
            assistant = TradingAssistant()
            assistant.client = None
            assistant.ensure_client(api_key_input)

            images = assistant.build_images_from_market_data(assistant_symbol)
            required = ["1wk", "4h", "1h", "15m", "5m"]
            missing = [tf for tf in required if tf not in images]
            if missing:
                st.error("Не удалось построить автографики для: " + ", ".join([tf_labels.get(x, x) for x in missing]))
                st.stop()

            result = assistant.full_vision_assessment(symbol=assistant_symbol, images=images, user_prompt_ru=user_prompt_ru)
            if "error" in result:
                st.error(f"Ошибка: {result['error']}")
                st.stop()
            _render_analysis_result(result, images, "🖼️ Автографики (по данным)", assistant_symbol)

        except Exception as e:
            st.error(f"Ошибка при запуске авто-анализа: {e}")
            import traceback
            st.code(traceback.format_exc())

with st.expander("🤖 Бот (управление + быстрый анализ картинки)", expanded=False):
    data_dir = os.getenv("TRADEBOT_DATA_DIR", "data")
    try:
        os.makedirs(data_dir, exist_ok=True)
    except Exception:
        pass
    cmd_path = os.path.join(data_dir, "bot_command.json")
    portfolio_path = os.path.join(data_dir, "portfolio_state.json")
    trades_path = os.path.join(data_dir, "trade_history.csv")
    signal_journal_path = os.path.join(data_dir, "signal_journal.csv")

    def _write_bot_command(payload: dict) -> bool:
        try:
            with open(cmd_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            return True
        except Exception:
            return False

    st.write("Управление работает через файл команды: " + cmd_path)

    if st.button("🔄 Обновить статус бота", key="refresh_bot_status", use_container_width=True):
        st.rerun()

    port = {}
    if os.path.exists(portfolio_path):
        try:
            with open(portfolio_path, "r", encoding="utf-8") as f:
                port = json.load(f) or {}
        except Exception:
            port = {}

    trades_df = None
    if os.path.exists(trades_path):
        try:
            trades_df = pd.read_csv(trades_path)
        except Exception:
            trades_df = None

    signal_df = None
    if os.path.exists(signal_journal_path):
        try:
            signal_df = pd.read_csv(signal_journal_path)
        except Exception:
            signal_df = None

    bal = float(port.get("balance") or 0.0) if isinstance(port, dict) else 0.0
    eq = float(port.get("equity") or bal) if isinstance(port, dict) else bal
    positions = port.get("positions") if isinstance(port, dict) else {}
    pos_count = len(positions) if isinstance(positions, dict) else 0
    trades_count = int(len(trades_df)) if isinstance(trades_df, pd.DataFrame) else 0

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Баланс", f"{bal:.2f}")
    s2.metric("Эквити", f"{eq:.2f}")
    s3.metric("Позиций", str(pos_count))
    s4.metric("Сделок", str(trades_count))

    if isinstance(signal_df, pd.DataFrame) and (not signal_df.empty):
        last_signals = signal_df.tail(50).copy()
        accepted_count = int((last_signals["status"].astype(str) == "accepted").sum()) if "status" in last_signals.columns else 0
        rejected_count = int((last_signals["status"].astype(str) == "rejected").sum()) if "status" in last_signals.columns else 0
        j1, j2, j3 = st.columns(3)
        j1.metric("Сигналов", str(len(last_signals)))
        j2.metric("Принято", str(accepted_count))
        j3.metric("Отклонено", str(rejected_count))

    if isinstance(positions, dict) and positions:
        with st.expander("Открытые позиции", expanded=False):
            st.json(positions)

    if isinstance(trades_df, pd.DataFrame) and (not trades_df.empty):
        with st.expander("Последние сделки", expanded=False):
            st.dataframe(trades_df.tail(30), use_container_width=True)

    if isinstance(signal_df, pd.DataFrame) and (not signal_df.empty):
        with st.expander("Журнал сигналов", expanded=False):
            st.dataframe(signal_df.tail(50), use_container_width=True)

    meta = port.get("meta") if isinstance(port, dict) else {}
    current_filters = meta.get("filters") if isinstance(meta, dict) else {}
    if not isinstance(current_filters, dict):
        current_filters = {}

    st.markdown("#### 🎛️ Пороги качества Vision")
    st.caption("Эти параметры отправляются в бота через `set_filters` и влияют на серверную валидацию `vision_trade`.")

    vision_presets = {
        "conservative": {
            "vision_trade_enabled": True,
            "vision_trade_min_conf": 0.72,
            "vision_trade_min_rr": 2.0,
            "vision_trade_min_setup_score": 75.0,
            "vision_trade_min_alignment_score": 0.70,
            "vision_trade_max_sl_pct": 4.0,
            "btc_vision_profile_enabled": True,
            "btc_vision_min_conf": 0.78,
            "btc_vision_min_rr": 2.2,
            "btc_vision_min_setup_score": 82.0,
            "btc_vision_min_alignment_score": 0.74,
            "btc_vision_max_sl_pct": 3.8,
        },
        "balanced": {
            "vision_trade_enabled": True,
            "vision_trade_min_conf": 0.62,
            "vision_trade_min_rr": 1.4,
            "vision_trade_min_setup_score": 60.0,
            "vision_trade_min_alignment_score": 0.55,
            "vision_trade_max_sl_pct": 6.0,
            "btc_vision_profile_enabled": True,
            "btc_vision_min_conf": 0.70,
            "btc_vision_min_rr": 1.8,
            "btc_vision_min_setup_score": 72.0,
            "btc_vision_min_alignment_score": 0.65,
            "btc_vision_max_sl_pct": 5.0,
        },
        "aggressive": {
            "vision_trade_enabled": True,
            "vision_trade_min_conf": 0.52,
            "vision_trade_min_rr": 1.1,
            "vision_trade_min_setup_score": 48.0,
            "vision_trade_min_alignment_score": 0.45,
            "vision_trade_max_sl_pct": 8.0,
            "btc_vision_profile_enabled": True,
            "btc_vision_min_conf": 0.62,
            "btc_vision_min_rr": 1.5,
            "btc_vision_min_setup_score": 60.0,
            "btc_vision_min_alignment_score": 0.55,
            "btc_vision_max_sl_pct": 6.2,
        },
    }

    market_profiles = {
        "btc": {
            "vision_trade_enabled": True,
            "vision_trade_min_conf": 0.66,
            "vision_trade_min_rr": 1.6,
            "vision_trade_min_setup_score": 64.0,
            "vision_trade_min_alignment_score": 0.58,
            "vision_trade_max_sl_pct": 6.5,
            "btc_vision_profile_enabled": True,
            "btc_vision_min_conf": 0.72,
            "btc_vision_min_rr": 1.9,
            "btc_vision_min_setup_score": 74.0,
            "btc_vision_min_alignment_score": 0.66,
            "btc_vision_max_sl_pct": 5.2,
        },
        "eurusd": {
            "vision_trade_enabled": True,
            "vision_trade_min_conf": 0.64,
            "vision_trade_min_rr": 1.5,
            "vision_trade_min_setup_score": 62.0,
            "vision_trade_min_alignment_score": 0.57,
            "vision_trade_max_sl_pct": 3.2,
            "btc_vision_profile_enabled": True,
            "btc_vision_min_conf": 0.70,
            "btc_vision_min_rr": 1.8,
            "btc_vision_min_setup_score": 72.0,
            "btc_vision_min_alignment_score": 0.65,
            "btc_vision_max_sl_pct": 5.0,
        },
        "gold": {
            "vision_trade_enabled": True,
            "vision_trade_min_conf": 0.67,
            "vision_trade_min_rr": 1.7,
            "vision_trade_min_setup_score": 66.0,
            "vision_trade_min_alignment_score": 0.60,
            "vision_trade_max_sl_pct": 4.6,
            "btc_vision_profile_enabled": True,
            "btc_vision_min_conf": 0.70,
            "btc_vision_min_rr": 1.8,
            "btc_vision_min_setup_score": 72.0,
            "btc_vision_min_alignment_score": 0.65,
            "btc_vision_max_sl_pct": 5.0,
        },
    }

    preset_labels = {
        "conservative": "Conservative",
        "balanced": "Balanced",
        "aggressive": "Aggressive",
    }

    def _apply_vision_preset(preset_key: str) -> None:
        preset = vision_presets.get(preset_key) or {}
        st.session_state["flt_vision_trade_enabled"] = bool(preset.get("vision_trade_enabled", True))
        st.session_state["flt_vision_trade_min_conf"] = float(preset.get("vision_trade_min_conf", 0.62))
        st.session_state["flt_vision_trade_min_rr"] = float(preset.get("vision_trade_min_rr", 1.4))
        st.session_state["flt_vision_trade_min_setup_score"] = float(preset.get("vision_trade_min_setup_score", 60.0))
        st.session_state["flt_vision_trade_min_alignment_score"] = float(preset.get("vision_trade_min_alignment_score", 0.55))
        st.session_state["flt_vision_trade_max_sl_pct"] = float(preset.get("vision_trade_max_sl_pct", 6.0))
        st.session_state["flt_btc_vision_profile_enabled"] = bool(preset.get("btc_vision_profile_enabled", True))
        st.session_state["flt_btc_vision_min_conf"] = float(preset.get("btc_vision_min_conf", 0.70))
        st.session_state["flt_btc_vision_min_rr"] = float(preset.get("btc_vision_min_rr", 1.8))
        st.session_state["flt_btc_vision_min_setup_score"] = float(preset.get("btc_vision_min_setup_score", 72.0))
        st.session_state["flt_btc_vision_min_alignment_score"] = float(preset.get("btc_vision_min_alignment_score", 0.65))
        st.session_state["flt_btc_vision_max_sl_pct"] = float(preset.get("btc_vision_max_sl_pct", 5.0))

    def _apply_market_profile(profile_key: str) -> None:
        preset = market_profiles.get(profile_key) or {}
        st.session_state["flt_vision_trade_enabled"] = bool(preset.get("vision_trade_enabled", True))
        st.session_state["flt_vision_trade_min_conf"] = float(preset.get("vision_trade_min_conf", 0.62))
        st.session_state["flt_vision_trade_min_rr"] = float(preset.get("vision_trade_min_rr", 1.4))
        st.session_state["flt_vision_trade_min_setup_score"] = float(preset.get("vision_trade_min_setup_score", 60.0))
        st.session_state["flt_vision_trade_min_alignment_score"] = float(preset.get("vision_trade_min_alignment_score", 0.55))
        st.session_state["flt_vision_trade_max_sl_pct"] = float(preset.get("vision_trade_max_sl_pct", 6.0))
        st.session_state["flt_btc_vision_profile_enabled"] = bool(preset.get("btc_vision_profile_enabled", True))
        st.session_state["flt_btc_vision_min_conf"] = float(preset.get("btc_vision_min_conf", 0.70))
        st.session_state["flt_btc_vision_min_rr"] = float(preset.get("btc_vision_min_rr", 1.8))
        st.session_state["flt_btc_vision_min_setup_score"] = float(preset.get("btc_vision_min_setup_score", 72.0))
        st.session_state["flt_btc_vision_min_alignment_score"] = float(preset.get("btc_vision_min_alignment_score", 0.65))
        st.session_state["flt_btc_vision_max_sl_pct"] = float(preset.get("btc_vision_max_sl_pct", 5.0))

    p1, p2 = st.columns([2, 1])
    with p1:
        selected_preset = st.selectbox(
            "Готовый пресет",
            options=["conservative", "balanced", "aggressive"],
            index=1,
            format_func=lambda x: preset_labels.get(x, x),
            key="vision_preset_select",
        )
    with p2:
        if st.button("Применить пресет", use_container_width=True, key="vision_apply_preset_btn"):
            _apply_vision_preset(selected_preset)
            st.success("Пресет применен к полям ниже.")

    st.caption("Рыночные профили подстраивают пороги под характер инструмента.")
    mp1, mp2, mp3, mp4 = st.columns(4)
    if mp1.button("BTC профиль", use_container_width=True, key="vision_market_btc"):
        _apply_market_profile("btc")
        st.success("Применен BTC-профиль.")
    if mp2.button("EURUSD профиль", use_container_width=True, key="vision_market_eurusd"):
        _apply_market_profile("eurusd")
        st.success("Применен EURUSD-профиль.")
    if mp3.button("Gold профиль", use_container_width=True, key="vision_market_gold"):
        _apply_market_profile("gold")
        st.success("Применен Gold-профиль.")
    auto_profile_label = "BTC" if "BTC" in str(assistant_symbol).upper() else "EURUSD" if str(assistant_symbol).upper() == "EURUSD=X" else "GOLD" if str(assistant_symbol).upper() in ("XAUUSD=X", "GLD") else "BALANCED"
    if mp4.button(f"Авто для {auto_profile_label}", use_container_width=True, key="vision_market_auto"):
        sym_upper = str(assistant_symbol).upper()
        if "BTC" in sym_upper:
            _apply_market_profile("btc")
        elif sym_upper == "EURUSD=X":
            _apply_market_profile("eurusd")
        elif sym_upper in ("XAUUSD=X", "GLD"):
            _apply_market_profile("gold")
        else:
            _apply_vision_preset("balanced")
        st.success("Автопрофиль применен.")

    q1, q2 = st.columns(2)
    with q1:
        vision_trade_enabled = st.checkbox(
            "Включить vision_trade",
            value=bool(current_filters.get("vision_trade_enabled", True)),
            key="flt_vision_trade_enabled",
        )
        vision_trade_min_conf = st.slider(
            "Мин. confidence",
            min_value=0.0,
            max_value=1.0,
            value=float(current_filters.get("vision_trade_min_conf", 0.62) or 0.62),
            step=0.01,
            key="flt_vision_trade_min_conf",
        )
        vision_trade_min_rr = st.number_input(
            "Мин. RR",
            min_value=0.1,
            max_value=10.0,
            value=float(current_filters.get("vision_trade_min_rr", 1.4) or 1.4),
            step=0.1,
            key="flt_vision_trade_min_rr",
        )
        vision_trade_min_setup_score = st.number_input(
            "Мин. setup score",
            min_value=0.0,
            max_value=100.0,
            value=float(current_filters.get("vision_trade_min_setup_score", 60.0) or 60.0),
            step=1.0,
            key="flt_vision_trade_min_setup_score",
        )
        vision_trade_min_alignment_score = st.slider(
            "Мин. alignment score",
            min_value=0.0,
            max_value=1.0,
            value=float(current_filters.get("vision_trade_min_alignment_score", 0.55) or 0.55),
            step=0.01,
            key="flt_vision_trade_min_alignment_score",
        )
        vision_trade_max_sl_pct = st.number_input(
            "Макс. SL (%)",
            min_value=0.1,
            max_value=50.0,
            value=float(current_filters.get("vision_trade_max_sl_pct", 6.0) or 6.0),
            step=0.1,
            key="flt_vision_trade_max_sl_pct",
        )
    with q2:
        btc_vision_profile_enabled = st.checkbox(
            "Включить BTC-профиль",
            value=bool(current_filters.get("btc_vision_profile_enabled", True)),
            key="flt_btc_vision_profile_enabled",
        )
        btc_vision_min_conf = st.slider(
            "BTC мин. confidence",
            min_value=0.0,
            max_value=1.0,
            value=float(current_filters.get("btc_vision_min_conf", 0.70) or 0.70),
            step=0.01,
            key="flt_btc_vision_min_conf",
        )
        btc_vision_min_rr = st.number_input(
            "BTC мин. RR",
            min_value=0.1,
            max_value=10.0,
            value=float(current_filters.get("btc_vision_min_rr", 1.8) or 1.8),
            step=0.1,
            key="flt_btc_vision_min_rr",
        )
        btc_vision_min_setup_score = st.number_input(
            "BTC мин. setup score",
            min_value=0.0,
            max_value=100.0,
            value=float(current_filters.get("btc_vision_min_setup_score", 72.0) or 72.0),
            step=1.0,
            key="flt_btc_vision_min_setup_score",
        )
        btc_vision_min_alignment_score = st.slider(
            "BTC мин. alignment",
            min_value=0.0,
            max_value=1.0,
            value=float(current_filters.get("btc_vision_min_alignment_score", 0.65) or 0.65),
            step=0.01,
            key="flt_btc_vision_min_alignment_score",
        )
        btc_vision_max_sl_pct = st.number_input(
            "BTC макс. SL (%)",
            min_value=0.1,
            max_value=50.0,
            value=float(current_filters.get("btc_vision_max_sl_pct", 5.0) or 5.0),
            step=0.1,
            key="flt_btc_vision_max_sl_pct",
        )

    if st.button("💾 Применить пороги Vision", use_container_width=True, key="apply_vision_filters"):
        payload = {
            "command": "set_filters",
            "vision_trade_enabled": bool(vision_trade_enabled),
            "vision_trade_min_conf": float(vision_trade_min_conf),
            "vision_trade_min_rr": float(vision_trade_min_rr),
            "vision_trade_min_setup_score": float(vision_trade_min_setup_score),
            "vision_trade_min_alignment_score": float(vision_trade_min_alignment_score),
            "vision_trade_max_sl_pct": float(vision_trade_max_sl_pct),
            "btc_vision_profile_enabled": bool(btc_vision_profile_enabled),
            "btc_vision_min_conf": float(btc_vision_min_conf),
            "btc_vision_min_rr": float(btc_vision_min_rr),
            "btc_vision_min_setup_score": float(btc_vision_min_setup_score),
            "btc_vision_min_alignment_score": float(btc_vision_min_alignment_score),
            "btc_vision_max_sl_pct": float(btc_vision_max_sl_pct),
            "time": datetime.now().isoformat(),
        }
        ok = _write_bot_command(payload)
        if ok:
            st.success("Команда отправлена: set_filters")
        else:
            st.error("Не удалось записать bot_command.json")

    b1, b2, b3 = st.columns(3)
    with b1:
        bot_tp = st.number_input("TP (%)", min_value=0.1, max_value=50.0, value=5.0, step=0.5, key="bot_tp")
    with b2:
        bot_sl = st.number_input("SL (%)", min_value=0.1, max_value=50.0, value=2.0, step=0.5, key="bot_sl")
    with b3:
        bot_leverage = st.number_input("Leverage", min_value=1, max_value=200, value=5, step=1, key="bot_leverage")

    st.markdown("#### 🔌 Режим торговли (Demo / Real)")
    mode_label = st.radio("Режим", ["Demo (Paper)", "Real (API)"], index=0, horizontal=True, key="bot_mode")
    is_real = mode_label == "Real (API)"
    exchange_id = st.selectbox("Биржа", ["bybit", "binance"], index=0, disabled=(not is_real), key="bot_exchange")
    api_k = st.text_input("API Key", type="password", disabled=(not is_real), key="bot_api_key")
    api_s = st.text_input("API Secret", type="password", disabled=(not is_real), key="bot_api_secret")
    confirm_api = st.checkbox("Подтвердить отправку ключей", value=False, key="bot_confirm_api")
    if st.button("💾 Применить режим/ключи", use_container_width=True, key="bot_apply_mode"):
        if is_real and (not confirm_api):
            st.error("Для REAL включите чекбокс подтверждения отправки ключей.")
            st.stop()
        payload = {
            "command": "update_api",
            "mode": ("real" if is_real else "demo"),
            "exchange": str(exchange_id),
            "key": (str(api_k) if is_real else None),
            "secret": (str(api_s) if is_real else None),
            "time": datetime.now().isoformat(),
        }
        ok = _write_bot_command(payload)
        if ok:
            st.success("Команда отправлена: update_api")
        else:
            st.error("Не удалось записать bot_command.json")

    c_start, c_stop = st.columns(2)
    if c_start.button("🚀 Start All (бот)", use_container_width=True):
        payload = {
            "command": "start_all",
            "symbols": list(assets or [assistant_symbol]),
            "tp": float(bot_tp),
            "sl": float(bot_sl),
            "leverage": int(bot_leverage),
            "time": datetime.now().isoformat(),
        }
        ok = _write_bot_command(payload)
        if ok:
            st.success("Команда отправлена: start_all")
        else:
            st.error("Не удалось записать bot_command.json")

    if c_stop.button("🛑 Stop All (бот)", use_container_width=True):
        payload = {"command": "stop_all", "time": datetime.now().isoformat()}
        ok = _write_bot_command(payload)
        if ok:
            st.success("Команда отправлена: stop_all")
        else:
            st.error("Не удалось записать bot_command.json")

    st.markdown("#### ⚡ Быстрый анализ картинки (1 скриншот)")
    quick_tf = st.selectbox("Таймфрейм картинки", ["1wk", "4h", "1h", "15m", "5m"], index=2, key="quick_tf")
    quick_file = st.file_uploader("Загрузите 1 скриншот графика для быстрого анализа", type=["png", "jpg", "jpeg"], key="quick_img")
    quick_prompt = st.text_area(
        "Задание для быстрого анализа (опционально)",
        value="Коротко: тренд, уровни, сценарий входа/SL/TP и отмена.",
        height=60,
        key="quick_prompt",
    )
    if st.button("👁️ Проанализировать 1 картинку", key="quick_analyze"):
        if not api_key_input:
            st.error("Нужен OPENROUTER_API_KEY (в .env или в поле слева).")
            st.stop()
        if quick_file is None:
            st.error("Загрузите картинку.")
            st.stop()
        try:
            from trading_assistant import TradingAssistant
            assistant = TradingAssistant()
            assistant.client = None
            assistant.ensure_client(api_key_input)
            out = assistant.analyze_timeframe_image(
                image_bytes=quick_file.getvalue(),
                mime=(quick_file.type or "image/png"),
                symbol=assistant_symbol,
                timeframe_key=quick_tf,
                user_prompt_ru=quick_prompt,
            )
            if isinstance(out, dict) and ("error" in out):
                st.error(str(out.get("error")))
            else:
                st.session_state["quick_vision_out"] = out
                st.session_state["quick_vision_symbol"] = assistant_symbol
                st.json(out)
        except Exception as e:
            st.error(f"Ошибка быстрого анализа: {e}")
            import traceback
            st.code(traceback.format_exc())

    st.markdown("#### 📤 Отправка результата в бота (vision_trade)")
    execute_now = st.checkbox("Выполнить сразу (execute_now)", value=True, key="vision_execute_now")
    allow_override = st.checkbox("Разрешить отправку даже если trade_allowed = NO", value=False, key="vision_allow_override")

    if st.button("📤 Отправить быстрый анализ в бота", key="send_quick_to_bot", use_container_width=True):
        q = st.session_state.get("quick_vision_out")
        sym = st.session_state.get("quick_vision_symbol") or assistant_symbol
        if not isinstance(q, dict):
            st.error("Сначала сделайте быстрый анализ картинки.")
            st.stop()
        pe = q.get("potential_entry") if isinstance(q.get("potential_entry"), dict) else {}
        direction = str(pe.get("direction") or "none").lower()
        if direction not in ("long", "short"):
            st.error("В быстром анализе нет сценария long/short.")
            st.stop()
        entry_price = float(pe.get("entry_price") or 0.0)
        sl_price = float(pe.get("stop_loss") or 0.0)
        tp_price = float(pe.get("take_profit_1") or pe.get("take_profit_2") or 0.0)
        conf = _to_conf01(pe.get("confidence") or q.get("confidence") or 0.0)
        tp_pct = _pct_from_prices(entry_price, tp_price) if (entry_price > 0 and tp_price > 0) else float(bot_tp)
        sl_pct = _pct_from_prices(entry_price, sl_price) if (entry_price > 0 and sl_price > 0) else float(bot_sl)
        strength = "strong" if conf >= 0.75 else "medium" if conf >= 0.55 else "weak"
        reason = str(q.get("analysis_notes") or q.get("invalidation") or "vision")[:200]
        payload = {
            "command": "vision_trade",
            "symbol": str(sym),
            "side": direction,
            "signal_strength": strength,
            "confidence": float(conf),
            "tp_pct": float(tp_pct),
            "sl_pct": float(sl_pct),
            "reasoning_short": reason,
            "source": "dashboard_quick_vision",
            "execute_now": bool(execute_now),
            "time": datetime.now().isoformat(),
        }
        ok = _write_bot_command(payload)
        if ok:
            st.success("Команда отправлена: vision_trade")
        else:
            st.error("Не удалось записать bot_command.json")

    if st.button("📤 Отправить последнюю MTF-рекомендацию в бота", key="send_last_mtf_to_bot", use_container_width=True):
        analysis = st.session_state.get("last_final_reco")
        sym = st.session_state.get("last_symbol") or assistant_symbol
        if not isinstance(analysis, dict):
            st.error("Нет последней рекомендации. Запустите авто-анализ или анализ по 5 скриншотам.")
            st.stop()
        if (not bool(analysis.get("trade_allowed"))) and (not allow_override):
            st.error("trade_allowed = NO. Либо улучшите сетап, либо включите override.")
            st.stop()
        entry = analysis.get("entry_recommendation") if isinstance(analysis.get("entry_recommendation"), dict) else {}
        direction = str(entry.get("direction") or "wait").lower()
        if direction not in ("long", "short"):
            st.error("В последней рекомендации направление не long/short.")
            st.stop()
        entry_price = float(entry.get("entry_price") or 0.0)
        sl_price = float(entry.get("stop_loss") or 0.0)
        tp_price = float(entry.get("take_profit_1") or 0.0)
        conf = _to_conf01(analysis.get("confidence") or 0.0)
        tp_pct = _pct_from_prices(entry_price, tp_price) if (entry_price > 0 and tp_price > 0) else float(bot_tp)
        sl_pct = _pct_from_prices(entry_price, sl_price) if (entry_price > 0 and sl_price > 0) else float(bot_sl)
        strength = "strong" if conf >= 0.75 else "medium" if conf >= 0.55 else "weak"
        reason = str(analysis.get("smart_money_analysis") or "vision_mtf")[:200]
        payload = {
            "command": "vision_trade",
            "symbol": str(sym),
            "side": direction,
            "signal_strength": strength,
            "confidence": float(conf),
            "tp_pct": float(tp_pct),
            "sl_pct": float(sl_pct),
            "setup_score": float(analysis.get("setup_score") or 0.0),
            "alignment_score": float(analysis.get("alignment_score") or 0.0),
            "trade_allowed": bool(analysis.get("trade_allowed")),
            "reasoning_short": reason,
            "source": "dashboard_mtf_vision",
            "execute_now": bool(execute_now),
            "time": datetime.now().isoformat(),
        }
        ok = _write_bot_command(payload)
        if ok:
            st.success("Команда отправлена: vision_trade (MTF)")
        else:
            st.error("Не удалось записать bot_command.json")

st.markdown("### 📤 Анализ по 5 скриншотам (резервный режим)")
if st.button("🚀 Запустить Полный Анализ (только компьютерное зрение)", key="run_vision_analysis"):
    with st.spinner("Анализируем 5 скриншотов (vision) и объединяем вывод (DeepSeek)..."):
        try:
            if not api_key_input:
                st.error("Нужен OPENROUTER_API_KEY (в .env или в поле слева).")
                st.stop()

            missing = [k for k, f in uploads.items() if f is None]
            if missing:
                st.error("Не хватает скриншотов для таймфреймов: " + ", ".join([tf_labels.get(k, k) for k in missing]))
                st.stop()

            images = {}
            for tf, f in uploads.items():
                images[tf] = {"bytes": f.getvalue(), "mime": (f.type or "image/png")}

            from trading_assistant import TradingAssistant
            assistant = TradingAssistant()
            assistant.client = None
            assistant.ensure_client(api_key_input)

            result = assistant.full_vision_assessment(symbol=assistant_symbol, images=images, user_prompt_ru=user_prompt_ru)
            if "error" in result:
                st.error(f"Ошибка: {result['error']}")
                st.stop()
            _render_analysis_result(result, images, "🖼️ Скриншоты", assistant_symbol)

        except Exception as e:
            st.error(f"Ошибка при запуске анализа: {e}")
            import traceback
            st.code(traceback.format_exc())

st.markdown("---")
st.info("💡 Этот дашборд использует только компьютерное зрение для анализа графиков на 5 таймфреймах (от недели до 5 минут с помощью концепций Smart Money.")

