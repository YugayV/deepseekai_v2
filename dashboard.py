
"""
Streamlit Dashboard for Computer Vision Trading Assistant
Focused exclusively on computer vision analysis with DeepSeek
"""

import os

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

            analysis = result.get("final_recommendation", {}) if isinstance(result, dict) else {}
            if isinstance(analysis, dict) and ("error" in analysis):
                st.error(f"Ошибка итогового объединения: {analysis.get('error')}")
                st.stop()

            trend = (analysis.get("overall_trend") or "neutral").lower()
            trend_emoji = "🟢" if trend == "bullish" else "🔴" if trend == "bearish" else "🟡"
            st.metric("Общий тренд", f"{trend_emoji} {trend.upper()}")

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

            st.subheader("�️ Скриншоты")
            i1, i2, i3 = st.columns(3)
            cols = [i1, i2, i3]
            for idx, (tf, f) in enumerate(uploads.items()):
                with cols[idx % 3]:
                    st.image(f.getvalue(), caption=tf_labels.get(tf, tf), use_container_width=True)

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

        except Exception as e:
            st.error(f"Ошибка при запуске анализа: {e}")
            import traceback
            st.code(traceback.format_exc())

st.markdown("---")
st.info("💡 Этот дашборд использует только компьютерное зрение для анализа графиков на 5 таймфреймах (от недели до 5 минут с помощью концепций Smart Money.")

