
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

# ============================================
# COMPUTER VISION ANALYSIS
# ============================================
st.subheader("👁️ Анализ с помощью Компьютерного Зрения")

assistant_symbol = st.selectbox(
    "Выберите актив для анализа",
    assets if assets else ["EURUSD=X"],
    index=0,
    key="assistant_symbol"
)

if st.button("🚀 Запустить Полный Анализ (Компьютерное Зрение)", key="run_vision_analysis"):
    with st.spinner("Анализируем графики с помощью компьютерного зрения и DeepSeek..."):
        try:
            if not api_key_input:
                st.error("Нужен OPENROUTER_API_KEY (в .env или в поле слева).")
                st.stop()

            from trading_assistant import TradingAssistant
            assistant = TradingAssistant()
            
            assistant.api_key = api_key_input
            if assistant.client is None:
                try:
                    import openai
                    headers = {}
                    if OPENROUTER_SITE_URL:
                        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
                    if OPENROUTER_APP_NAME:
                        headers["X-Title"] = OPENROUTER_APP_NAME
                    assistant.client = openai.OpenAI(
                        api_key=api_key_input,
                        base_url="https://openrouter.ai/api/v1",
                        default_headers=headers if headers else None
                    )
                except Exception as e:
                    st.error(f"Ошибка инициализации клиента: {e}")
                    st.stop()
            
            result = assistant.full_analysis(assistant_symbol)
            
            if "error" in result:
                st.error(f"Ошибка: {result['error']}")
            else:
                analysis = result.get("final_recommendation", {})
                
                # Display overall trend
                trend = analysis.get("overall_trend", "neutral")
                trend_emoji = "🟢" if trend == "bullish" else "🔴" if trend == "bearish" else "🟡"
                st.metric("Общий Тренд", f"{trend_emoji} {trend.upper()}")
                
                # Display entry recommendation
                entry = analysis.get("entry_recommendation", {})
                if entry:
                    st.subheader("🎯 Рекомендация по Входу")
                    col_e1, col_e2, col_e3 = st.columns(3)
                    col_e1.metric("Направление", entry.get("direction", "wait").upper())
                    col_e2.metric("Цена Входа", f"{entry.get('entry_price', 0):.5f}")
                    col_e3.metric("Риск/Доход", analysis.get("risk_reward_ratio", "N/A"))
                    
                    col_tp1, col_tp2, col_tp3 = st.columns(3)
                    col_tp1.metric("Тейк-Профит 1", f"{entry.get('take_profit_1', 0):.5f}")
                    col_tp2.metric("Тейк-Профит 2", f"{entry.get('take_profit_2', 0):.5f}")
                    col_tp3.metric("Тейк-Профит 3", f"{entry.get('take_profit_3', 0):.5f}")
                    st.metric("Стоп-Лосс", f"{entry.get('stop_loss', 0):.5f}")
                
                # Display Smart Money analysis
                smc_analysis = analysis.get("smart_money_analysis", "")
                if smc_analysis:
                    st.subheader("🧠 Анализ по Концепциям Smart Money + Компьютерное Зрение")
                    st.write(smc_analysis)
                
                # Display confidence
                confidence = analysis.get("confidence", 0)
                st.metric("Уверенность AI", f"{confidence}%")
                
                # Display charts for all timeframes
                st.subheader("📊 Графики на всех Таймфреймах")
                charts = result.get("charts", {})
                if charts:
                    tf_cols = st.columns(3)
                    for i, (tf, chart_img) in enumerate(charts.items()):
                        with tf_cols[i % 3]:
                            st.image(chart_img, caption=f"{assistant.timeframes.get(tf, tf)}", use_container_width=True)
                
                # Display computer vision analysis for each timeframe
                vision_analyses = result.get("vision_analyses", {})
                if vision_analyses:
                    st.subheader("👁️ Анализ Компьютерным Зрением по Таймфреймам")
                    for tf, vision_data in vision_analyses.items():
                        tf_name = assistant.timeframes.get(tf, tf)
                        with st.expander(f"{tf_name} - Анализ Компьютерным Зрением", expanded=False):
                            if "error" in vision_data:
                                st.error(f"Ошибка анализа зрения: {vision_data['error']}")
                            else:
                                st.write(f"**Тренд (Зрение):** {vision_data.get('trend', 'N/A')}")
                                
                                support = vision_data.get('support_levels', [])
                                if support:
                                    st.write(f"**Уровни Поддержки:** {', '.join([f'{s:.5f}' for s in support])}")
                                
                                resistance = vision_data.get('resistance_levels', [])
                                if resistance:
                                    st.write(f"**Уровни Сопротивления:** {', '.join([f'{r:.5f}' for r in resistance])}")
                                
                                entry_vision = vision_data.get('potential_entry', {})
                                if entry_vision:
                                    st.write(f"**Потенциальный Вход:** {entry_vision.get('direction', 'none').upper()}")
                                    st.write(f"**Цена Входа:** {entry_vision.get('entry_price', 0):.5f}")
                                    st.write(f"**Стоп-Лосс:** {entry_vision.get('stop_loss', 0):.5f}")
                                    st.write(f"**Тейк-Профит:** {entry_vision.get('take_profit', 0):.5f}")
                                    st.write(f"**Уверенность:** {entry_vision.get('confidence', 0)}%")
                                
                                notes = vision_data.get('analysis_notes', '')
                                if notes:
                                    st.write(f"**Заметки:** {notes}")
                
                # Display timeframe-by-timeframe analysis from final recommendation
                tf_analysis_list = analysis.get("timeframe_analysis", [])
                if tf_analysis_list:
                    st.subheader("📅 Разбор по Таймфреймам (Финальная рекомендация)")
                    for tf_analysis in tf_analysis_list:
                        with st.expander(f"{tf_analysis.get('timeframe', 'N/A')}", expanded=False):
                            st.write(f"**Тренд:** {tf_analysis.get('trend', 'neutral')}")
                            levels = tf_analysis.get('key_levels', {})
                            if levels:
                                st.write(f"**Поддержка:** {levels.get('support', 0):.5f}")
                                st.write(f"**Сопротивление:** {levels.get('resistance', 0):.5f}")
                            notes = tf_analysis.get('notes', '')
                            if notes:
                                st.write(f"**Заметки:** {notes}")
                
        except Exception as e:
            st.error(f"Ошибка при запуске анализа: {e}")
            import traceback
            st.code(traceback.format_exc())

st.markdown("---")
st.info("💡 Этот дашборд использует только компьютерное зрение для анализа графиков на 5 таймфреймах (от недели до 5 минут с помощью концепций Smart Money.")

