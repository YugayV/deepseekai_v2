"""
EURUSD AI Trading Bot with AI
Inspired by: https://github.com/tot-gromov/llm-deepseek-trading
Slim strategy defaults (no mandatory Alligator/Fractals)
"""

import os
import logging
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# BOT HTTP API (for 2-service Railway setup)
# - GET  /health    -> OK
# - POST /command   -> send command from dashboard
# - GET  /portfolio -> portfolio_state.json
# - GET  /trades    -> trade_history.csv as JSON
# ============================================
DEFAULT_PORT = 8080

def _resolve_listen_port() -> int:
    raw = os.getenv("PORT")
    if raw is None:
        return DEFAULT_PORT
    raw = str(raw).strip()
    if not raw:
        return DEFAULT_PORT
    try:
        port = int(raw)
    except Exception:
        return DEFAULT_PORT
    if port <= 0 or port > 65535:
        return DEFAULT_PORT
    return port

API_PORT = _resolve_listen_port()

DATA_DIR_EARLY = os.getenv("TRADEBOT_DATA_DIR", "data")
os.makedirs(DATA_DIR_EARLY, exist_ok=True)
PORTFOLIO_PATH_EARLY = os.path.join(DATA_DIR_EARLY, "portfolio_state.json")
TRADES_PATH_EARLY = os.path.join(DATA_DIR_EARLY, "trade_history.csv")

BOT_LOOP = None

class BotApiHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload):
        import json as _json
        body = _json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, text: str):
        body = text.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/health', '/'): 
            return self._send_text(200, 'OK')

        if self.path == '/portfolio':
            import json as _json
            if os.path.exists(PORTFOLIO_PATH_EARLY):
                try:
                    with open(PORTFOLIO_PATH_EARLY, 'r', encoding='utf-8') as f:
                        return self._send_json(200, _json.load(f))
                except Exception as e:
                    return self._send_json(500, {'error': str(e)})
            return self._send_json(200, {})

        if self.path == '/trades':
            if os.path.exists(TRADES_PATH_EARLY):
                try:
                    import pandas as _pd
                    df = _pd.read_csv(TRADES_PATH_EARLY)
                    return self._send_json(200, {'rows': df.to_dict('records')})
                except Exception as e:
                    return self._send_json(500, {'error': str(e)})
            return self._send_json(200, {'rows': []})

        return self._send_text(404, 'Not Found')

    def do_POST(self):
        if self.path != '/command':
            return self._send_text(404, 'Not Found')

        try:
            import json as _json
            length = int(self.headers.get('Content-Length', '0') or '0')
            raw = self.rfile.read(length) if length > 0 else b'{}'
            cmd = _json.loads(raw.decode('utf-8') or '{}')
        except Exception as e:
            return self._send_json(400, {'ok': False, 'error': f'bad_json: {e}'})

        try:
            if 'bot_instance' in globals() and bot_instance and BOT_LOOP:
                import asyncio as _asyncio
                fut = _asyncio.run_coroutine_threadsafe(bot_instance.apply_command(cmd), BOT_LOOP)
                fut.result(timeout=5)
                return self._send_json(200, {'ok': True})
            return self._send_json(503, {'ok': False, 'error': 'bot_not_ready'})
        except Exception as e:
            return self._send_json(500, {'ok': False, 'error': str(e)})

    def log_message(self, format, *args):
        return


def _create_api_server(port: int) -> HTTPServer:
    return HTTPServer(("0.0.0.0", port), BotApiHandler)


def start_api_server():
    port = API_PORT
    server = None

    try:
        server = _create_api_server(port)
    except Exception as e:
        if port != DEFAULT_PORT:
            try:
                server = _create_api_server(DEFAULT_PORT)
                port = DEFAULT_PORT
            except Exception:
                raise e
        else:
            raise

    print(f"✅ Bot API listening on 0.0.0.0:{port}")

    t = threading.Thread(target=server.serve_forever, daemon=False, name=f"api_{port}")
    t.start()


if __name__ == "__main__" and os.getenv("BOT_API_ENABLED", "").lower() in ("1", "true", "yes"):
    start_api_server()

# Now import heavy libraries with safety
try:
    import json
    import asyncio
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import joblib
    from datetime import datetime
    from dotenv import load_dotenv
    import openai

    from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
    from telegram.request import HTTPXRequest
except Exception as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    # Fail fast to avoid Streamlit hanging on import
    raise

load_dotenv()

# ============================================
# CONFIG
# ============================================
PAPER_CAPITAL = float(os.getenv("PAPER_START_CAPITAL", 10000))
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", 5.0))
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 2.0))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", 4.0))

MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", 5))
DAILY_TP_TARGET_PERCENT = float(os.getenv("DAILY_TP_TARGET_PERCENT", 10.0))
DEMO_CYCLE_SECONDS = int(os.getenv("DEMO_CYCLE_SECONDS", 300))
REAL_CYCLE_SECONDS = int(os.getenv("REAL_CYCLE_SECONDS", 600))

BLOCK_WEAK_SIGNALS = (os.getenv("BLOCK_WEAK_SIGNALS", "false").strip().lower() in ("1", "true", "yes"))
COOLDOWN_BARS = int(os.getenv("COOLDOWN_BARS", 0))
USE_ATR_RISK = (os.getenv("USE_ATR_RISK", "true").strip().lower() in ("1", "true", "yes"))
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", 1.5))
ATR_TP_MULT = float(os.getenv("ATR_TP_MULT", 2.5))

WAVE_WINDOW = int(os.getenv("WAVE_WINDOW", 64))
WAVE_LEVELS = int(os.getenv("WAVE_LEVELS", 4))
WAVE_TREND_BLOCK_PCT = float(os.getenv("WAVE_TREND_BLOCK_PCT", 0.30))
USE_WAVE_FILTER_DEFAULT = (os.getenv("USE_WAVE_FILTER_DEFAULT", "false").strip().lower() in ("1", "true", "yes"))

MACRO_SYMBOLS = [s.strip() for s in os.getenv("MACRO_SYMBOLS", "DX-Y.NYB,GC=F,CL=F,^TNX,^VIX").split(",") if s.strip()]
USE_MACRO_DEFAULT = (os.getenv("USE_MACRO_DEFAULT", "false").strip().lower() in ("1", "true", "yes"))
USE_POLYMARKET_DEFAULT = (os.getenv("USE_POLYMARKET_DEFAULT", "false").strip().lower() in ("1", "true", "yes"))
POLYMARKET_FEED_URL = (os.getenv("POLYMARKET_FEED_URL") or "").strip()
DXY_TREND_BLOCK_PCT = float(os.getenv("DXY_TREND_BLOCK_PCT", 0.20))
USE_DXY_FILTER_DEFAULT = (os.getenv("USE_DXY_FILTER_DEFAULT", "false").strip().lower() in ("1", "true", "yes"))

RAILWAY = os.getenv("RAILWAY", "false").lower() == "true"
TRADING_MODE = os.getenv("TRADING_MODE", "demo").lower()
STRATEGY_MODE = (os.getenv("STRATEGY_MODE") or "classic").strip().lower()
STRATEGY_VERSION = (os.getenv("STRATEGY_VERSION") or "v2").strip()

AI_API_KEY = os.getenv("OPENROUTER_API_KEY")
AI_MAX_TOKENS = int(os.getenv("AI_MAX_TOKENS", 96))
AI_MIN_TOKENS_ON_402 = int(os.getenv("AI_MIN_TOKENS_ON_402", 16))

DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
ENABLE_TRADE_REVIEW = (os.getenv("ENABLE_TRADE_REVIEW", "true").strip().lower() in ("1", "true", "yes"))
REVIEW_MAX_TOKENS = int(os.getenv("REVIEW_MAX_TOKENS", 96))
LESSONS_LIMIT = int(os.getenv("LESSONS_LIMIT", 8))

RISK_GUARD_ENABLED_DEFAULT = (os.getenv("RISK_GUARD_ENABLED_DEFAULT", "false").strip().lower() in ("1", "true", "yes"))
MAX_OPEN_POSITIONS_DEFAULT = int(os.getenv("MAX_OPEN_POSITIONS_DEFAULT", 2))
MAX_DAILY_DRAWDOWN_PCT_DEFAULT = float(os.getenv("MAX_DAILY_DRAWDOWN_PCT_DEFAULT", 10.0))
MAX_LOSS_STREAK_DEFAULT = int(os.getenv("MAX_LOSS_STREAK_DEFAULT", 6))
GUARD_PAUSE_SECONDS_DEFAULT = int(os.getenv("GUARD_PAUSE_SECONDS_DEFAULT", 300))

QUALITY_MODE_DEFAULT = (os.getenv("QUALITY_MODE_DEFAULT") or ("high" if (os.getenv("TRADING_MODE", "demo").lower() == "real") else "balanced")).strip().lower()
MIN_SETUP_SCORE_DEFAULT = int(os.getenv("MIN_SETUP_SCORE_DEFAULT", 4 if (os.getenv("TRADING_MODE", "demo").lower() == "real") else 3))
USE_SESSION_FILTER_DEFAULT = (os.getenv("USE_SESSION_FILTER_DEFAULT", "true" if (os.getenv("TRADING_MODE", "demo").lower() == "real") else "false").strip().lower() in ("1", "true", "yes"))
MIN_ATR_PCT_DEFAULT = float(os.getenv("MIN_ATR_PCT_DEFAULT", 0.05))
MAX_ATR_PCT_DEFAULT = float(os.getenv("MAX_ATR_PCT_DEFAULT", 1.50))

RISK_PER_TRADE_PCT_DEMO = float(os.getenv("RISK_PER_TRADE_PCT_DEMO", 1.0))
RISK_PER_TRADE_PCT_REAL = float(os.getenv("RISK_PER_TRADE_PCT_REAL", 0.5))
PAPER_FEE_BPS = float(os.getenv("PAPER_FEE_BPS", 2.0))
PAPER_SPREAD_BPS = float(os.getenv("PAPER_SPREAD_BPS", 1.0))

ENABLE_REAL_TRADING = (os.getenv("ENABLE_REAL_TRADING", "false").strip().lower() in ("1", "true", "yes"))

YF_MULTI_TF = (os.getenv("YF_MULTI_TF", "false").strip().lower() in ("1", "true", "yes"))
YF_MIN_FETCH_INTERVAL_SECONDS = int(os.getenv("YF_MIN_FETCH_INTERVAL_SECONDS", 180))
YF_CACHE_TTL_SECONDS = int(os.getenv("YF_CACHE_TTL_SECONDS", 3600))

FOREX_SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "EURUSD=X,GBPUSD=X,USDJPY=X").split(",") if s.strip()]
CRYPTO_SYMBOLS = [s.strip() for s in os.getenv("CRYPTO_SYMBOLS", "").split(",") if s.strip()]
ALL_SYMBOLS = FOREX_SYMBOLS + CRYPTO_SYMBOLS

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

EXCHANGE_ID = os.getenv("EXCHANGE_ID", "bybit")
EXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY")
EXCHANGE_API_SECRET = os.getenv("EXCHANGE_API_SECRET")

MODEL_PATH = os.path.join("models", "voting_ensemble.pkl")
SCALER_PATH = os.path.join("models", "feature_scaler.pkl")
METADATA_PATH = os.path.join("models", "model_metadata.json")

DATA_DIR = os.getenv("TRADEBOT_DATA_DIR", "data")
PORTFOLIO_PATH = os.path.join(DATA_DIR, "portfolio_state.json")
TRADES_PATH = os.path.join(DATA_DIR, "trade_history.csv")

os.makedirs("models", exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================
# TELEGRAM NOTIFIER (Menu + Signals)
# ============================================

class MultiChannelNotifier:
    def __init__(self, token=None, group_id=None, admin_chat_id=None):
        # WhatsApp/Twilio intentionally not used (removed to prevent startup crashes)
        self.bot_token = token
        self.group_id = group_id
        self.admin_chat_id = admin_chat_id
        self.application = None
        self.bot = None

    async def initialize(self):
        if self.bot_token:
            try:
                request = HTTPXRequest(
                    connect_timeout=10,
                    read_timeout=20,
                    write_timeout=20,
                    pool_timeout=10,
                )

                self.application = Application.builder().token(self.bot_token).request(request).build()
                self.bot = self.application.bot

                # Handlers
                self.application.add_handler(CommandHandler("start", self._cmd_start))
                self.application.add_handler(CommandHandler("menu", self._cmd_start))
                self.application.add_handler(CallbackQueryHandler(self._handle_callbacks))
                self.application.add_error_handler(self._on_error)

                await self.application.initialize()
                await self.application.start()
                if self.application.updater:
                    await self.application.updater.start_polling()
                logger.info("✅ Telegram bot initialized with Main Menu")
            except Exception as e:
                logger.error(f"Telegram init error: {e}")

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        try:
            logger.error(f"Telegram error: {context.error}")
        except Exception:
            logger.error("Telegram error occurred")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [InlineKeyboardButton("📊 Portfolio Status", callback_data='status')],
            [InlineKeyboardButton("💰 Current Prices", callback_data='prices')],
            [InlineKeyboardButton("🔮 AI Forecast", callback_data='select_forecast')],
            [InlineKeyboardButton("🚀 Auto-Trade Controls", callback_data='trade_menu')],
            [InlineKeyboardButton("⚙️ Risk Settings", callback_data='risk_menu')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        text = "🤖 *AI Trading Bot Control Panel*\nSelect an option below:"
        if update.message:
            await update.message.reply_text(text, reply_markup=reply_markup, parse_mode='Markdown')
        else:
            await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode='Markdown')

    async def _handle_callbacks(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        try:
            await query.answer()
        except Exception as e:
            logger.warning(f"CallbackQuery answer failed: {e}")

        data = query.data
        cmd_path = os.path.join(DATA_DIR, "bot_command.json")

        if data == 'status':
            # Dynamic status fetching
            current_prices = {}
            for s in ALL_SYMBOLS:
                df = fetch_data(s)
                if df is not None: current_prices[s] = df['close'].iloc[-1]
            state = bot_instance.engine.get_state(current_prices) if 'bot_instance' in globals() and bot_instance else {'balance': 0, 'equity': 0, 'pnl': 0}
            
            status_text = f"📊 *Portfolio Status*\n\n"
            status_text += f"💰 Balance: ${state.get('balance', 0):.2f}\n"
            status_text += f"📈 Equity: ${state.get('equity', 0):.2f}\n"
            status_text += f"💹 PnL: {state.get('pnl', 0):+.2f}%\n"
            
            await query.edit_message_text(status_text, parse_mode='Markdown', 
                                          reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='main_menu')]]))

        elif data == 'prices':
            price_text = "💰 *Current Market Prices*\n\n"
            for s in ALL_SYMBOLS:
                df = fetch_data(s)
                if df is not None:
                    price_text += f"• `{s}`: {df['close'].iloc[-1]:.5f}\n"
            await query.edit_message_text(price_text, parse_mode='Markdown', 
                                          reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='main_menu')]]))

        elif data == 'select_forecast':
            keyboard = [[InlineKeyboardButton(f"🔮 {s}", callback_data=f"get_fc_{s}")] for s in ALL_SYMBOLS]
            keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data='main_menu')])
            await query.edit_message_text("🔮 *Select Asset for AI Forecast:*", 
                                          reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')

        elif data.startswith('get_fc_'):
            symbol = data.replace('get_fc_', '')
            await query.edit_message_text(f"⌛ Requesting AI analysis for {symbol}...", parse_mode='Markdown')
            # Trigger analysis logic
            if 'bot_instance' in globals() and bot_instance:
                await bot_instance.process_symbol(symbol)
            await query.edit_message_text(f"✅ AI Analysis for {symbol} triggered. Results will appear in the dashboard/logs.", 
                                          reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='select_forecast')]]), parse_mode='Markdown')

        elif data == 'trade_menu':
            keyboard = [
                [InlineKeyboardButton("🚀 START ALL PAIRS", callback_data='start_all')],
                [InlineKeyboardButton("🛑 STOP ALL PAIRS", callback_data='stop_all')],
                [InlineKeyboardButton("⚙️ Risk Settings", callback_data='risk_menu')],
                [InlineKeyboardButton("⬅️ Back", callback_data='main_menu')]
            ]
            await query.edit_message_text("⚙️ *Auto-Trade Controls*", 
                                          reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')

        elif data == 'start_all':
            with open(cmd_path, "w") as f:
                json.dump({"command": "start_all", "time": str(datetime.now())}, f)
            if 'bot_instance' in globals() and bot_instance:
                bot_instance.force_cycle = True
            await query.edit_message_text("🚀 Command sent: *START ALL*", 
                                          reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='trade_menu')]]), parse_mode='Markdown')

        elif data == 'stop_all':
            with open(cmd_path, "w") as f:
                json.dump({"command": "stop_all", "time": str(datetime.now())}, f)
            if 'bot_instance' in globals() and bot_instance:
                bot_instance.engine.positions = {}
            await query.edit_message_text("🛑 Command sent: *STOP ALL*", 
                                          reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='trade_menu')]]), parse_mode='Markdown')

        elif data == 'risk_menu':
            keyboard = [[InlineKeyboardButton("🌐 All Pairs", callback_data="risk_sel_ALL")]]
            keyboard += [[InlineKeyboardButton(f"{s}", callback_data=f"risk_sel_{s}")] for s in ALL_SYMBOLS]
            keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data='main_menu')])
            await query.edit_message_text("⚙️ *Select asset to change TP/SL:*", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')

        elif data.startswith('risk_sel_'):
            symbol = data.replace('risk_sel_', '', 1)
            keyboard = [
                [InlineKeyboardButton("🛡️ Safe (TP 2 / SL 1)", callback_data=f"risk_set_{symbol}_safe")],
                [InlineKeyboardButton("⚖️ Normal (TP 4 / SL 2)", callback_data=f"risk_set_{symbol}_normal")],
                [InlineKeyboardButton("🔥 Strong (TP 6 / SL 2.5)", callback_data=f"risk_set_{symbol}_strong")],
                [InlineKeyboardButton("⬅️ Back", callback_data='risk_menu')],
            ]
            await query.edit_message_text(f"⚙️ *Risk presets for:* `{symbol}`", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')

        elif data.startswith('risk_set_'):
            parts = data.split('_', 3)
            symbol = parts[2] if len(parts) > 2 else 'ALL'
            preset = parts[3] if len(parts) > 3 else 'normal'

            if preset == 'safe':
                tp, sl = 2.0, 1.0
            elif preset == 'strong':
                tp, sl = 6.0, 2.5
            else:
                tp, sl = 4.0, 2.0

            payload = {
                "command": "set_risk",
                "scope": "all" if symbol == "ALL" else "symbol",
                "symbol": None if symbol == "ALL" else symbol,
                "tp": tp,
                "sl": sl,
                "time": str(datetime.now()),
            }

            try:
                if 'bot_instance' in globals() and bot_instance and BOT_LOOP:
                    import asyncio as _asyncio
                    fut = _asyncio.run_coroutine_threadsafe(bot_instance.apply_command(payload), BOT_LOOP)
                    fut.result(timeout=5)
                else:
                    with open(cmd_path, "w") as f:
                        json.dump(payload, f)
            except Exception as e:
                logger.error(f"Risk preset apply error: {e}")

            scope_txt = "ALL" if symbol == "ALL" else symbol
            await query.edit_message_text(f"✅ Risk updated for *{scope_txt}*: TP={tp}% SL={sl}%", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='risk_menu')]]), parse_mode='Markdown')

        elif data == 'main_menu':
            await self._cmd_start(update, context)

    async def send_message(self, text, chat_id=None):
        """Send message to Telegram"""
        if not self.bot:
            return

        target_chat = chat_id or self.group_id or self.admin_chat_id
        if not target_chat:
            logger.warning("Telegram send skipped: TELEGRAM_GROUP_ID/TELEGRAM_CHAT_ID not set")
            return

        try:
            await self.bot.send_message(
                chat_id=target_chat,
                text=text,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    async def send_signal(self, signal_type, data):
        """Send a simplified signal"""
        if not self.bot:
            return

        target_chat = self.group_id or self.admin_chat_id
        if not target_chat:
            return

        if signal_type == "ENTRY":
            emoji = "🟢" if data.get('side') == 'long' else "🔴"
            tp_pct = data.get('tp_pct')
            sl_pct = data.get('sl_pct')
            strength = data.get('signal_strength')

            notional = data.get('notional')
            margin = data.get('margin')
            balance_before = data.get('balance_before')
            balance_after = data.get('balance_after')
            entry_price = data.get('entry_price')
            size = data.get('size')
            tp_price = data.get('take_profit')
            sl_price = data.get('stop_loss')

            lines = []
            if entry_price is not None and size is not None:
                try:
                    lines.append(f"*Entry:* {float(entry_price):.6f} | *Size:* {float(size):.6f}")
                except Exception:
                    pass

            if tp_price is not None and sl_price is not None:
                try:
                    lines.append(f"*TP/SL (price):* {float(tp_price):.6f} / {float(sl_price):.6f}")
                except Exception:
                    pass

            if tp_pct is not None and sl_pct is not None:
                try:
                    lines.append(f"*TP/SL (%):* {float(tp_pct):.2f}% / {float(sl_pct):.2f}%")
                except Exception:
                    pass

            if notional is not None or margin is not None:
                try:
                    n = float(notional) if notional is not None else None
                    m = float(margin) if margin is not None else None
                    if n is not None and m is not None:
                        lines.append(f"*Notional/Margin:* ${n:.2f} / ${m:.2f}")
                    elif n is not None:
                        lines.append(f"*Notional:* ${n:.2f}")
                    elif m is not None:
                        lines.append(f"*Margin:* ${m:.2f}")
                except Exception:
                    pass

            if balance_before is not None and balance_after is not None:
                try:
                    b0 = float(balance_before)
                    b1 = float(balance_after)
                    lines.append(f"*Balance:* ${b0:.2f} -> ${b1:.2f} (margin locked)")
                except Exception:
                    pass

            if strength:
                lines.append(f"*Signal Strength:* {strength}")

            details = "\n".join(lines)

            text = f"""
{emoji} *TRADE OPENED: {data['asset']}*

*Decision:* {data.get('trade_decision', 'NO')}
{details}

*Reason:* {data.get('analysis', '')}
"""
            await self.send_message(text, target_chat)

        elif signal_type == "CLOSE":
            emoji = "✅" if data.get('pnl', 0) > 0 else "❌"
            text = f"""
{emoji} *{signal_type} SIGNAL*

*Asset:* {data['asset']}
*Direction:* {data.get('side', 'N/A').upper()}
*Entry:* {data.get('entry_price', 0):.5f}
*Exit:* {data.get('exit_price', 0):.5f}
*P&L:* {data.get('pnl', 0):+.2f} USD ({data.get('pnl_percent', 0):+.2f}%)
*New Balance:* ${data.get('balance', 0):.2f}

*Exit Reason:* {data.get('reason', 'TP/SL hit')}
"""
            await self.send_message(text, target_chat)

    async def send_error(self, error_msg):
        """Send errors to admin (private message)"""
        if self.bot and self.admin_chat_id:
            text = f"⚠️ *ERROR*\n\n{error_msg[:500]}"
            await self.send_message(text, self.admin_chat_id)

    async def send_daily_summary(self, stats):
        """Send detailed report on portfolio and positions"""
        if not self.bot:
            return

        target_chat = self.group_id or self.admin_chat_id
        if not target_chat:
            return

        text = f"""
📊 *TRADING PORTFOLIO SUMMARY*

💰 Balance: ${stats.get('balance', 0):.2f}
📈 Equity: ${stats.get('equity', 0):.2f}
📊 PnL: {stats.get('pnl', 0):+.2f}%
🎯 Open Positions: {stats.get('open_positions', 0)}
✅ Trades Today: {stats.get('trades_today', 0)}

*Active Positions:*
"""
        positions_info = stats.get('positions_details', [])
        if not positions_info:
            text += "_No active positions._"
        else:
            for pos in positions_info:
                emoji = "🟢" if pos['side'] == 'long' else "🔴"
                text += f"""
{emoji} *{pos['symbol']}* ({pos['side'].upper()})
• Entry: {pos['entry_price']:.5f}
• Current: {pos['current_price']:.5f}
• P&L: {pos['pnl_percent']:+.2f}% (${pos['pnl_usd']:+.2f})
"""
        await self.send_message(text, target_chat)

    async def send_startup_message(self):
        """Send bot startup message"""
        if not self.bot:
            return

        target_chat = self.group_id or self.admin_chat_id
        if not target_chat:
            return

        text = f"""
🚀 *TRADING BOT STARTED*

*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
*Capital:* ${PAPER_CAPITAL:.2f}
*Max Risk:* {MAX_POSITION_SIZE}%
*Stop Loss:* {STOP_LOSS_PERCENT}%
*Take Profit:* {TAKE_PROFIT_PERCENT}%

*Assets:* {', '.join(ALL_SYMBOLS)}
"""
        await self.send_message(text, target_chat)


# ============================================
# ML MODEL LOADER (using optimized models)
# ============================================
class MLModelLoader:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH, metadata_path=METADATA_PATH):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.feature_cols = self.metadata.get('feature_columns', [])
            raw_map = self.metadata.get('target_mapping', {0: 'bearish', 1: 'flat', 2: 'bullish'})
            norm_map: dict[int, str] = {}
            if isinstance(raw_map, dict):
                for k, v in raw_map.items():
                    try:
                        ki = int(float(k))
                    except Exception:
                        continue
                    if v is None:
                        continue
                    vv = str(v).strip()
                    if not vv:
                        continue
                    norm_map[ki] = vv
            self.target_mapping = norm_map or {0: 'bearish', 1: 'flat', 2: 'bullish'}
        except Exception as e:
            logger.warning(f"MLModelLoader metadata load failed: {e}")
            self.feature_cols = []
            self.target_mapping = {0: 'bearish', 1: 'flat', 2: 'bullish'}

    def predict(self, df):
        if len(self.feature_cols) == 0 or len(df) == 0:
            return None

        # Extract features with case-insensitive matching
        features = pd.DataFrame(index=df.index)
        df_cols_lower = {c.lower(): c for c in df.columns}
        
        missing_cols = []
        for col in self.feature_cols:
            col_lower = col.lower()
            if col_lower in df_cols_lower:
                features[col] = df[df_cols_lower[col_lower]]
            else:
                features[col] = 0
                missing_cols.append(col)
        
        if missing_cols and len(missing_cols) < len(self.feature_cols):
            logger.debug(f"⚠️ Some features missing, filled with 0: {missing_cols[:5]}...")
        elif len(missing_cols) == len(self.feature_cols):
            logger.error("❌ ALL features missing! Check indicator names vs metadata.")

        features = features.dropna()
        if len(features) == 0:
            return None

        X = self.scaler.transform(features)
        proba = self.model.predict_proba(X)
        pred = np.argmax(proba, axis=1)

        return {
            'regime': int(pred[-1]),
            'regime_name': self.target_mapping.get(int(pred[-1]), 'unknown'),
            'confidence': float(max(proba[-1])),
            'probabilities': {
                'bearish': float(proba[-1][0]),
                'flat': float(proba[-1][1]),
                'bullish': float(proba[-1][2])
            }
        }


# ============================================
# INDICATORS CALCULATION
# ============================================
def _haar_trend_pct(arr: np.ndarray, levels: int) -> float:
    x = np.asarray(arr, dtype=float)
    if x.size == 0 or np.any(~np.isfinite(x)):
        return float("nan")

    L = int(levels)
    if L < 1:
        return 0.0

    min_n = 2 ** L
    n = (x.size // min_n) * min_n
    if n < min_n:
        return float("nan")

    a = x[-n:]
    for _ in range(L):
        a = (a[0::2] + a[1::2]) / np.sqrt(2.0)

    base = float(a[0])
    if not np.isfinite(base):
        return float("nan")

    return float((float(a[-1]) - base) / (abs(base) + 1e-12) * 100.0)


def _haar_energy(arr: np.ndarray, levels: int) -> float:
    x = np.asarray(arr, dtype=float)
    if x.size == 0 or np.any(~np.isfinite(x)):
        return float("nan")

    L = int(levels)
    if L < 1:
        return 0.0

    min_n = 2 ** L
    n = (x.size // min_n) * min_n
    if n < min_n:
        return float("nan")

    a = x[-n:]
    total = 0.0
    for _ in range(L):
        d = (a[0::2] - a[1::2]) / np.sqrt(2.0)
        a = (a[0::2] + a[1::2]) / np.sqrt(2.0)
        total += float(np.mean(d * d))

    return float(total)


def calculate_indicators(df):
    """Calculate technical indicators used by the bot."""
    df = df.copy()

    # Returns
    df['returns'] = df['close'].pct_change() * 100
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['returns'].rolling(20).std()

    # EMA
    df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # MACD (12, 26, 9)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI (Wilder)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR (Wilder)
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        (df['high'] - df['low']).abs(),
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()

    # Bollinger Bands (20, 2)
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_middle'] = bb_mid
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std

    # Volume
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # EMA Cross
    df['ema_cross'] = 0
    df.loc[(df['ema_8'] > df['ema_21']) & (df['ema_8'].shift(1) <= df['ema_21'].shift(1)), 'ema_cross'] = 1
    df.loc[(df['ema_8'] < df['ema_21']) & (df['ema_8'].shift(1) >= df['ema_21'].shift(1)), 'ema_cross'] = -1

    # (Alligator/Fractals removed to reduce constraints and simplify strategy)

    # Wave (Haar) features
    if int(WAVE_WINDOW) >= 8:
        df['wave_trend'] = df['close'].rolling(int(WAVE_WINDOW)).apply(lambda a: _haar_trend_pct(a, int(WAVE_LEVELS)), raw=True)
        df['wave_energy'] = df['close'].rolling(int(WAVE_WINDOW)).apply(lambda a: _haar_energy(a, int(WAVE_LEVELS)), raw=True)

    # Lags
    for lag in [1, 2, 3, 5]:
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        df[f'close_lag_{lag}'] = df['close'].shift(lag)

    return df


# ============================================
# DEEPSEEK ADVISOR (following project contract)
# ============================================
class AIAdvisor:
    def __init__(self, api_key=None):
        self.client = openai.OpenAI(
            api_key=(api_key or AI_API_KEY),
            base_url="https://openrouter.ai/api/v1"
        ) if (api_key or AI_API_KEY) else None
        self.model = "deepseek/deepseek-chat"
        self.max_tokens = int(AI_MAX_TOKENS)
        self.disabled_until_ts = 0.0
        self.last_auth_error_log_ts = 0.0

    def _is_disabled(self) -> bool:
        try:
            return float(self.disabled_until_ts) > time.time()
        except Exception:
            return False

    def get_decision(self, symbol, df, ml_prediction, context: dict | None = None):
        if not self.client or self._is_disabled():
            return {
                'trade_decision': 'NO',
                'action': 'hold',
                'signal_strength': 'weak',
                'reasoning_short': 'AI unavailable (auth/disabled)',
                'ai_error_code': 401
            }

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        recent = df[['open', 'high', 'low', 'close']].tail(6).round(6).to_dict('records')

        rsi_v = float(latest.get('rsi', 0.0) or 0.0)
        macd_h = float(latest.get('macd_hist', 0.0) or 0.0)
        ema_fast = float(latest.get('ema_8', 0.0) or 0.0)
        ema_slow = float(latest.get('ema_21', 0.0) or 0.0)

        ctx = context or {}
        macro = ctx.get("macro") if isinstance(ctx, dict) else None
        poly = ctx.get("polymarket") if isinstance(ctx, dict) else None

        macro_lines = []
        if isinstance(macro, dict) and macro:
            for k, v in list(macro.items())[:8]:
                if not isinstance(v, dict):
                    continue
                try:
                    wt = float(v.get("wave_trend_pct") or 0.0)
                except Exception:
                    wt = 0.0
                try:
                    r1 = float(v.get("ret_1") or 0.0)
                except Exception:
                    r1 = 0.0
                macro_lines.append(f"{k}: ret1={r1:+.2f}%, wave={wt:+.2f}%")

        poly_lines = []
        if isinstance(poly, list) and poly:
            for m in poly[:5]:
                if not isinstance(m, dict):
                    continue
                q = str(m.get("question") or m.get("title") or "").strip()
                p = m.get("prob")
                try:
                    p = float(p)
                except Exception:
                    p = None
                if q and p is not None:
                    poly_lines.append(f"{q} => {p:.1%}")

        prompt = f"""
You are a binary trading decision engine for {symbol}.

INPUTS:
- Timeframe: 1h (primary)
- ML Regime: {ml_prediction['regime_name'].upper()}
- ML Confidence: {ml_prediction['confidence']:.1%}
- EMA8/EMA21: {ema_fast:.6f} / {ema_slow:.6f}
- RSI(14): {rsi_v:.2f}
- MACD hist: {macd_h:.6f}
- Wave trend (Haar, %): {float(latest.get('wave_trend', 0.0)):.3f}
- Wave energy (Haar): {float(latest.get('wave_energy', 0.0)):.6f}
- Macro/commodities snapshot: {"; ".join(macro_lines) if macro_lines else "N/A"}
- Polymarket snapshot: {"; ".join(poly_lines) if poly_lines else "N/A"}
- Lessons (must follow): {"; ".join([str(x) for x in (ctx.get('lessons') or [])][:8]) if isinstance(ctx, dict) else "N/A"}
- Last candles (JSON): {recent}

TASK:
Decide if we should open a position RIGHT NOW.
Use trend/momentum (EMA, RSI, MACD) plus wave trend as context; ML regime is supportive.

Respond ONLY with valid JSON (no extra text):
{{
  "trade_decision": "YES" or "NO",
  "signal_strength": "weak" or "medium" or "strong",
  "tp_pct": 0,
  "sl_pct": 0,
  "reasoning_short": "Max 1 sentence in English",
  "ml_forecast_pct": "{ml_prediction['confidence']:.1%}",
  "action": "entry" or "hold",
  "side": "long" or "short"
}}
Rules:
- tp_pct/sl_pct are percentages relative to entry price.
- If signal_strength is weak, use more conservative risk: smaller tp_pct and tighter sl_pct.
- If signal_strength is strong, tp_pct can be larger; keep sl_pct reasonable.
"""

        try:
            system = (
                "You are a strict trading decision function. "
                "Rules: (1) Use ONLY the provided INPUTS; do NOT invent news, events, prices, probabilities, or sources. "
                "(2) If information is missing, respond with trade_decision=NO and a short reasoning_short. "
                "(3) Output MUST be valid JSON ONLY, no markdown, no extra text. "
                "(4) Allowed values: trade_decision YES/NO; action entry/hold; side long/short; signal_strength weak/medium/strong."
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=int(self.max_tokens)
            )
            content = response.choices[0].message.content or ""

            import re
            match = re.search(r'\{[\s\S]*\}', content)
            if not match:
                return {'trade_decision': 'NO', 'action': 'hold', 'signal_strength': 'weak', 'reasoning_short': 'Parse error'}

            data = json.loads(match.group())
            if not isinstance(data, dict):
                return {'trade_decision': 'NO', 'action': 'hold', 'signal_strength': 'weak', 'reasoning_short': 'Invalid JSON'}

            td = str(data.get('trade_decision') or 'NO').upper()
            act = str(data.get('action') or 'hold').lower()
            side = str(data.get('side') or 'long').lower()
            strength = str(data.get('signal_strength') or 'weak').lower()

            if td not in ('YES', 'NO'):
                td = 'NO'
            if act not in ('entry', 'hold'):
                act = 'hold'
            if side not in ('long', 'short'):
                side = 'long'
            if strength not in ('weak', 'medium', 'strong'):
                strength = 'weak'

            data['trade_decision'] = td
            data['action'] = act
            data['side'] = side
            data['signal_strength'] = strength
            return data
        except Exception as e:
            error_msg = str(e)
            if "402" in error_msg:
                self.max_tokens = max(int(AI_MIN_TOKENS_ON_402), 1)
                logger.error(f"❌ AI credits exhausted (402). Reduce AI_MAX_TOKENS or add credits. {error_msg}")
                return {
                    'trade_decision': 'NO',
                    'action': 'hold',
                    'signal_strength': 'weak',
                    'reasoning_short': 'AI credits exhausted',
                    'ai_error_code': 402
                }
            if "401" in error_msg or "Unauthorized" in error_msg:
                self.disabled_until_ts = time.time() + 3600
                if (time.time() - float(self.last_auth_error_log_ts or 0.0)) > 300:
                    self.last_auth_error_log_ts = time.time()
                    logger.error("❌ API Key Error: Please check your OPENROUTER_API_KEY")
                return {
                    'trade_decision': 'NO',
                    'action': 'hold',
                    'signal_strength': 'weak',
                    'reasoning_short': 'AI API key error',
                    'ai_error_code': 401
                }
            logger.error(f"AI error: {e}")
            return {
                'trade_decision': 'NO',
                'action': 'hold',
                'signal_strength': 'weak',
                'reasoning_short': 'AI API error',
                'ai_error_code': 500
            }

    def review_trade(self, trade: dict, context: dict | None = None) -> dict:
        if not self.client:
            return {"lesson": "No AI key", "tags": ["ai_off"], "severity": 1, "action_items": []}

        t = trade or {}
        ctx = context or {}
        prompt = f"""
You are a strict post-trade reviewer.
Rules:
- Use ONLY provided TRADE + CONTEXT; do not invent news/events/metrics.
- Output MUST be valid JSON only (no markdown, no extra text).

TRADE:
- asset: {t.get('asset')}
- side: {t.get('side')}
- entry_price: {t.get('entry_price')}
- exit_price: {t.get('exit_price')}
- pnl: {t.get('pnl')}
- pnl_percent: {t.get('pnl_percent')}
- strategy_mode: {t.get('strategy_mode')}
- signal_strength: {t.get('signal_strength')}
- reasoning: {t.get('analysis')}
- wave_trend: {t.get('wave_trend')}
- dxy_trend: {t.get('dxy_trend')}

CONTEXT:
- filters: {ctx.get('filters')}
- macro: {ctx.get('macro')}

Return JSON:
{{
  \"lesson\": \"one short sentence\",
  \"tags\": [\"tag1\",\"tag2\"],
  \"severity\": 1,
  \"action_items\": [\"one short item\"]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You output strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=int(REVIEW_MAX_TOKENS),
            )
            content = response.choices[0].message.content or ""
            import re
            match = re.search(r"\{[\s\S]*\}", content)
            if not match:
                return {"lesson": "Parse error", "tags": ["parse"], "severity": 1, "action_items": []}
            data = json.loads(match.group())
            if not isinstance(data, dict):
                return {"lesson": "Invalid JSON", "tags": ["parse"], "severity": 1, "action_items": []}

            lesson = str(data.get("lesson") or "").strip()[:240] or "No lesson"
            tags = data.get("tags")
            if not isinstance(tags, list):
                tags = []
            tags = [str(x).strip()[:32] for x in tags[:6] if str(x).strip()]
            severity = int(data.get("severity") or 1)
            severity = max(1, min(5, severity))
            items = data.get("action_items")
            if not isinstance(items, list):
                items = []
            items = [str(x).strip()[:120] for x in items[:5] if str(x).strip()]
            return {"lesson": lesson, "tags": tags, "severity": severity, "action_items": items}
        except Exception as e:
            if "402" in str(e):
                return {"lesson": "AI credits exhausted", "tags": ["ai_402"], "severity": 1, "action_items": []}
            return {"lesson": "AI review error", "tags": ["ai_error"], "severity": 1, "action_items": []}

    def get_reinforse_decision(self, symbol, df, ml_prediction, context: dict | None = None):
        if not self.client or self._is_disabled():
            return {
                'trade_decision': 'NO',
                'action': 'hold',
                'signal_strength': 'weak',
                'reasoning_short': 'AI unavailable (auth/disabled)',
                'ai_error_code': 401
            }

        latest = df.iloc[-1]
        recent = df[['open', 'high', 'low', 'close']].tail(12).round(6).to_dict('records')

        try:
            bb_pos = float((latest.get('close', 0) - latest.get('bb_lower', 0)) / (latest.get('bb_upper', 1) - latest.get('bb_lower', 0)))
        except Exception:
            bb_pos = 0.0

        ctx = context or {}
        macro = ctx.get("macro") if isinstance(ctx, dict) else None
        poly = ctx.get("polymarket") if isinstance(ctx, dict) else None

        macro_lines = []
        if isinstance(macro, dict) and macro:
            for k, v in list(macro.items())[:8]:
                if not isinstance(v, dict):
                    continue
                try:
                    wt = float(v.get("wave_trend_pct") or 0.0)
                except Exception:
                    wt = 0.0
                try:
                    r1 = float(v.get("ret_1") or 0.0)
                except Exception:
                    r1 = 0.0
                macro_lines.append(f"{k}: ret1={r1:+.2f}%, wave={wt:+.2f}%")

        poly_lines = []
        if isinstance(poly, list) and poly:
            for m in poly[:5]:
                if not isinstance(m, dict):
                    continue
                q = str(m.get("question") or m.get("title") or "").strip()
                p = m.get("prob")
                try:
                    p = float(p)
                except Exception:
                    p = None
                if q and p is not None:
                    poly_lines.append(f"{q} => {p:.1%}")

        prompt = f"""
You are a trading strategy engine for {symbol}.

INPUTS (latest bar):
- ML Regime: {ml_prediction['regime_name'].upper()} | Confidence: {ml_prediction['confidence']:.1%}
- RSI(14): {float(latest.get('rsi', 0)):.2f}
- MACD hist: {float(latest.get('macd_hist', 0)):.6f}
- Bollinger position (0=lower..1=upper): {bb_pos:.3f}
- ATR(14): {float(latest.get('atr', 0)):.6f}
- Wave trend (Haar, %): {float(latest.get('wave_trend', 0.0)):.3f}
- Wave energy (Haar): {float(latest.get('wave_energy', 0.0)):.6f}
- Macro/commodities snapshot: {"; ".join(macro_lines) if macro_lines else "N/A"}
- Polymarket snapshot: {"; ".join(poly_lines) if poly_lines else "N/A"}
- Lessons (must follow): {"; ".join([str(x) for x in (ctx.get('lessons') or [])][:8]) if isinstance(ctx, dict) else "N/A"}
- Last candles (JSON): {recent}

TASK:
Decide whether to TRADE now and propose conservative/normal/aggressive risk based on signal strength.

Respond ONLY with valid JSON (no extra text):
{{
  "trade_decision": "YES" or "NO",
  "signal_strength": "weak" or "medium" or "strong",
  "tp_pct": 0,
  "sl_pct": 0,
  "reasoning_short": "Max 1 sentence in English",
  "ml_forecast_pct": "{ml_prediction['confidence']:.1%}",
  "action": "entry" or "hold",
  "side": "long" or "short"
}}
"""

        try:
            system = (
                "You are a strict trading decision function. "
                "Rules: (1) Use ONLY the provided INPUTS; do NOT invent news, events, prices, probabilities, or sources. "
                "(2) If information is missing, respond with trade_decision=NO and a short reasoning_short. "
                "(3) Output MUST be valid JSON ONLY, no markdown, no extra text. "
                "(4) Allowed values: trade_decision YES/NO; action entry/hold; side long/short; signal_strength weak/medium/strong."
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=int(self.max_tokens),
            )
            content = response.choices[0].message.content or ""

            import re
            match = re.search(r'\{[\s\S]*\}', content)
            if not match:
                return {'trade_decision': 'NO', 'action': 'hold', 'signal_strength': 'weak', 'reasoning_short': 'Parse error'}

            data = json.loads(match.group())
            if not isinstance(data, dict):
                return {'trade_decision': 'NO', 'action': 'hold', 'signal_strength': 'weak', 'reasoning_short': 'Invalid JSON'}

            td = str(data.get('trade_decision') or 'NO').upper()
            act = str(data.get('action') or 'hold').lower()
            side = str(data.get('side') or 'long').lower()
            strength = str(data.get('signal_strength') or 'weak').lower()

            if td not in ('YES', 'NO'):
                td = 'NO'
            if act not in ('entry', 'hold'):
                act = 'hold'
            if side not in ('long', 'short'):
                side = 'long'
            if strength not in ('weak', 'medium', 'strong'):
                strength = 'weak'

            data['trade_decision'] = td
            data['action'] = act
            data['side'] = side
            data['signal_strength'] = strength
            return data
        except Exception as e:
            error_msg = str(e)
            if "402" in error_msg:
                self.max_tokens = max(int(AI_MIN_TOKENS_ON_402), 1)
                logger.error(f"❌ AI credits exhausted (402). Reduce AI_MAX_TOKENS or add credits. {error_msg}")
                return {
                    'trade_decision': 'NO',
                    'action': 'hold',
                    'signal_strength': 'weak',
                    'reasoning_short': 'AI credits exhausted',
                    'ai_error_code': 402
                }
            return {
                'trade_decision': 'NO',
                'action': 'hold',
                'signal_strength': 'weak',
                'reasoning_short': 'AI API error',
                'ai_error_code': 500
            }

    def review_trade(self, trade: dict, context: dict | None = None) -> dict:
        if not self.client:
            return {"lesson": "AI API key missing", "tags": ["ai_off"], "severity": 1, "action_items": []}

        t = trade or {}
        ctx = context or {}

        prompt = f"""
You are a strict post-trade reviewer.
Rules:
- Use ONLY provided TRADE and CONTEXT. Do NOT invent news, sources, or prices.
- Output MUST be valid JSON only.

TRADE:
- asset: {t.get('asset')}
- side: {t.get('side')}
- entry_price: {t.get('entry_price')}
- exit_price: {t.get('exit_price')}
- pnl: {t.get('pnl')}
- pnl_percent: {t.get('pnl_percent')}
- strategy_mode: {t.get('strategy_mode')}
- signal_strength: {t.get('signal_strength')}
- analysis: {t.get('analysis')}
- wave_trend: {t.get('wave_trend')}
- dxy_trend: {t.get('dxy_trend')}

CONTEXT:
- filters: {ctx.get('filters')}
- macro: {ctx.get('macro')}

Return JSON:
{{
  "lesson": "one short sentence",
  "tags": ["tag1", "tag2"],
  "severity": 1,
  "action_items": ["one short item"]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return strict JSON only. Never hallucinate."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=int(REVIEW_MAX_TOKENS),
            )
            content = response.choices[0].message.content or ""

            import re
            match = re.search(r'\{[\s\S]*\}', content)
            if not match:
                return {"lesson": "Parse error", "tags": ["parse"], "severity": 1, "action_items": []}

            data = json.loads(match.group())
            if not isinstance(data, dict):
                return {"lesson": "Invalid JSON", "tags": ["parse"], "severity": 1, "action_items": []}

            lesson = str(data.get('lesson') or '').strip()[:240] or "No lesson"
            tags = data.get('tags')
            if not isinstance(tags, list):
                tags = []
            tags = [str(x).strip()[:32] for x in tags[:6] if str(x).strip()]
            severity = int(data.get('severity') or 1)
            severity = max(1, min(5, severity))
            items = data.get('action_items')
            if not isinstance(items, list):
                items = []
            items = [str(x).strip()[:120] for x in items[:5] if str(x).strip()]

            return {"lesson": lesson, "tags": tags, "severity": severity, "action_items": items}
        except Exception as e:
            if "402" in str(e):
                return {"lesson": "AI credits exhausted", "tags": ["ai_402"], "severity": 1, "action_items": []}
            return {"lesson": "AI review error", "tags": ["ai_error"], "severity": 1, "action_items": []}


# ============================================
# DATA FETCHER (multi-timeframe)
# ============================================
_YF_DATA_CACHE = {}


def fetch_data(symbol):
    now = time.time()
    cached = _YF_DATA_CACHE.get(symbol)
    if cached is not None:
        age = now - float(cached.get("ts") or 0.0)
        if age < float(YF_MIN_FETCH_INTERVAL_SECONDS):
            return cached.get("df")

    try:
        from yfinance.exceptions import YFRateLimitError
    except Exception:
        YFRateLimitError = None

    try:
        ticker = yf.Ticker(symbol)

        if bool(YF_MULTI_TF):
            df_1h = ticker.history(period="1mo", interval="1h")
            if df_1h is None or df_1h.empty:
                df_4h = ticker.history(period="3mo", interval="4h")
            else:
                df_4h = None
        else:
            df_1h = ticker.history(period="3mo", interval="1h")
            df_4h = None

        df = df_1h if df_1h is not None and not df_1h.empty else df_4h
        if df is None or df.empty:
            df = ticker.history(period="3mo", interval="1d")

        if df is None or df.empty:
            if cached is not None and (now - float(cached.get("ts") or 0.0)) < float(YF_CACHE_TTL_SECONDS):
                logger.warning(f"⚠️ {symbol}: yfinance empty, using cached data")
                return cached.get("df")
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["open", "high", "low", "close", "volume"]

        out = calculate_indicators(df)
        _YF_DATA_CACHE[symbol] = {"ts": now, "df": out}
        return out
    except Exception as e:
        if YFRateLimitError is not None and isinstance(e, YFRateLimitError):
            if cached is not None and (now - float(cached.get("ts") or 0.0)) < float(YF_CACHE_TTL_SECONDS):
                logger.warning(f"⚠️ {symbol}: yfinance rate limited, using cached data")
                return cached.get("df")
            logger.warning(f"⚠️ {symbol}: yfinance rate limited")
            return None

        if cached is not None and (now - float(cached.get("ts") or 0.0)) < float(YF_CACHE_TTL_SECONDS):
            logger.warning(f"⚠️ {symbol}: yfinance error, using cached data: {e}")
            return cached.get("df")
        raise


def _safe_fetch_json(url: str, allowed_hosts: set[str]):
    try:
        from urllib.parse import urlparse
        from urllib.request import Request, urlopen

        p = urlparse(url)
        host = (p.hostname or "").lower()
        if not host or host not in {h.lower() for h in allowed_hosts}:
            return None

        req = Request(url, headers={"User-Agent": "ai-trader/1.0"})
        with urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        return json.loads(raw)
    except Exception:
        return None


_MACRO_CACHE = {"ts": 0.0, "data": {}}


def fetch_macro_snapshot():
    now = time.time()
    try:
        age = now - float(_MACRO_CACHE.get("ts") or 0.0)
        if age < float(YF_MIN_FETCH_INTERVAL_SECONDS):
            data = _MACRO_CACHE.get("data")
            if isinstance(data, dict) and data:
                return data
    except Exception:
        pass

    out = {}
    for sym in MACRO_SYMBOLS:
        try:
            t = yf.Ticker(sym)
            df = t.history(period="14d", interval="1h")
            if df is None or df.empty:
                df = t.history(period="180d", interval="1d")
            if df is None or df.empty:
                continue

            close = df["Close"].astype(float).dropna()
            if close.empty:
                continue

            ret_1 = float(close.pct_change().iloc[-1] * 100.0) if len(close) >= 2 else 0.0
            ret_24 = float(close.pct_change(24).iloc[-1] * 100.0) if len(close) >= 25 else 0.0
            wave_trend_pct = _haar_trend_pct(close.tail(int(WAVE_WINDOW)).to_numpy(), int(WAVE_LEVELS))
            wave_energy = _haar_energy(close.tail(int(WAVE_WINDOW)).to_numpy(), int(WAVE_LEVELS))

            out[sym] = {
                "last": float(close.iloc[-1]),
                "ret_1": float(ret_1) if np.isfinite(ret_1) else 0.0,
                "ret_24": float(ret_24) if np.isfinite(ret_24) else 0.0,
                "wave_trend_pct": float(wave_trend_pct) if np.isfinite(wave_trend_pct) else 0.0,
                "wave_energy": float(wave_energy) if np.isfinite(wave_energy) else 0.0,
            }
        except Exception:
            continue

    try:
        _MACRO_CACHE["ts"] = now
        _MACRO_CACHE["data"] = out
    except Exception:
        pass

    return out


def fetch_polymarket_snapshot():
    url = POLYMARKET_FEED_URL
    if not url:
        return []

    data = _safe_fetch_json(url, allowed_hosts={"gamma-api.polymarket.com", "polymarket.com"})
    if data is None:
        return []

    markets = None
    if isinstance(data, list):
        markets = data
    elif isinstance(data, dict):
        if isinstance(data.get("markets"), list):
            markets = data.get("markets")
        elif isinstance(data.get("data"), list):
            markets = data.get("data")

    if not isinstance(markets, list):
        return []

    out = []
    for m in markets[:200]:
        if not isinstance(m, dict):
            continue
        q = str(m.get("question") or m.get("title") or m.get("name") or "").strip()
        if not q:
            continue

        prob = None
        for k in ("probability", "prob", "lastTradePrice", "last_trade_price"):
            if k in m:
                prob = m.get(k)
                break

        if prob is None and isinstance(m.get("outcomePrices"), list) and m.get("outcomePrices"):
            prob = m.get("outcomePrices")[0]

        try:
            prob = float(prob)
        except Exception:
            continue

        if prob > 1.0:
            prob = prob / 100.0

        if prob < 0.0 or prob > 1.0:
            continue

        out.append({"question": q, "prob": prob})

    return out[:10]


# ============================================
# REAL TRADING ENGINE (CCXT)
# ============================================
class RealTradingEngine:
    def __init__(self, exchange_id, api_key, secret):
        self.exchange = None
        self.enabled = bool(ENABLE_REAL_TRADING)
        self.positions = {}
        try:
            import ccxt
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True,
            })
            try:
                self.exchange.load_markets()
            except Exception:
                pass
            logger.info(f"✅ Real Trading initialized on {exchange_id} (enabled={self.enabled})")
        except Exception as e:
            logger.error(f"❌ Failed to init real trading: {e}")
            self.exchange = None

    def _to_exchange_symbol(self, symbol: str) -> str | None:
        s = str(symbol or '').strip()
        if not s:
            return None
        if '/' in s:
            return s
        if s.endswith('-USD'):
            base = s.replace('-USD', '').strip()
            if base:
                return f"{base}/USDT"
        return None

    def execute_entry(self, symbol, decision, price):
        if not self.exchange:
            return None
        if not self.enabled:
            logger.warning("Real trading disabled (set ENABLE_REAL_TRADING=true to allow orders)")
            return None

        ex_symbol = self._to_exchange_symbol(symbol)
        if not ex_symbol:
            logger.warning(f"Real mode: unsupported symbol {symbol}")
            return None

        try:
            side = str(decision.get('side', 'long') or 'long').lower()
            order_side = 'buy' if side == 'long' else 'sell'

            bal = self.exchange.fetch_balance() or {}
            usdt = float(((bal.get('free') or {}).get('USDT') or (bal.get('total') or {}).get('USDT') or 0.0))
            if usdt <= 0:
                return None

            try:
                risk_pct = float(getattr(decision, 'risk_per_trade_pct', None) or 0.0)
            except Exception:
                risk_pct = 0.0

            if risk_pct <= 0:
                risk_pct = float(RISK_PER_TRADE_PCT_REAL)

            sl = float(decision.get('stop_loss') or 0.0)
            if sl <= 0:
                return None

            unit_risk = abs(float(price) - float(sl))
            if unit_risk <= 0:
                return None

            risk_amount = float(usdt) * (float(risk_pct) / 100.0)
            amount = float(risk_amount) / float(unit_risk)

            if amount <= 0:
                return None

            order = self.exchange.create_order(ex_symbol, 'market', order_side, amount)

            self.positions[ex_symbol] = {
                'asset': symbol,
                'exchange_symbol': ex_symbol,
                'side': side,
                'entry_price': float(price),
                'size': float(amount),
                'stop_loss': float(decision.get('stop_loss') or 0.0),
                'take_profit': float(decision.get('take_profit') or 0.0),
                'order': order,
            }

            return {'side': side, 'entry_price': float(price), 'size': float(amount)}
        except Exception as e:
            logger.error(f"Order error: {e}")
            return None

    def check_exits(self, symbol, price):
        ex_symbol = self._to_exchange_symbol(symbol)
        if not ex_symbol:
            return None
        pos = self.positions.get(ex_symbol)
        if not isinstance(pos, dict):
            return None

        side = str(pos.get('side') or 'long')
        sl = float(pos.get('stop_loss') or 0.0)
        tp = float(pos.get('take_profit') or 0.0)

        hit = False
        reason = None
        if side == 'long':
            if sl > 0 and float(price) <= sl:
                hit = True
                reason = 'stop_loss'
            elif tp > 0 and float(price) >= tp:
                hit = True
                reason = 'take_profit'
        else:
            if sl > 0 and float(price) >= sl:
                hit = True
                reason = 'stop_loss'
            elif tp > 0 and float(price) <= tp:
                hit = True
                reason = 'take_profit'

        if not hit:
            return None

        if not self.enabled or not self.exchange:
            return None

        try:
            close_side = 'sell' if side == 'long' else 'buy'
            amount = float(pos.get('size') or 0.0)
            if amount <= 0:
                return None
            self.exchange.create_order(ex_symbol, 'market', close_side, amount)
        except Exception as e:
            logger.error(f"Close order error: {e}")
            return None

        out = {
            'asset': symbol,
            'side': side,
            'entry_price': float(pos.get('entry_price') or 0.0),
            'exit_price': float(price),
            'size': float(pos.get('size') or 0.0),
            'pnl': (float(price) - float(pos.get('entry_price') or 0.0)) * float(pos.get('size') or 0.0) if side == 'long' else (float(pos.get('entry_price') or 0.0) - float(price)) * float(pos.get('size') or 0.0),
            'exit_reason': reason,
        }

        try:
            del self.positions[ex_symbol]
        except Exception:
            pass

        return out

    def get_state(self, current_prices):
        if not self.exchange: return {'balance': 0, 'equity': 0, 'positions': 0, 'pnl': 0}
        try:
            bal = self.exchange.fetch_balance()
            total_usdt = bal['total'].get('USDT', 0)
            return {
                'balance': total_usdt,
                'equity': total_usdt,
                'positions': 0,
                'pnl': 0
            }
        except:
            return {'balance': 0, 'equity': 0, 'positions': 0, 'pnl': 0}

# ============================================
# PAPER TRADING ENGINE
# ============================================
class PaperTradingEngine:
    def __init__(self, initial_capital=PAPER_CAPITAL):
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.positions = {}
        self.trades = []
        self.daily_trades = 0
        self.loss_streak = 0

        self.max_trades_per_day = MAX_TRADES_PER_DAY
        self.daily_tp_target_percent = DAILY_TP_TARGET_PERCENT
        self.day_date = datetime.utcnow().date().isoformat()
        self.day_start_balance = initial_capital

        self._load_state()
        self._roll_day_if_needed()

    def _save_state(self):
        """Save portfolio state and trade history for the dashboard"""
        try:
            existing_meta = None
            if os.path.exists(PORTFOLIO_PATH):
                try:
                    with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
                        prev = json.load(f) or {}
                    if isinstance(prev, dict) and isinstance(prev.get("meta"), dict):
                        existing_meta = prev.get("meta")
                except Exception:
                    existing_meta = None

            serializable_positions = {}
            used_margin = 0.0
            unrealized_pnl = 0.0

            for s, p in self.positions.items():
                p_copy = dict(p or {})
                if 'entry_date' in p_copy:
                    p_copy['entry_date'] = str(p_copy['entry_date'])

                entry = float(p_copy.get('entry_price') or 0.0)
                size = float(p_copy.get('size') or 0.0)
                side = str(p_copy.get('side') or 'long')
                last_price = float(p_copy.get('last_price') or entry)

                margin = float(p_copy.get('margin') or (size * entry))
                used_margin += margin

                if side == 'short':
                    unrealized_pnl += (entry - last_price) * size
                else:
                    unrealized_pnl += (last_price - entry) * size

                p_copy['margin'] = margin
                p_copy['last_price'] = last_price
                p_copy['unrealized_pnl'] = (entry - last_price) * size if side == 'short' else (last_price - entry) * size

                serializable_positions[s] = p_copy

            equity = float(self.balance) + float(used_margin) + float(unrealized_pnl)

            state = {
                'balance': float(self.balance),
                'equity': float(equity),
                'used_margin': float(used_margin),
                'unrealized_pnl': float(unrealized_pnl),
                'positions': serializable_positions,
                'last_update': str(datetime.now()),
                'daily': {
                    'date': self.day_date,
                    'start_balance': float(self.day_start_balance),
                    'trades': int(self.daily_trades),
                    'max_trades_per_day': int(self.max_trades_per_day),
                    'daily_tp_target_percent': float(self.daily_tp_target_percent)
                }
            }
            meta = dict(existing_meta or {})
            meta['loss_streak'] = int(getattr(self, 'loss_streak', 0) or 0)
            state['meta'] = meta

            with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
            
            # Save trade history to CSV
            if self.trades:
                df = pd.DataFrame(self.trades)
                df.to_csv(TRADES_PATH, index=False)
                
            logger.info("💾 Saved portfolio state and trade history")
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def _load_state(self):
        """Load state if exists"""
        if os.path.exists(PORTFOLIO_PATH):
            try:
                with open(PORTFOLIO_PATH, "r") as f:
                    state = json.load(f)
                self.balance = state.get('balance', PAPER_CAPITAL)
                self.positions = state.get('positions', {})

                daily = state.get('daily') or {}
                self.day_date = str(daily.get('date') or self.day_date)
                self.day_start_balance = float(daily.get('start_balance') or self.balance)
                self.daily_trades = int(daily.get('trades') or self.daily_trades)

                meta = state.get('meta') if isinstance(state, dict) else None
                if isinstance(meta, dict):
                    self.loss_streak = int(meta.get('loss_streak') or self.loss_streak)
                self.max_trades_per_day = int(daily.get('max_trades_per_day') or self.max_trades_per_day)
                self.daily_tp_target_percent = float(daily.get('daily_tp_target_percent') or self.daily_tp_target_percent)
                # Restore datetime objects
                for s, p in self.positions.items():
                    if 'entry_date' in p: p['entry_date'] = datetime.fromisoformat(p['entry_date'])
                logger.info("📂 Loaded existing portfolio state")
            except Exception as e:
                logger.error(f"Error loading state: {e}")
        
        if os.path.exists(TRADES_PATH):
            try:
                df = pd.read_csv(TRADES_PATH)
                self.trades = df.to_dict('records')
                logger.info(f"📂 Loaded {len(self.trades)} historical trades")
            except Exception as e:
                logger.error(f"Error loading trade history: {e}")

    def _roll_day_if_needed(self):
        today = datetime.utcnow().date().isoformat()
        if self.day_date != today:
            self.day_date = today
            self.day_start_balance = float(self.balance)
            self.daily_trades = 0

    def reset_stats(self, reset_balance: bool = True, symbol: str | None = None):
        if symbol:
            if symbol in self.positions:
                del self.positions[symbol]
            if self.trades:
                self.trades = [t for t in self.trades if str(t.get('asset') or t.get('symbol') or '') != symbol]
        else:
            self.positions = {}
            self.trades = []
            self.daily_trades = 0
            self.day_date = datetime.utcnow().date().isoformat()
            if reset_balance:
                self.balance = float(self.initial_capital)
            self.day_start_balance = float(self.balance)

        try:
            if os.path.exists(TRADES_PATH):
                os.remove(TRADES_PATH)
        except Exception:
            pass

        self._save_state()

        try:
            if self.trades:
                df = pd.DataFrame(self.trades)
                df.to_csv(TRADES_PATH, index=False)
        except Exception as e:
            logger.error(f"Error saving trade history after reset: {e}")

    def _daily_profit_pct(self):
        if self.day_start_balance <= 0:
            return 0.0
        return (float(self.balance) - float(self.day_start_balance)) / float(self.day_start_balance) * 100.0

    def can_open_new_trade(self):
        self._roll_day_if_needed()
        if self.daily_trades >= int(self.max_trades_per_day):
            return False, "MAX_TRADES_PER_DAY"
        if self._daily_profit_pct() >= float(self.daily_tp_target_percent):
            return False, "DAILY_TP_TARGET_REACHED"
        return True, "OK"

    def execute_entry(self, symbol, decision, price):
        allowed, reason = self.can_open_new_trade()
        if not allowed:
            logger.info(f"⛔ {symbol}: Entry blocked ({reason}). TradesToday={self.daily_trades}/{self.max_trades_per_day}, DayPnL={self._daily_profit_pct():+.2f}%")
            return None

        if decision.get('action') != 'entry':
            logger.info(f"⏳ {symbol}: Action is not 'entry' ({decision.get('action')})")
            return None

        side = decision.get('side', 'long')
        stop = decision.get('stop_loss', price * (1 - STOP_LOSS_PERCENT/100 if side == 'long' else 1 + STOP_LOSS_PERCENT/100))
        target = decision.get('take_profit', price * (1 + TAKE_PROFIT_PERCENT/100 if side == 'long' else 1 - TAKE_PROFIT_PERCENT/100))
        confidence = float(decision.get('confidence', 50)) / 100

        strength = str(decision.get('signal_strength') or '').lower()
        mult = 1.0
        if strength == 'weak':
            mult = 0.6
        elif strength == 'strong':
            mult = 1.2

        try:
            risk_pct = float(getattr(self, 'risk_per_trade_pct', 1.0) or 0.0)
        except Exception:
            risk_pct = 1.0
        risk_pct = max(0.0, risk_pct)

        risk_amount = float(self.balance) * (risk_pct / 100.0) * float(mult)
        risk_per_unit = abs(float(price) - float(stop))

        if risk_per_unit <= 0:
            logger.warning(f"❌ {symbol}: Risk per unit is 0 (Price={price}, SL={stop})")
            return None

        size = float(risk_amount) / float(risk_per_unit)
        if size <= 0:
            logger.warning(f"❌ {symbol}: Calculated size is 0 (Risk={risk_amount}, UnitRisk={risk_per_unit})")
            return None

        try:
            lev = float(getattr(self, 'leverage', 1.0) or 1.0)
        except Exception:
            lev = 1.0
        lev = max(1.0, lev)

        notional = float(size) * float(price)
        margin = float(notional) / lev

        max_margin = float(self.balance) * 0.30
        if margin > max_margin and max_margin > 0:
            scale = max_margin / margin
            size *= scale
            notional = float(size) * float(price)
            margin = float(notional) / lev

        try:
            fee_bps = float(getattr(self, 'fee_bps', 0.0) or 0.0)
        except Exception:
            fee_bps = 0.0
        try:
            spread_bps = float(getattr(self, 'spread_bps', 0.0) or 0.0)
        except Exception:
            spread_bps = 0.0

        entry_cost = float(abs(notional)) * (max(0.0, fee_bps) + max(0.0, spread_bps)) / 10_000.0

        if margin + entry_cost > float(self.balance):
            logger.warning(f"❌ {symbol}: Insufficient balance (need margin+cost={margin+entry_cost:.2f}, bal={self.balance:.2f})")
            return None

        balance_before = float(self.balance)

        self.positions[symbol] = {
            'asset': symbol,
            'side': side,
            'entry_price': float(price),
            'stop_loss': float(stop),
            'take_profit': float(target),
            'size': float(size),
            'confidence': float(confidence),
            'entry_date': datetime.now(),
            'analysis': str(decision.get('reasoning_short', '') or ''),
            'signal_strength': str(decision.get('signal_strength', '') or ''),
            'tp_pct': float(decision.get('tp_pct', 0) or 0),
            'sl_pct': float(decision.get('sl_pct', 0) or 0),
            'ml_regime': str(decision.get('ml_regime', '') or ''),
            'ml_confidence': float(decision.get('ml_confidence', 0) or 0),
            'strategy_mode': str(decision.get('strategy_mode', '') or ''),
            'margin': float(margin),
            'notional': float(notional),
            'leverage': float(lev),
            'entry_cost': float(entry_cost),
            'last_price': float(price),
            'balance_before': float(balance_before),
        }

        self.balance -= float(margin) + float(entry_cost)
        self.positions[symbol]['balance_after'] = float(self.balance)
        self.daily_trades += 1

        logger.info(f"🚀 {symbol} ENTRY EXECUTED: {side.upper()} {size:.4f} @ {price:.5f} | TradesToday={self.daily_trades}/{self.max_trades_per_day} | DayPnL={self._daily_profit_pct():+.2f}%")
        self._save_state() # Save after entry
        return self.positions[symbol]

    def check_exits(self, symbol, price):
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        close_reason = None
        pnl = 0

        if pos['side'] == 'long':
            if price <= pos['stop_loss']:
                close_reason = 'stop_loss'
                pnl = (price - pos['entry_price']) * pos['size']
            elif price >= pos['take_profit']:
                close_reason = 'take_profit'
                pnl = (price - pos['entry_price']) * pos['size']
        else:
            if price >= pos['stop_loss']:
                close_reason = 'stop_loss'
                pnl = (pos['entry_price'] - price) * pos['size']
            elif price <= pos['take_profit']:
                close_reason = 'take_profit'
                pnl = (pos['entry_price'] - price) * pos['size']

        if close_reason:
            margin = float(pos.get('margin') or (pos['size'] * pos['entry_price']))

            try:
                fee_bps = float(getattr(self, 'fee_bps', 0.0) or 0.0)
            except Exception:
                fee_bps = 0.0
            try:
                spread_bps = float(getattr(self, 'spread_bps', 0.0) or 0.0)
            except Exception:
                spread_bps = 0.0

            exit_notional = float(abs(float(price) * float(pos.get('size') or 0.0)))
            exit_cost = float(exit_notional) * (max(0.0, fee_bps) + max(0.0, spread_bps)) / 10_000.0

            self.balance += float(margin) + float(pnl) - float(exit_cost)
            pos['exit_price'] = price
            pos['pnl'] = pnl
            pos['exit_cost'] = float(exit_cost)
            pos['pnl_percent'] = (pnl / margin) * 100 if margin > 0 else 0
            pos['exit_reason'] = close_reason
            pos['exit_date'] = str(datetime.now())

            self.trades.append(pos)

            try:
                if float(pos.get('pnl') or 0.0) < 0:
                    self.loss_streak = int(getattr(self, 'loss_streak', 0) or 0) + 1
                else:
                    self.loss_streak = 0
            except Exception:
                self.loss_streak = int(getattr(self, 'loss_streak', 0) or 0)

            del self.positions[symbol]
            
            self._save_state() # Save after exit
            return pos

        return None

    def get_state(self, current_prices):
        used_margin = 0.0
        unrealized_pnl = 0.0

        for symbol, pos in self.positions.items():
            try:
                entry = float(pos.get('entry_price') or 0.0)
                size = float(pos.get('size') or 0.0)
                side = str(pos.get('side') or 'long')
                price = float((current_prices or {}).get(symbol, pos.get('last_price') or entry) or entry)
                margin = float(pos.get('margin') or (size * entry))

                used_margin += margin
                if side == 'short':
                    unrealized_pnl += (entry - price) * size
                else:
                    unrealized_pnl += (price - entry) * size
            except Exception:
                continue

        equity = float(self.balance) + float(used_margin) + float(unrealized_pnl)
        start = float(self.initial_capital) if float(self.initial_capital) > 0 else float(PAPER_CAPITAL)

        return {
            'balance': float(self.balance),
            'equity': float(equity),
            'used_margin': float(used_margin),
            'unrealized_pnl': float(unrealized_pnl),
            'positions': len(self.positions),
            'pnl': (float(equity) - start) / start * 100 if start > 0 else 0.0
        }


# ============================================
# MAIN BOT
# ============================================
class TradingBot:
    def __init__(self):
        self.ml_loader = MLModelLoader()
        self.ai = AIAdvisor()

        self.cycle_seconds = DEMO_CYCLE_SECONDS if TRADING_MODE == "demo" else REAL_CYCLE_SECONDS
        
        # Select Engine based on mode
        if TRADING_MODE == "real" and EXCHANGE_API_KEY:
            self.engine = RealTradingEngine(EXCHANGE_ID, EXCHANGE_API_KEY, EXCHANGE_API_SECRET)
        else:
            self.engine = PaperTradingEngine()

        if isinstance(self.engine, PaperTradingEngine):
            self.engine.leverage = float(getattr(self.engine, "leverage", 1.0) or 1.0)
            self.engine.risk_per_trade_pct = float(getattr(self.engine, "risk_per_trade_pct", (RISK_PER_TRADE_PCT_REAL if TRADING_MODE == "real" else RISK_PER_TRADE_PCT_DEMO)))
            self.engine.fee_bps = float(getattr(self.engine, "fee_bps", PAPER_FEE_BPS))
            self.engine.spread_bps = float(getattr(self.engine, "spread_bps", PAPER_SPREAD_BPS))

        self.global_risk = {
            "tp_pct": float(TAKE_PROFIT_PERCENT),
            "sl_pct": float(STOP_LOSS_PERCENT),
        }
        self.symbol_risk = {s: {"tp_pct": float(self.global_risk["tp_pct"]), "sl_pct": float(self.global_risk["sl_pct"])} for s in ALL_SYMBOLS}

        self.strategy_mode = STRATEGY_MODE
        self.strategy = {
            "block_weak_signals": bool(BLOCK_WEAK_SIGNALS),
            "cooldown_bars": int(COOLDOWN_BARS),
            "use_atr_risk": bool(USE_ATR_RISK),
            "atr_sl_mult": float(ATR_SL_MULT),
            "atr_tp_mult": float(ATR_TP_MULT),
            "use_macro": bool(USE_MACRO_DEFAULT),
            "use_polymarket": bool(USE_POLYMARKET_DEFAULT),
            "use_wave_filter": bool(USE_WAVE_FILTER_DEFAULT),
            "wave_trend_block_pct": float(WAVE_TREND_BLOCK_PCT),
            "use_dxy_filter": bool(USE_DXY_FILTER_DEFAULT),
            "dxy_trend_block_pct": float(DXY_TREND_BLOCK_PCT),
            "risk_guard_enabled": bool(RISK_GUARD_ENABLED_DEFAULT),
            "max_open_positions": int(MAX_OPEN_POSITIONS_DEFAULT),
            "max_daily_drawdown_pct": float(MAX_DAILY_DRAWDOWN_PCT_DEFAULT),
            "max_loss_streak": int(MAX_LOSS_STREAK_DEFAULT),
            "guard_pause_seconds": int(GUARD_PAUSE_SECONDS_DEFAULT),
            "quality_mode": str(QUALITY_MODE_DEFAULT or "balanced"),
            "min_setup_score": int(MIN_SETUP_SCORE_DEFAULT),
            "use_session_filter": bool(USE_SESSION_FILTER_DEFAULT),
            "min_atr_pct": float(MIN_ATR_PCT_DEFAULT),
            "max_atr_pct": float(MAX_ATR_PCT_DEFAULT),
            "risk_per_trade_pct": float((RISK_PER_TRADE_PCT_REAL if TRADING_MODE == "real" else RISK_PER_TRADE_PCT_DEMO)),
            "paper_fee_bps": float(PAPER_FEE_BPS),
            "paper_spread_bps": float(PAPER_SPREAD_BPS),
        }
        self.context = {}
        self.last_action_ts = {}
        self.pause_until_ts = 0.0
        self.blocked = {"total": 0, "cooldown": 0, "weak": 0}
        self.blocked_by_symbol = {}

        # Multi-channel: group chat for signals, private for errors
        self.notifier = MultiChannelNotifier(
            token=TELEGRAM_TOKEN,
            group_id=os.getenv("TELEGRAM_GROUP_ID") or os.getenv("TELEGRAM_CHAT_ID"),
            admin_chat_id=os.getenv("TELEGRAM_CHAT_ID")
        )

        self.db_enabled = bool(DATABASE_URL)
        self._db_ready = False
        if self.db_enabled:
            self._db_ready = self._db_init()

        self.running = True

    def _db_connect(self):
        if not DATABASE_URL:
            return None
        try:
            import psycopg2
            return psycopg2.connect(DATABASE_URL)
        except Exception as e:
            logger.error(f"DB connect error: {e}")
            return None

    def _db_init(self) -> bool:
        conn = self._db_connect()
        if conn is None:
            return False
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_events (
                    id BIGSERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    asset TEXT,
                    side TEXT,
                    entry_price DOUBLE PRECISION,
                    exit_price DOUBLE PRECISION,
                    pnl DOUBLE PRECISION,
                    pnl_percent DOUBLE PRECISION,
                    strategy_mode TEXT,
                    signal_strength TEXT,
                    strategy_version TEXT,
                    wave_trend DOUBLE PRECISION,
                    dxy_trend DOUBLE PRECISION,
                    analysis TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_lessons (
                    id BIGSERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    trade_id BIGINT,
                    asset TEXT,
                    win BOOLEAN,
                    pnl DOUBLE PRECISION,
                    lesson TEXT,
                    tags TEXT,
                    severity INTEGER
                )
                """
            )
            conn.commit()
            return True
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            logger.error(f"DB init error: {e}")
            return False
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _db_insert_trade(self, trade: dict) -> int | None:
        if not self._db_ready:
            return None
        conn = self._db_connect()
        if conn is None:
            return None
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO trade_events (asset, side, entry_price, exit_price, pnl, pnl_percent, strategy_mode, signal_strength, strategy_version, wave_trend, dxy_trend, analysis)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
                """,
                (
                    trade.get('asset'),
                    trade.get('side'),
                    float(trade.get('entry_price') or 0.0),
                    float(trade.get('exit_price') or 0.0),
                    float(trade.get('pnl') or 0.0),
                    float(trade.get('pnl_percent') or 0.0),
                    trade.get('strategy_mode'),
                    trade.get('signal_strength'),
                    trade.get('strategy_version'),
                    float(trade.get('wave_trend') or 0.0),
                    float(trade.get('dxy_trend') or 0.0),
                    trade.get('analysis'),
                ),
            )
            trade_id = cur.fetchone()[0]
            conn.commit()
            return int(trade_id)
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            logger.error(f"DB insert trade error: {e}")
            return None
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _db_insert_lesson(self, trade_id: int | None, trade: dict, lesson: dict):
        if not self._db_ready:
            return
        conn = self._db_connect()
        if conn is None:
            return
        try:
            cur = conn.cursor()
            tags = lesson.get('tags')
            if isinstance(tags, list):
                tags = ",".join([str(x) for x in tags])
            cur.execute(
                """
                INSERT INTO trade_lessons (trade_id, asset, win, pnl, lesson, tags, severity)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    int(trade_id) if trade_id is not None else None,
                    trade.get('asset'),
                    bool((trade.get('pnl') or 0.0) > 0),
                    float(trade.get('pnl') or 0.0),
                    str(lesson.get('lesson') or '')[:400],
                    str(tags or '')[:200],
                    int(lesson.get('severity') or 1),
                ),
            )
            conn.commit()
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            logger.error(f"DB insert lesson error: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _load_lessons(self, limit: int = 8) -> list[str]:
        if not self._db_ready:
            return []
        conn = self._db_connect()
        if conn is None:
            return []
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT lesson FROM trade_lessons ORDER BY id DESC LIMIT %s",
                (int(limit),),
            )
            rows = cur.fetchall() or []
            out = []
            for r in rows:
                if r and r[0]:
                    out.append(str(r[0])[:200])
            return out
        except Exception:
            return []
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _review_and_persist_closed_trade(self, closed: dict, df, ctx: dict):
        if not isinstance(closed, dict):
            return

        trade = dict(closed)
        trade['asset'] = trade.get('asset') or trade.get('symbol')
        trade['analysis'] = trade.get('analysis') or trade.get('reasoning_short')

        try:
            trade['wave_trend'] = float(df.iloc[-1].get('wave_trend', 0.0))
        except Exception:
            trade['wave_trend'] = 0.0

        dxy_tr = 0.0
        try:
            macro = ctx.get('macro') if isinstance(ctx, dict) else None
            if isinstance(macro, dict):
                dxy_tr = float(macro.get('DX-Y.NYB', {}).get('wave_trend_pct', 0.0) or 0.0)
        except Exception:
            dxy_tr = 0.0
        trade['dxy_trend'] = dxy_tr

        trade_id = self._db_insert_trade(trade)

        if ENABLE_TRADE_REVIEW:
            review_ctx = {"filters": dict(self.strategy or {}), "macro": ctx.get('macro') if isinstance(ctx, dict) else None}
            lesson = self.ai.review_trade(trade, context=review_ctx)
            if isinstance(lesson, dict):
                self._db_insert_lesson(trade_id, trade, lesson)

    async def apply_command(self, cmd_data: dict):
        global TAKE_PROFIT_PERCENT, STOP_LOSS_PERCENT, MAX_POSITION_SIZE, TRADING_MODE

        cmd = (cmd_data or {}).get("command")
        if not cmd:
            return

        if cmd == "start_all":
            logger.info(f"🚀 Command: START ALL (TP={cmd_data.get('tp', TAKE_PROFIT_PERCENT)}, SL={cmd_data.get('sl', STOP_LOSS_PERCENT)}, Lev={cmd_data.get('leverage', 1)})")
            TAKE_PROFIT_PERCENT = float(cmd_data.get("tp", TAKE_PROFIT_PERCENT))
            STOP_LOSS_PERCENT = float(cmd_data.get("sl", STOP_LOSS_PERCENT))

            if isinstance(self.engine, PaperTradingEngine):
                lev = float(cmd_data.get("leverage", getattr(self.engine, "leverage", 1.0) or 1.0) or 1.0)
                self.engine.leverage = max(1.0, lev)

                if "risk_per_trade_pct" in (cmd_data or {}):
                    try:
                        self.engine.risk_per_trade_pct = float(cmd_data.get("risk_per_trade_pct"))
                    except Exception:
                        pass

                if "paper_fee_bps" in (cmd_data or {}):
                    try:
                        self.engine.fee_bps = float(cmd_data.get("paper_fee_bps"))
                    except Exception:
                        pass

                if "paper_spread_bps" in (cmd_data or {}):
                    try:
                        self.engine.spread_bps = float(cmd_data.get("paper_spread_bps"))
                    except Exception:
                        pass

                self.engine.max_trades_per_day = int(cmd_data.get("max_trades_per_day", self.engine.max_trades_per_day))
                self.engine.daily_tp_target_percent = float(cmd_data.get("daily_tp_target_percent", self.engine.daily_tp_target_percent))
                self.engine._roll_day_if_needed()
                logger.info(f"✅ Demo limits: max_trades_per_day={self.engine.max_trades_per_day}, daily_tp_target_percent={self.engine.daily_tp_target_percent}, leverage={self.engine.leverage}")

            self.force_cycle = True
            logger.info("⚡ Force-cycle set")
            return

        if cmd == "set_risk":
            scope = (cmd_data.get("scope") or "all").lower()
            symbol = (cmd_data.get("symbol") or "").strip()

            tp = cmd_data.get("tp")
            sl = cmd_data.get("sl")

            def _as_float(v, fallback):
                try:
                    if v is None:
                        return float(fallback)
                    return float(v)
                except Exception:
                    return float(fallback)

            if scope == "all" or not symbol:
                self.global_risk["tp_pct"] = _as_float(tp, self.global_risk["tp_pct"])
                self.global_risk["sl_pct"] = _as_float(sl, self.global_risk["sl_pct"])
                for s in ALL_SYMBOLS:
                    self.symbol_risk.setdefault(s, {})
                    self.symbol_risk[s]["tp_pct"] = float(self.global_risk["tp_pct"])
                    self.symbol_risk[s]["sl_pct"] = float(self.global_risk["sl_pct"])
                logger.info(f"✅ Risk updated for ALL: TP={self.global_risk['tp_pct']}% SL={self.global_risk['sl_pct']}%")
            else:
                self.symbol_risk.setdefault(symbol, {"tp_pct": float(self.global_risk["tp_pct"]), "sl_pct": float(self.global_risk["sl_pct"])})
                self.symbol_risk[symbol]["tp_pct"] = _as_float(tp, self.symbol_risk[symbol]["tp_pct"])
                self.symbol_risk[symbol]["sl_pct"] = _as_float(sl, self.symbol_risk[symbol]["sl_pct"])
                logger.info(f"✅ Risk updated for {symbol}: TP={self.symbol_risk[symbol]['tp_pct']}% SL={self.symbol_risk[symbol]['sl_pct']}%")

            if isinstance(self.engine, PaperTradingEngine):
                if "max_trades_per_day" in (cmd_data or {}):
                    self.engine.max_trades_per_day = int(cmd_data.get("max_trades_per_day", self.engine.max_trades_per_day))
                if "daily_tp_target_percent" in (cmd_data or {}):
                    self.engine.daily_tp_target_percent = float(cmd_data.get("daily_tp_target_percent", self.engine.daily_tp_target_percent))
                self.engine._roll_day_if_needed()

            self.force_cycle = True
            return

        if cmd == "set_filters":
            def _as_float(v, fallback):
                try:
                    if v is None:
                        return float(fallback)
                    return float(v)
                except Exception:
                    return float(fallback)

            def _as_int(v, fallback):
                try:
                    if v is None:
                        return int(fallback)
                    return int(v)
                except Exception:
                    return int(fallback)

            def _as_bool(v, fallback):
                if v is None:
                    return bool(fallback)
                if isinstance(v, bool):
                    return v
                return str(v).strip().lower() in ("1", "true", "yes")

            self.strategy["block_weak_signals"] = _as_bool(cmd_data.get("block_weak_signals"), self.strategy.get("block_weak_signals", True))
            self.strategy["cooldown_bars"] = max(0, _as_int(cmd_data.get("cooldown_bars"), self.strategy.get("cooldown_bars", 3)))
            self.strategy["use_atr_risk"] = _as_bool(cmd_data.get("use_atr_risk"), self.strategy.get("use_atr_risk", True))
            self.strategy["atr_sl_mult"] = max(0.1, _as_float(cmd_data.get("atr_sl_mult"), self.strategy.get("atr_sl_mult", 1.5)))
            self.strategy["atr_tp_mult"] = max(0.1, _as_float(cmd_data.get("atr_tp_mult"), self.strategy.get("atr_tp_mult", 2.5)))
            self.strategy["use_macro"] = _as_bool(cmd_data.get("use_macro"), self.strategy.get("use_macro", USE_MACRO_DEFAULT))
            self.strategy["use_polymarket"] = _as_bool(cmd_data.get("use_polymarket"), self.strategy.get("use_polymarket", USE_POLYMARKET_DEFAULT))
            self.strategy["use_wave_filter"] = _as_bool(cmd_data.get("use_wave_filter"), self.strategy.get("use_wave_filter", USE_WAVE_FILTER_DEFAULT))
            self.strategy["wave_trend_block_pct"] = max(0.0, _as_float(cmd_data.get("wave_trend_block_pct"), self.strategy.get("wave_trend_block_pct", WAVE_TREND_BLOCK_PCT)))
            self.strategy["use_dxy_filter"] = _as_bool(cmd_data.get("use_dxy_filter"), self.strategy.get("use_dxy_filter", USE_DXY_FILTER_DEFAULT))
            self.strategy["dxy_trend_block_pct"] = max(0.0, _as_float(cmd_data.get("dxy_trend_block_pct"), self.strategy.get("dxy_trend_block_pct", DXY_TREND_BLOCK_PCT)))

            self.strategy["quality_mode"] = str((cmd_data.get("quality_mode") if cmd_data.get("quality_mode") is not None else self.strategy.get("quality_mode", QUALITY_MODE_DEFAULT)) or "balanced").strip().lower()
            self.strategy["min_setup_score"] = max(0, _as_int(cmd_data.get("min_setup_score"), self.strategy.get("min_setup_score", MIN_SETUP_SCORE_DEFAULT)))
            self.strategy["use_session_filter"] = _as_bool(cmd_data.get("use_session_filter"), self.strategy.get("use_session_filter", USE_SESSION_FILTER_DEFAULT))
            self.strategy["min_atr_pct"] = max(0.0, _as_float(cmd_data.get("min_atr_pct"), self.strategy.get("min_atr_pct", MIN_ATR_PCT_DEFAULT)))
            self.strategy["max_atr_pct"] = max(0.0, _as_float(cmd_data.get("max_atr_pct"), self.strategy.get("max_atr_pct", MAX_ATR_PCT_DEFAULT)))
            self.strategy["risk_per_trade_pct"] = max(0.0, _as_float(cmd_data.get("risk_per_trade_pct"), self.strategy.get("risk_per_trade_pct", (RISK_PER_TRADE_PCT_REAL if TRADING_MODE == "real" else RISK_PER_TRADE_PCT_DEMO))))
            self.strategy["paper_fee_bps"] = max(0.0, _as_float(cmd_data.get("paper_fee_bps"), self.strategy.get("paper_fee_bps", PAPER_FEE_BPS)))
            self.strategy["paper_spread_bps"] = max(0.0, _as_float(cmd_data.get("paper_spread_bps"), self.strategy.get("paper_spread_bps", PAPER_SPREAD_BPS)))

            if isinstance(self.engine, PaperTradingEngine):
                try:
                    self.engine.risk_per_trade_pct = float(self.strategy.get("risk_per_trade_pct"))
                except Exception:
                    pass
                try:
                    self.engine.fee_bps = float(self.strategy.get("paper_fee_bps"))
                except Exception:
                    pass
                try:
                    self.engine.spread_bps = float(self.strategy.get("paper_spread_bps"))
                except Exception:
                    pass

            self.strategy["risk_guard_enabled"] = _as_bool(cmd_data.get("risk_guard_enabled"), self.strategy.get("risk_guard_enabled", RISK_GUARD_ENABLED_DEFAULT))
            self.strategy["max_open_positions"] = max(0, _as_int(cmd_data.get("max_open_positions"), self.strategy.get("max_open_positions", MAX_OPEN_POSITIONS_DEFAULT)))
            self.strategy["max_daily_drawdown_pct"] = max(0.0, _as_float(cmd_data.get("max_daily_drawdown_pct"), self.strategy.get("max_daily_drawdown_pct", MAX_DAILY_DRAWDOWN_PCT_DEFAULT)))
            self.strategy["max_loss_streak"] = max(0, _as_int(cmd_data.get("max_loss_streak"), self.strategy.get("max_loss_streak", MAX_LOSS_STREAK_DEFAULT)))
            self.strategy["guard_pause_seconds"] = max(0, _as_int(cmd_data.get("guard_pause_seconds"), self.strategy.get("guard_pause_seconds", GUARD_PAUSE_SECONDS_DEFAULT)))

            mode = (cmd_data.get("strategy_mode") or self.strategy_mode or "classic").strip().lower()
            if mode in ("classic", "reinforse", "pro", "mix"):
                self.strategy_mode = "reinforse" if mode == "pro" else mode

            logger.info(
                "✅ Filters updated: "
                f"block_weak_signals={self.strategy['block_weak_signals']}, "
                f"cooldown_bars={self.strategy['cooldown_bars']}, "
                f"use_atr_risk={self.strategy['use_atr_risk']}, "
                f"atr_sl_mult={self.strategy['atr_sl_mult']}, "
                f"atr_tp_mult={self.strategy['atr_tp_mult']}"
            )

            self.force_cycle = True
            return

        if cmd in ("start_single", "trade_single"):
            symbol = cmd_data.get("symbol")
            logger.info(f"🚀 Command: START SINGLE {symbol}")
            if symbol:
                await self.process_symbol(symbol)
            return

        if cmd == "stop_all":
            logger.info("🛑 Command: STOP ALL")
            self.engine.positions = {}
            return

        if cmd == "reset_stats":
            scope = (cmd_data.get("scope") or "all").lower()
            symbol = (cmd_data.get("symbol") or "").strip()
            reset_balance = bool(cmd_data.get("reset_balance", True))

            if isinstance(self.engine, PaperTradingEngine):
                if scope == "symbol" and symbol:
                    self.engine.reset_stats(reset_balance=False, symbol=symbol)
                    logger.info(f"🧹 Stats reset for {symbol}")
                else:
                    self.engine.reset_stats(reset_balance=reset_balance, symbol=None)
                    logger.info("🧹 Stats reset for ALL")
            else:
                try:
                    self.engine.positions = {}
                except Exception:
                    pass

            self.force_cycle = False
            return

        if cmd == "update_api":
            mode = (cmd_data.get("mode") or "demo").lower()
            logger.info(f"🔄 Command: SWITCH MODE -> {mode}")
            TRADING_MODE = mode
            if mode == "real":
                self.engine = RealTradingEngine(
                    cmd_data.get("exchange", "bybit"),
                    cmd_data.get("key"),
                    cmd_data.get("secret")
                )
            else:
                self.engine = PaperTradingEngine()
            return

    async def _check_dashboard_commands(self):
        cmd_path = os.path.join(DATA_DIR, "bot_command.json")
        if not os.path.exists(cmd_path):
            return

        try:
            logger.info(f"📂 Found command file: {cmd_path}")
            with open(cmd_path, "r") as f:
                cmd_data = json.load(f)
            os.remove(cmd_path)
            logger.info(f"📥 Processing file command: {cmd_data.get('command')}")
            await self.apply_command(cmd_data)
        except Exception as e:
            logger.error(f"❌ Error processing dashboard command: {e}")
            if os.path.exists(cmd_path):
                os.remove(cmd_path)

    async def _send_startup(self):
        await asyncio.sleep(2)
        await self.notifier.send_startup_message()

    def _rule_based_decision(self, symbol: str, df, ml_pred: dict) -> dict:
        latest = df.iloc[-1]
        reg = str((ml_pred or {}).get('regime_name') or '').lower()

        try:
            ema_fast = float(latest.get('ema_8', 0.0) or 0.0)
            ema_slow = float(latest.get('ema_21', 0.0) or 0.0)
            macd_h = float(latest.get('macd_hist', 0.0) or 0.0)
            rsi_v = float(latest.get('rsi', 50.0) or 50.0)
        except Exception:
            ema_fast, ema_slow, macd_h, rsi_v = 0.0, 0.0, 0.0, 50.0

        trend_up = ema_fast > ema_slow
        trend_dn = ema_fast < ema_slow

        allow_long = (reg != 'bearish')
        allow_short = (reg != 'bullish')

        long_score = 0
        short_score = 0

        if trend_up:
            long_score += 1
        if trend_dn:
            short_score += 1

        if macd_h > 0:
            long_score += 1
        if macd_h < 0:
            short_score += 1

        if rsi_v <= 60:
            long_score += 1
        if rsi_v >= 40:
            short_score += 1

        strength = 'weak'
        if max(long_score, short_score) >= 3:
            strength = 'medium'

        if long_score >= 3 and allow_long:
            return {
                'trade_decision': 'YES',
                'action': 'entry',
                'side': 'long',
                'signal_strength': strength,
                'tp_pct': float(TAKE_PROFIT_PERCENT),
                'sl_pct': float(STOP_LOSS_PERCENT),
                'reasoning_short': 'Fallback: EMA/MACD/RSI bullish alignment',
            }

        if short_score >= 3 and allow_short:
            return {
                'trade_decision': 'YES',
                'action': 'entry',
                'side': 'short',
                'signal_strength': strength,
                'tp_pct': float(TAKE_PROFIT_PERCENT),
                'sl_pct': float(STOP_LOSS_PERCENT),
                'reasoning_short': 'Fallback: EMA/MACD/RSI bearish alignment',
            }

        return {
            'trade_decision': 'NO',
            'action': 'hold',
            'side': 'long',
            'signal_strength': 'weak',
            'tp_pct': float(TAKE_PROFIT_PERCENT),
            'sl_pct': float(STOP_LOSS_PERCENT),
            'reasoning_short': 'Fallback: no clean alignment',
        }

    def _atr_pct(self, df) -> float | None:
        try:
            latest = df.iloc[-1]
            close = float(latest.get('close', 0.0) or 0.0)
            atr = float(latest.get('atr', 0.0) or 0.0)
            if close <= 0 or atr <= 0:
                return None
            return float(atr / close * 100.0)
        except Exception:
            return None

    def _setup_score(self, df, side: str) -> int:
        side = str(side or 'long').lower()
        if side not in ('long', 'short'):
            side = 'long'

        try:
            latest = df.iloc[-1]
            ema_fast = float(latest.get('ema_8', 0.0) or 0.0)
            ema_slow = float(latest.get('ema_21', 0.0) or 0.0)
            macd_h = float(latest.get('macd_hist', 0.0) or 0.0)
            rsi_v = float(latest.get('rsi', 50.0) or 50.0)
            wave = float(latest.get('wave_trend', 0.0) or 0.0)
        except Exception:
            ema_fast, ema_slow, macd_h, rsi_v, wave = 0.0, 0.0, 0.0, 50.0, 0.0

        score = 0

        trend_up = ema_fast > ema_slow
        trend_dn = ema_fast < ema_slow

        if side == 'long' and trend_up:
            score += 1
        if side == 'short' and trend_dn:
            score += 1

        if side == 'long' and macd_h > 0:
            score += 1
        if side == 'short' and macd_h < 0:
            score += 1

        if side == 'long' and 35.0 <= rsi_v <= 65.0:
            score += 1
        if side == 'short' and 35.0 <= rsi_v <= 65.0:
            score += 1

        if bool(self.strategy.get('use_wave_filter', USE_WAVE_FILTER_DEFAULT)):
            if side == 'long' and wave >= 0:
                score += 1
            if side == 'short' and wave <= 0:
                score += 1

        return int(score)

    def _session_ok(self, symbol: str, df) -> bool:
        sym = str(symbol or '')
        if not sym.endswith('=X'):
            return True

        try:
            ts = df.index[-1]
            if hasattr(ts, 'to_pydatetime'):
                ts = ts.to_pydatetime()
            hour = int(getattr(ts, 'hour', None))
            if hour < 0 or hour > 23:
                hour = int(datetime.utcnow().hour)
        except Exception:
            hour = int(datetime.utcnow().hour)

        return 6 <= hour <= 20

    def _bar_seconds(self, df) -> float:
        try:
            diffs = df.index.to_series().diff().dropna().tail(10)
            if diffs.empty:
                return 3600.0
            sec = float(diffs.median().total_seconds())
            if sec <= 0:
                return 3600.0
            return max(60.0, sec)
        except Exception:
            return 3600.0

    def _in_cooldown(self, symbol: str, df) -> bool:
        cooldown_bars = int(self.strategy.get("cooldown_bars", 0) or 0)
        if cooldown_bars <= 0:
            return False
        last_ts = self.last_action_ts.get(symbol)
        if last_ts is None:
            return False
        try:
            now_ts = pd.Timestamp(df.index[-1])
            bar_sec = self._bar_seconds(df)
            return (now_ts - pd.Timestamp(last_ts)).total_seconds() < (bar_sec * cooldown_bars)
        except Exception:
            return False

    def _persist_bot_meta(self):
        if not isinstance(self.engine, PaperTradingEngine):
            return

        try:
            data = {}
            if os.path.exists(PORTFOLIO_PATH):
                with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}

            meta = dict((data.get("meta") or {}))
            meta["strategy_mode"] = str(getattr(self, "strategy_mode", "classic"))
            meta["filters"] = dict(self.strategy or {})
            reasons = {}
            for k, v in (self.blocked or {}).items():
                if k == "total":
                    continue
                try:
                    reasons[str(k)] = int(v)
                except Exception:
                    continue

            reasons.setdefault("cooldown", int((self.blocked or {}).get("cooldown", 0)))
            reasons.setdefault("weak", int((self.blocked or {}).get("weak", 0)))

            meta["blocked"] = {
                "total": int((self.blocked or {}).get("total", 0)),
                "reasons": reasons,
                "by_symbol": dict(self.blocked_by_symbol or {}),
            }

            data["meta"] = meta
            data["last_update"] = str(datetime.now())

            with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving bot meta: {e}")

    def _record_block(self, symbol: str, reason: str):
        try:
            rsn = str(reason or "unknown")
            self.blocked["total"] = int(self.blocked.get("total", 0)) + 1
            self.blocked[rsn] = int(self.blocked.get(rsn, 0)) + 1

            sym = str(symbol)
            s = dict(self.blocked_by_symbol.get(sym) or {})
            s["total"] = int(s.get("total", 0)) + 1
            s[rsn] = int(s.get(rsn, 0)) + 1
            self.blocked_by_symbol[sym] = s
        except Exception:
            return

        self._persist_bot_meta()

    def _decorate_decision_with_risk(self, symbol: str, price: float, ml_pred: dict, decision: dict, atr: float | None = None) -> dict:
        d = dict(decision or {})
        base = (self.symbol_risk.get(symbol) or self.global_risk)
        base_tp = float(base.get("tp_pct", TAKE_PROFIT_PERCENT))
        base_sl = float(base.get("sl_pct", STOP_LOSS_PERCENT))

        strength = (str(d.get("signal_strength") or "").strip().lower())
        if strength not in ("weak", "medium", "strong"):
            conf = float((ml_pred or {}).get("confidence") or 0.0)
            if conf >= 0.72:
                strength = "strong"
            elif conf >= 0.58:
                strength = "medium"
            else:
                strength = "weak"
        d["signal_strength"] = strength

        def _as_float(v, fallback):
            try:
                if v is None:
                    return float(fallback)
                return float(v)
            except Exception:
                return float(fallback)

        side = (d.get("side") or "long").lower()
        if side not in ("long", "short"):
            side = "long"
        d["side"] = side

        use_atr = bool(self.strategy.get("use_atr_risk", False))
        atr_sl_mult = float(self.strategy.get("atr_sl_mult", 1.5))
        atr_tp_mult = float(self.strategy.get("atr_tp_mult", 2.5))

        if strength == "weak":
            atr_tp_mult = min(atr_tp_mult, float(self.strategy.get("atr_tp_mult", 2.5)) * 0.8)
            atr_sl_mult = max(atr_sl_mult, float(self.strategy.get("atr_sl_mult", 1.5)))
        elif strength == "strong":
            atr_tp_mult = max(atr_tp_mult, float(self.strategy.get("atr_tp_mult", 2.5)) * 1.3)
            atr_sl_mult = max(atr_sl_mult, float(self.strategy.get("atr_sl_mult", 1.5)))

        if use_atr and atr is not None:
            try:
                atr_val = float(atr)
            except Exception:
                atr_val = 0.0
        else:
            atr_val = 0.0

        if atr_val > 0:
            if side == "long":
                sl_price = price - (atr_val * atr_sl_mult)
                tp_price = price + (atr_val * atr_tp_mult)
            else:
                sl_price = price + (atr_val * atr_sl_mult)
                tp_price = price - (atr_val * atr_tp_mult)

            sl_pct = abs((price - sl_price) / price) * 100.0
            tp_pct = abs((tp_price - price) / price) * 100.0

            d["stop_loss"] = float(sl_price)
            d["take_profit"] = float(tp_price)
        else:
            tp_pct = _as_float(d.get("tp_pct"), base_tp)
            sl_pct = _as_float(d.get("sl_pct"), base_sl)

            if strength == "weak":
                tp_pct = min(tp_pct, base_tp * 0.7)
                sl_pct = min(sl_pct, base_sl * 0.9)
            elif strength == "strong":
                tp_pct = max(tp_pct, base_tp * 1.4)
                sl_pct = max(sl_pct, base_sl)

            if side == "long":
                d["take_profit"] = price * (1.0 + tp_pct / 100.0)
                d["stop_loss"] = price * (1.0 - sl_pct / 100.0)
            else:
                d["take_profit"] = price * (1.0 - tp_pct / 100.0)
                d["stop_loss"] = price * (1.0 + sl_pct / 100.0)

        tp_pct = max(0.5, min(15.0, float(tp_pct)))
        sl_pct = max(0.3, min(10.0, float(sl_pct)))

        d["tp_pct"] = tp_pct
        d["sl_pct"] = sl_pct

        try:
            d["confidence"] = int(round(float((ml_pred or {}).get("confidence") or 0.0) * 100))
        except Exception:
            d["confidence"] = 50

        d["ml_regime"] = (ml_pred or {}).get("regime_name")
        d["ml_confidence"] = (ml_pred or {}).get("confidence")
        return d

    async def process_symbol(self, symbol):
        try:
            df = fetch_data(symbol)
            if df is None or len(df) < 50:
                logger.warning(f"⚠️ {symbol}: Not enough data")
                return

            current_price = df['close'].iloc[-1]

            try:
                if symbol in self.engine.positions and isinstance(self.engine.positions.get(symbol), dict):
                    self.engine.positions[symbol]['last_price'] = float(current_price)
            except Exception:
                pass

            # ML Prediction
            ml_pred = self.ml_loader.predict(df)
            if not ml_pred:
                ml_pred = {
                    'regime': 1,
                    'regime_name': 'flat',
                    'confidence': 0.34,
                    'probabilities': {'bearish': 0.33, 'flat': 0.34, 'bullish': 0.33},
                }

            # Check exits first
            closed = self.engine.check_exits(symbol, current_price)
            if closed:
                try:
                    self.last_action_ts[symbol] = df.index[-1]
                except Exception:
                    pass

                await self.notifier.send_signal("CLOSE", {
                    'asset': symbol,
                    'side': closed['side'],
                    'entry_price': closed['entry_price'],
                    'exit_price': closed['exit_price'],
                    'pnl': closed['pnl'],
                    'pnl_percent': closed['pnl_percent'],
                    'balance': self.engine.balance,
                    'reason': closed.get('exit_reason', 'TP/SL hit')
                })

                try:
                    ctx = self.context if isinstance(getattr(self, "context", None), dict) else {}
                    self._review_and_persist_closed_trade(closed, df, ctx)
                except Exception as e:
                    logger.error(f"Trade review/persist error: {e}")

            # Only enter if no position
            if symbol not in self.engine.positions:
                if self._in_cooldown(symbol, df):
                    logger.info(f"⏳ {symbol}: Cooldown active")
                    self._record_block(symbol, "cooldown")
                    return

                if float(getattr(self, 'pause_until_ts', 0.0) or 0.0) > time.time():
                    logger.info(f"⛔ {symbol}: Risk Guard pause active")
                    return

                if bool(self.strategy.get('risk_guard_enabled', RISK_GUARD_ENABLED_DEFAULT)):
                    max_pos = int(self.strategy.get('max_open_positions', MAX_OPEN_POSITIONS_DEFAULT) or 0)
                    if max_pos > 0 and len(self.engine.positions) >= max_pos:
                        logger.info(f"⛔ {symbol}: Risk Guard (max_open_positions={max_pos})")
                        return

                    if isinstance(self.engine, PaperTradingEngine):
                        max_ls = int(self.strategy.get('max_loss_streak', MAX_LOSS_STREAK_DEFAULT) or 0)
                        if max_ls > 0 and int(getattr(self.engine, 'loss_streak', 0) or 0) >= max_ls:
                            self.pause_until_ts = time.time() + float(self.strategy.get('guard_pause_seconds', GUARD_PAUSE_SECONDS_DEFAULT) or 0)
                            logger.info(f"⛔ {symbol}: Risk Guard (loss_streak={self.engine.loss_streak} >= {max_ls})")
                            return

                        dd_limit = float(self.strategy.get('max_daily_drawdown_pct', MAX_DAILY_DRAWDOWN_PCT_DEFAULT) or 0.0)
                        try:
                            used_margin = sum(float(p.get('margin') or 0.0) for p in self.engine.positions.values() if isinstance(p, dict))
                            unreal = sum(float(p.get('unrealized_pnl') or 0.0) for p in self.engine.positions.values() if isinstance(p, dict))
                            equity_now = float(self.engine.balance) + used_margin + unreal
                            day_start = float(getattr(self.engine, 'day_start_balance', 0.0) or 0.0)
                            dd_pct = ((equity_now - day_start) / day_start * 100.0) if day_start > 0 else 0.0
                            if dd_limit > 0 and dd_pct <= -abs(dd_limit):
                                self.pause_until_ts = time.time() + float(self.strategy.get('guard_pause_seconds', GUARD_PAUSE_SECONDS_DEFAULT) or 0)
                                logger.info(f"⛔ {symbol}: Risk Guard (daily_drawdown={dd_pct:.2f}% <= -{dd_limit:.2f}%)")
                                return
                        except Exception:
                            pass

                logger.info(f"🔍 {symbol}: Requesting AI decision...")
                ctx = self.context if isinstance(getattr(self, "context", None), dict) else {}
                mode = (getattr(self, "strategy_mode", None) or STRATEGY_MODE or "classic").strip().lower()

                # Backward-compat: "pro" == "reinforse"
                if mode == "pro":
                    mode = "reinforse"

                if mode == "mix":
                    d_classic = self.ai.get_decision(symbol, df, ml_pred, context=ctx) or {}
                    d_reinf = self.ai.get_reinforse_decision(symbol, df, ml_pred, context=ctx) or {}

                    def _is_yes(d: dict) -> bool:
                        return str(d.get('trade_decision') or '').upper() == 'YES' and str(d.get('action') or '').lower() == 'entry'

                    yes_c = _is_yes(d_classic)
                    yes_r = _is_yes(d_reinf)
                    side_c = str(d_classic.get('side') or 'long').lower()
                    side_r = str(d_reinf.get('side') or 'long').lower()

                    def _rank(v: str) -> int:
                        v = str(v or "").lower()
                        return 3 if v == "strong" else 2 if v == "medium" else 1 if v == "weak" else 0

                    # MIX: (1) open if BOTH agree and same side
                    if yes_c and yes_r and side_c == side_r:
                        decision = dict(d_reinf)  # prefer reinforse payload when aligned
                        decision['side'] = side_r
                        decision['trade_decision'] = 'YES'
                        decision['action'] = 'entry'
                        decision['reasoning_short'] = f"MIX agree: classic={d_classic.get('reasoning_short','')}; reinforse={d_reinf.get('reasoning_short','')}"[:250]

                    # (2) strong override: if exactly one says YES and it's STRONG -> allow
                    elif yes_c ^ yes_r:
                        winner = d_reinf if yes_r else d_classic
                        loser = d_classic if yes_r else d_reinf
                        if _rank(winner.get('signal_strength')) >= 3:
                            decision = dict(winner)
                            decision['trade_decision'] = 'YES'
                            decision['action'] = 'entry'
                            decision['reasoning_short'] = f"MIX strong override: yes={winner.get('reasoning_short','')}; no={loser.get('reasoning_short','')}"[:250]
                        else:
                            decision = {'trade_decision': 'NO', 'action': 'hold', 'signal_strength': 'weak', 'reasoning_short': 'MIX disagree'}
                    else:
                        decision = {'trade_decision': 'NO', 'action': 'hold', 'signal_strength': 'weak', 'reasoning_short': 'MIX disagree'}

                    mode = "mix"
                elif mode == "reinforse":
                    decision = self.ai.get_reinforse_decision(symbol, df, ml_pred, context=ctx)
                    if int((decision or {}).get('ai_error_code') or 0) == 402:
                        decision = self._rule_based_decision(symbol, df, ml_pred)
                        mode = 'reinforse'
                else:
                    mode = "classic"
                    decision = self.ai.get_decision(symbol, df, ml_pred, context=ctx)
                    if int((decision or {}).get('ai_error_code') or 0) == 402:
                        decision = self._rule_based_decision(symbol, df, ml_pred)
                        mode = 'classic'

                if not isinstance(decision, dict):
                    decision = {'trade_decision': 'NO', 'action': 'hold', 'signal_strength': 'weak', 'reasoning_short': 'invalid_decision'}

                if int(decision.get('ai_error_code') or 0) in (401, 402, 500):
                    decision = self._rule_based_decision(symbol, df, ml_pred)

                decision["strategy_mode"] = mode

                atr = None
                try:
                    atr = float(df.iloc[-1].get("atr", 0.0))
                except Exception:
                    atr = None
                decision = self._decorate_decision_with_risk(symbol, float(current_price), ml_pred, decision, atr=atr)

                side = str(decision.get('side') or 'long').lower()
                quality_mode = str(self.strategy.get('quality_mode') or 'balanced').lower()
                min_score = int(self.strategy.get('min_setup_score', MIN_SETUP_SCORE_DEFAULT) or 0)

                setup_score = self._setup_score(df, side)
                atr_pct = self._atr_pct(df)
                use_sess = bool(self.strategy.get('use_session_filter', USE_SESSION_FILTER_DEFAULT))
                sess_ok = (not use_sess) or self._session_ok(symbol, df)

                min_atr = float(self.strategy.get('min_atr_pct', MIN_ATR_PCT_DEFAULT) or 0.0)
                max_atr = float(self.strategy.get('max_atr_pct', MAX_ATR_PCT_DEFAULT) or 0.0)
                atr_ok = True
                if atr_pct is not None and max_atr > 0:
                    atr_ok = (atr_pct >= min_atr) and (atr_pct <= max_atr)

                if str(decision.get('trade_decision') or '').upper() == 'YES':
                    if not sess_ok:
                        decision['trade_decision'] = 'NO'
                        decision['action'] = 'hold'
                        decision['reasoning_short'] = 'Quality: session filter'
                        self._record_block(symbol, 'quality_session')
                    elif not atr_ok:
                        decision['trade_decision'] = 'NO'
                        decision['action'] = 'hold'
                        decision['reasoning_short'] = f"Quality: ATR% out of range ({(atr_pct or 0.0):.3f}%)"
                        self._record_block(symbol, 'quality_atr')
                    elif quality_mode == 'high' and min_score > 0 and setup_score < min_score:
                        decision['trade_decision'] = 'NO'
                        decision['action'] = 'hold'
                        decision['reasoning_short'] = f"Quality: setup_score {setup_score}/{min_score}"
                        self._record_block(symbol, 'quality_score')
                    elif quality_mode == 'high' and str(decision.get('signal_strength') or '').lower() == 'weak':
                        decision['trade_decision'] = 'NO'
                        decision['action'] = 'hold'
                        decision['reasoning_short'] = 'Quality: weak strength'
                        self._record_block(symbol, 'quality_strength')

                if bool(self.strategy.get("block_weak_signals", False)) and str(decision.get("signal_strength") or "").lower() == "weak":
                    logger.info(f"⛔ {symbol}: Weak signal blocked")
                    self._record_block(symbol, "weak")
                    return

                try:
                    wave_trend = float(df.iloc[-1].get('wave_trend', 0.0))
                except Exception:
                    wave_trend = 0.0

                side = str(decision.get('side') or 'long').lower()

                use_wave = bool(self.strategy.get('use_wave_filter', USE_WAVE_FILTER_DEFAULT))
                wave_thr = float(self.strategy.get('wave_trend_block_pct', WAVE_TREND_BLOCK_PCT) or WAVE_TREND_BLOCK_PCT)
                if use_wave and abs(wave_trend) >= wave_thr:
                    if (wave_trend > 0 and side == 'short') or (wave_trend < 0 and side == 'long'):
                        decision['trade_decision'] = 'NO'
                        decision['action'] = 'hold'
                        decision['reasoning_short'] = f"Wave trend conflict ({wave_trend:.2f}%)"
                        logger.info(f"⛔ {symbol}: Wave trend conflicts with side ({wave_trend:.2f}% vs {side})")

                macro = ctx.get('macro') if isinstance(ctx, dict) else None
                use_dxy = bool(self.strategy.get('use_dxy_filter', USE_DXY_FILTER_DEFAULT))
                dxy_thr = float(self.strategy.get('dxy_trend_block_pct', DXY_TREND_BLOCK_PCT) or DXY_TREND_BLOCK_PCT)
                if use_dxy and isinstance(macro, dict) and abs(float(macro.get('DX-Y.NYB', {}).get('wave_trend_pct', 0.0) or 0.0)) >= dxy_thr:
                    dxy_tr = float(macro.get('DX-Y.NYB', {}).get('wave_trend_pct', 0.0) or 0.0)
                    sym = str(symbol)
                    if sym.endswith('=X') and len(sym) >= 6 and sym[3:6] == 'USD':
                        if (dxy_tr > 0 and side == 'long') or (dxy_tr < 0 and side == 'short'):
                            decision['trade_decision'] = 'NO'
                            decision['action'] = 'hold'
                            decision['reasoning_short'] = f"DXY trend conflict ({dxy_tr:.2f}%)"
                            logger.info(f"⛔ {symbol}: DXY trend conflicts with side ({dxy_tr:.2f}% vs {side})")
                    elif sym.endswith('=X') and len(sym) >= 6 and sym[0:3] == 'USD':
                        if (dxy_tr > 0 and side == 'short') or (dxy_tr < 0 and side == 'long'):
                            decision['trade_decision'] = 'NO'
                            decision['action'] = 'hold'
                            decision['reasoning_short'] = f"DXY trend conflict ({dxy_tr:.2f}%)"
                            logger.info(f"⛔ {symbol}: DXY trend conflicts with side ({dxy_tr:.2f}% vs {side})")

                if decision.get('trade_decision') == 'YES':
                    logger.info(f"✅ AI signal: {decision.get('side','long').upper()} {symbol} | TP={decision.get('tp_pct')}% SL={decision.get('sl_pct')}% | strength={decision.get('signal_strength')} | mode={decision.get('strategy_mode')}")
                    position = self.engine.execute_entry(symbol, decision, current_price)
                    if position:
                        try:
                            self.last_action_ts[symbol] = df.index[-1]
                        except Exception:
                            pass

                        await self.notifier.send_signal("ENTRY", {
                            'asset': symbol,
                            'side': position.get('side', decision.get('side', 'long')),
                            'trade_decision': 'YES',
                            'analysis': decision.get('reasoning_short', ''),
                            'tp_pct': decision.get('tp_pct'),
                            'sl_pct': decision.get('sl_pct'),
                            'signal_strength': decision.get('signal_strength'),
                            'strategy_mode': decision.get('strategy_mode'),
                            'entry_price': position.get('entry_price'),
                            'size': position.get('size'),
                            'take_profit': position.get('take_profit'),
                            'stop_loss': position.get('stop_loss'),
                            'notional': position.get('notional'),
                            'margin': position.get('margin'),
                            'balance_before': position.get('balance_before'),
                            'balance_after': position.get('balance_after'),
                        })
                    else:
                        logger.warning(f"❌ {symbol}: Entry execution failed (insufficient balance or engine error)")
                else:
                    logger.info(f"⏳ {symbol}: AI said NO. Reason: {decision.get('reasoning_short', 'No reason provided')}")
            else:
                logger.info(f"ℹ️ {symbol}: Already in position, skipping entry check.")

            # Log current status
            logger.info(f"{symbol}: {current_price:.5f} | ML: {ml_pred['regime_name']} ({ml_pred['confidence']:.1%})")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    async def run_cycle(self):
        # Check for commands from dashboard
        await self._check_dashboard_commands()
        
        logger.info("=" * 60)
        logger.info("STARTING TRADING CYCLE")
        logger.info("=" * 60)

        ctx = {"ts": str(datetime.now())}
        ctx["filters"] = dict(self.strategy or {})
        if bool(self.strategy.get("use_macro", USE_MACRO_DEFAULT)):
            ctx["macro"] = fetch_macro_snapshot()
        if bool(self.strategy.get("use_polymarket", USE_POLYMARKET_DEFAULT)):
            ctx["polymarket"] = fetch_polymarket_snapshot()
        if self._db_ready:
            ctx["lessons"] = self._load_lessons(limit=int(LESSONS_LIMIT))
        self.context = ctx

        for symbol in ALL_SYMBOLS:
            await self.process_symbol(symbol)

        # Removed automatic summary notification to prevent spam
        # Summary is now available only via /status command

        current_prices = {}
        for symbol in ALL_SYMBOLS:
            df = fetch_data(symbol)
            if df is not None:
                current_prices[symbol] = df['close'].iloc[-1]

        state = self.engine.get_state(current_prices)

        if isinstance(self.engine, PaperTradingEngine):
            try:
                for s, px in (current_prices or {}).items():
                    if s in self.engine.positions and isinstance(self.engine.positions.get(s), dict):
                        self.engine.positions[s]['last_price'] = float(px)
                self.engine._save_state()
            except Exception as e:
                logger.error(f"Error saving cycle state: {e}")

        logger.info(f"Portfolio: Balance=${state['balance']:.2f}, Equity=${state['equity']:.2f}, PnL={state['pnl']:.2f}%")
        next_in = REAL_CYCLE_SECONDS if TRADING_MODE == "real" else DEMO_CYCLE_SECONDS
        logger.info(f"Cycle completed. Next in {int(next_in)} seconds.")

    async def send_portfolio_summary(self):
        """Calculate state and send summary via notifier"""
        current_prices = {}
        for s in ALL_SYMBOLS:
            df = fetch_data(s)
            if df is not None: current_prices[s] = df['close'].iloc[-1]
        state = self.engine.get_state(current_prices)
        await self.notifier.send_daily_summary(state)

    async def run(self):
        global BOT_LOOP
        BOT_LOOP = asyncio.get_running_loop()

        # Initialize Telegram
        await self.notifier.initialize()
        await self._send_startup()
        
        last_summary_time = time.time()
        last_cycle_time = 0
        self.force_cycle = False
        
        logger.info("📡 Bot is active and waiting for dashboard commands...")
        
        while self.running:
            try:
                # 1. Check for commands from dashboard
                await self._check_dashboard_commands()
                
                current_time = time.time()
                
                # 2. Check if it's time for a scheduled trading cycle (demo faster) or forced
                cycle_seconds = REAL_CYCLE_SECONDS if TRADING_MODE == "real" else DEMO_CYCLE_SECONDS
                if (current_time - last_cycle_time > cycle_seconds) or getattr(self, 'force_cycle', False):
                    if getattr(self, 'force_cycle', False):
                        logger.info("⚡ Forced cycle triggered by command")
                        self.force_cycle = False
                    
                    await self.run_cycle()
                    last_cycle_time = current_time
                
                # 3. Check if it's time for 12-hour summary
                if current_time - last_summary_time > 12 * 3600:
                    logger.info("🕒 Sending scheduled 12-hour summary...")
                    await self.send_portfolio_summary()
                    last_summary_time = current_time
                
                # Frequent polling for commands (every 2 seconds)
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(10)


# ============================================
# ENTRY POINT
# ============================================
bot_instance = None

async def main():
    global bot_instance
    try:
        bot_instance = TradingBot()
        await bot_instance.run()
    except Exception as e:
        logger.error(f"FATAL ERROR DURING BOT RUN: {e}")
        if RAILWAY:
            logger.info("Keeping process alive for health check...")
            while True:
                await asyncio.sleep(10)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        if os.getenv("RAILWAY", "false").lower() == "true":
            while True:
                time.sleep(10)