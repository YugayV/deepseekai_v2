"""
EURUSD AI Trading Bot with DeepSeek
Inspired by: https://github.com/tot-gromov/llm-deepseek-trading
Optimized models with Alligator + Fractals
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
DEMO_CYCLE_SECONDS = int(os.getenv("DEMO_CYCLE_SECONDS", 60))
REAL_CYCLE_SECONDS = int(os.getenv("REAL_CYCLE_SECONDS", 600))

MIN_ML_CONFIDENCE = float(os.getenv("MIN_ML_CONFIDENCE", 0.70))
BLOCK_WEAK_SIGNALS = (os.getenv("BLOCK_WEAK_SIGNALS", "true").strip().lower() in ("1", "true", "yes"))
COOLDOWN_BARS = int(os.getenv("COOLDOWN_BARS", 3))
USE_ATR_RISK = (os.getenv("USE_ATR_RISK", "true").strip().lower() in ("1", "true", "yes"))
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", 1.5))
ATR_TP_MULT = float(os.getenv("ATR_TP_MULT", 2.5))

RAILWAY = os.getenv("RAILWAY", "false").lower() == "true"
TRADING_MODE = os.getenv("TRADING_MODE", "demo").lower()
STRATEGY_MODE = (os.getenv("STRATEGY_MODE") or "classic").strip().lower()

DEEPSEEK_API_KEY = os.getenv("OPENROUTER_API_KEY")

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
            await query.edit_message_text(f"⌛ Requesting DeepSeek analysis for {symbol}...", parse_mode='Markdown')
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
            risk_line = ""
            if tp_pct is not None and sl_pct is not None:
                risk_line = f"*TP/SL:* {float(tp_pct):.2f}% / {float(sl_pct):.2f}%\n"
            if strength:
                risk_line += f"*Signal Strength:* {strength}\n"

            text = f"""
{emoji} *TRADE SIGNAL: {data['asset']}*

*ML Prediction:* {data.get('ml_forecast_pct', '0%')}
*Decision:* {data.get('trade_decision', 'NO')}
{risk_line}
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
            self.target_mapping = self.metadata.get('target_mapping', {0: 'bearish', 1: 'flat', 2: 'bullish'})
        except:
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
# INDICATORS CALCULATION (Alligator + Fractals)
# ============================================
def calculate_indicators(df):
    """Calculate all technical indicators (Alligator + Fractals)"""
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

    # ALLIGATOR (SMAs with shifts)
    df['jaw'] = df['close'].rolling(13).mean().shift(8)
    df['teeth'] = df['close'].rolling(8).mean().shift(5)
    df['lips'] = df['close'].rolling(5).mean().shift(3)

    jaw_lips_diff = (df['jaw'] - df['lips']).abs()
    df['alligator_asleep'] = (jaw_lips_diff < df['atr'] * 0.5).astype(int)
    df['alligator_bullish'] = ((df['jaw'] < df['teeth']) & (df['teeth'] < df['lips'])).astype(int)
    df['alligator_bearish'] = ((df['jaw'] > df['teeth']) & (df['teeth'] > df['lips'])).astype(int)

    jaw_teeth_diff = (df['jaw'] - df['teeth']).abs()
    teeth_lips_diff = (df['teeth'] - df['lips']).abs()
    df['alligator_expanding'] = ((jaw_teeth_diff > df['atr'] * 0.3) & (teeth_lips_diff > df['atr'] * 0.3)).astype(int)

    # FRACTALS
    window = 2
    bullish = np.zeros(len(df))
    bearish = np.zeros(len(df))

    for i in range(window, len(df) - window):
        if all(df['low'].iloc[i] < df['low'].iloc[i - j] for j in range(1, window + 1)) and \
           all(df['low'].iloc[i] < df['low'].iloc[i + j] for j in range(1, window + 1)):
            bullish[i] = 1
        if all(df['high'].iloc[i] > df['high'].iloc[i - j] for j in range(1, window + 1)) and \
           all(df['high'].iloc[i] > df['high'].iloc[i + j] for j in range(1, window + 1)):
            bearish[i] = 1

    df['fractal_bullish'] = pd.Series(bullish).shift(window).fillna(0).values
    df['fractal_bearish'] = pd.Series(bearish).shift(window).fillna(0).values

    # Combined signals
    df['bullish_fractal_alligator'] = ((df['fractal_bullish'] == 1) &
                                        ((df['alligator_bullish'] == 1) | (df['alligator_asleep'] == 1))).astype(int)
    df['bearish_fractal_alligator'] = ((df['fractal_bearish'] == 1) &
                                        ((df['alligator_bearish'] == 1) | (df['alligator_asleep'] == 1))).astype(int)
    df['strong_bullish'] = ((df['fractal_bullish'] == 1) &
                            (df['alligator_bullish'] == 1) &
                            (df['alligator_expanding'] == 1)).astype(int)
    df['strong_bearish'] = ((df['fractal_bearish'] == 1) &
                            (df['alligator_bearish'] == 1) &
                            (df['alligator_expanding'] == 1)).astype(int)

    # Lags
    for lag in [1, 2, 3, 5]:
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        df[f'close_lag_{lag}'] = df['close'].shift(lag)

    return df


# ============================================
# DEEPSEEK ADVISOR (following project contract)
# ============================================
class DeepSeekAdvisor:
    def __init__(self, api_key=None):
        self.client = openai.OpenAI(
            api_key=api_key or DEEPSEEK_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        ) if api_key or DEEPSEEK_API_KEY else None
        self.model = "deepseek/deepseek-chat"

    def get_decision(self, symbol, df, ml_prediction):
        if not self.client:
            return {'action': 'hold', 'confidence': 0, 'reasoning': 'No API key'}

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        recent = df[['open', 'high', 'low', 'close']].tail(6).round(6).to_dict('records')

        if int(latest.get('alligator_bullish', 0)) == 1:
            alligator_state = "BULLISH"
        elif int(latest.get('alligator_bearish', 0)) == 1:
            alligator_state = "BEARISH"
        else:
            alligator_state = "NEUTRAL"

        fractal_state = "BULLISH" if int(latest.get('fractal_bullish', 0)) == 1 else "BEARISH" if int(latest.get('fractal_bearish', 0)) == 1 else "NONE"

        prompt = f"""
You are a binary trading decision engine for {symbol}.

INPUTS:
- Timeframe: 1h (primary)
- ML Regime: {ml_prediction['regime_name'].upper()}
- ML Confidence: {ml_prediction['confidence']:.1%}
- Alligator: {alligator_state} (jaw={float(latest.get('jaw', 0)):.6f}, teeth={float(latest.get('teeth', 0)):.6f}, lips={float(latest.get('lips', 0)):.6f})
- Fractals: {fractal_state}
- Last candles (JSON): {recent}

TASK:
Decide if we should open a position RIGHT NOW.
Use Alligator + Fractals as primary filter; ML confidence is secondary.

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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5, # Increased for more variety
                max_tokens=300
            )
            content = response.choices[0].message.content
            import re
            match = re.search(r'\{[^{}]*\}', content)
            if match:
                return json.loads(match.group())
            return {'action': 'hold', 'confidence': 0, 'reasoning': 'Parse error'}
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg:
                logger.error(f"❌ API Key Error: Please check your OPENROUTER_API_KEY in .env")
            else:
                logger.error(f"DeepSeek error: {e}")
            return {'action': 'hold', 'confidence': 0, 'reasoning': f"API Error: {error_msg}"}

    def get_pro_decision(self, symbol, df, ml_prediction):
        if not self.client:
            return {'action': 'hold', 'confidence': 0, 'reasoning': 'No API key'}

        latest = df.iloc[-1]
        recent = df[['open', 'high', 'low', 'close']].tail(12).round(6).to_dict('records')

        try:
            bb_pos = float((latest.get('close', 0) - latest.get('bb_lower', 0)) / (latest.get('bb_upper', 1) - latest.get('bb_lower', 0)))
        except Exception:
            bb_pos = 0.0

        prompt = f"""
You are a trading strategy engine for {symbol}.

INPUTS (latest bar):
- ML Regime: {ml_prediction['regime_name'].upper()} | Confidence: {ml_prediction['confidence']:.1%}
- RSI(14): {float(latest.get('rsi', 0)):.2f}
- MACD hist: {float(latest.get('macd_hist', 0)):.6f}
- Bollinger position (0=lower..1=upper): {bb_pos:.3f}
- ATR(14): {float(latest.get('atr', 0)):.6f}
- Alligator bullish: {int(latest.get('alligator_bullish', 0))} | bearish: {int(latest.get('alligator_bearish', 0))} | asleep: {int(latest.get('alligator_asleep', 0))}
- Fractal bullish: {int(latest.get('fractal_bullish', 0))} | bearish: {int(latest.get('fractal_bearish', 0))}
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=300,
            )
            content = response.choices[0].message.content
            import re
            match = re.search(r'\{[^{}]*\}', content)
            if match:
                return json.loads(match.group())
            return {'action': 'hold', 'confidence': 0, 'reasoning': 'Parse error'}
        except Exception as e:
            return {'action': 'hold', 'confidence': 0, 'reasoning': f"API Error: {e}"}


# ============================================
# DATA FETCHER (multi-timeframe)
# ============================================
def fetch_data(symbol):
    """Fetch latest data and calculate indicators (15m, 1h, 4h for multi-timeframe)"""
    ticker = yf.Ticker(symbol)

    # Fetch multiple timeframes
    df_15m = ticker.history(period="5d", interval="15m")
    df_1h = ticker.history(period="1mo", interval="1h")
    df_4h = ticker.history(period="3mo", interval="4h")

    # Use 1h as primary for trading signals
    df = df_1h if not df_1h.empty else df_4h

    if df.empty:
        # Fallback to daily if hourly fails
        df = ticker.history(period="3mo", interval="1d")

    if df.empty:
        return None

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']

    return calculate_indicators(df)


# ============================================
# REAL TRADING ENGINE (CCXT)
# ============================================
class RealTradingEngine:
    def __init__(self, exchange_id, api_key, secret):
        try:
            import ccxt
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True,
            })
            logger.info(f"✅ Real Trading initialized on {exchange_id}")
        except Exception as e:
            logger.error(f"❌ Failed to init real trading: {e}")
            self.exchange = None

    def execute_entry(self, symbol, decision, price):
        if not self.exchange: return None
        try:
            side = decision.get('side', 'long')
            order_side = 'buy' if side == 'long' else 'sell'
            balance = self.exchange.fetch_balance()
            amount = (balance['total']['USDT'] * (MAX_POSITION_SIZE / 100)) / price
            
            logger.info(f"🚀 PLACING REAL ORDER: {symbol} {order_side} {amount}")
            return {'side': side, 'entry_price': price, 'size': amount}
        except Exception as e:
            logger.error(f"Order error: {e}")
            return None

    def check_exits(self, symbol, price):
        return None

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

        self.max_trades_per_day = MAX_TRADES_PER_DAY
        self.daily_tp_target_percent = DAILY_TP_TARGET_PERCENT
        self.day_date = datetime.utcnow().date().isoformat()
        self.day_start_balance = initial_capital

        self._load_state()
        self._roll_day_if_needed()

    def _save_state(self):
        """Save portfolio state and trade history for the dashboard"""
        try:
            # Save portfolio state
            state = {
                'balance': self.balance,
                'equity': self.balance,
                'positions': self.positions,
                'last_update': str(datetime.now()),
                'daily': {
                    'date': self.day_date,
                    'start_balance': self.day_start_balance,
                    'trades': self.daily_trades,
                    'max_trades_per_day': self.max_trades_per_day,
                    'daily_tp_target_percent': self.daily_tp_target_percent
                }
            }
            # Handle datetime objects for JSON
            serializable_positions = {}
            for s, p in self.positions.items():
                p_copy = p.copy()
                if 'entry_date' in p_copy: p_copy['entry_date'] = str(p_copy['entry_date'])
                serializable_positions[s] = p_copy
            state['positions'] = serializable_positions
            
            with open(PORTFOLIO_PATH, "w") as f:
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

        risk_amount = self.balance * (MAX_POSITION_SIZE / 100) * confidence
        risk_per_unit = abs(price - stop)
        
        if risk_per_unit <= 0:
            logger.warning(f"❌ {symbol}: Risk per unit is 0 (Price={price}, SL={stop})")
            return None

        size = risk_amount / risk_per_unit
        size = min(size, self.balance * 0.3 / price) # Max 30% of balance

        if size <= 0:
            logger.warning(f"❌ {symbol}: Calculated size is 0 (Risk={risk_amount}, UnitRisk={risk_per_unit})")
            return None

        self.positions[symbol] = {
            'asset': symbol,
            'side': side,
            'entry_price': price,
            'stop_loss': stop,
            'take_profit': target,
            'size': size,
            'confidence': confidence,
            'entry_date': datetime.now(),
            'analysis': str(decision.get('reasoning_short', '') or ''),
            'signal_strength': str(decision.get('signal_strength', '') or ''),
            'tp_pct': float(decision.get('tp_pct', 0) or 0),
            'sl_pct': float(decision.get('sl_pct', 0) or 0),
            'ml_regime': str(decision.get('ml_regime', '') or ''),
            'ml_confidence': float(decision.get('ml_confidence', 0) or 0),
            'strategy_mode': str(decision.get('strategy_mode', '') or ''),
        }

        margin = size * price
        self.balance -= margin
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
            self.balance += pos['size'] * price + pnl
            pos['exit_price'] = price
            pos['pnl'] = pnl
            pos['pnl_percent'] = (pnl / (pos['size'] * pos['entry_price'])) * 100
            pos['exit_reason'] = close_reason
            pos['exit_date'] = str(datetime.now())

            self.trades.append(pos)
            del self.positions[symbol]
            
            self._save_state() # Save after exit
            return pos

        return None

    def get_state(self, current_prices):
        equity = self.balance
        for symbol, pos in self.positions.items():
            price = current_prices.get(symbol, 0)
            if pos['side'] == 'long':
                equity += pos['size'] * price
            else:
                equity += pos['size'] * (2 * pos['entry_price'] - price)

        return {
            'balance': self.balance,
            'equity': equity,
            'positions': len(self.positions),
            'pnl': (equity - PAPER_CAPITAL) / PAPER_CAPITAL * 100
        }


# ============================================
# MAIN BOT
# ============================================
class TradingBot:
    def __init__(self):
        self.ml_loader = MLModelLoader()
        self.deepseek = DeepSeekAdvisor()

        self.cycle_seconds = DEMO_CYCLE_SECONDS if TRADING_MODE == "demo" else REAL_CYCLE_SECONDS
        
        # Select Engine based on mode
        if TRADING_MODE == "real" and EXCHANGE_API_KEY:
            self.engine = RealTradingEngine(EXCHANGE_ID, EXCHANGE_API_KEY, EXCHANGE_API_SECRET)
        else:
            self.engine = PaperTradingEngine()
        
        self.global_risk = {
            "tp_pct": float(TAKE_PROFIT_PERCENT),
            "sl_pct": float(STOP_LOSS_PERCENT),
        }
        self.symbol_risk = {s: {"tp_pct": float(self.global_risk["tp_pct"]), "sl_pct": float(self.global_risk["sl_pct"])} for s in ALL_SYMBOLS}

        self.strategy_mode = STRATEGY_MODE
        self.strategy = {
            "min_ml_confidence": float(MIN_ML_CONFIDENCE),
            "block_weak_signals": bool(BLOCK_WEAK_SIGNALS),
            "cooldown_bars": int(COOLDOWN_BARS),
            "use_atr_risk": bool(USE_ATR_RISK),
            "atr_sl_mult": float(ATR_SL_MULT),
            "atr_tp_mult": float(ATR_TP_MULT),
        }
        self.last_action_ts = {}

        # Multi-channel: group chat for signals, private for errors
        self.notifier = MultiChannelNotifier(
            token=TELEGRAM_TOKEN,
            group_id=os.getenv("TELEGRAM_GROUP_ID") or os.getenv("TELEGRAM_CHAT_ID"),
            admin_chat_id=os.getenv("TELEGRAM_CHAT_ID")
        )
        self.running = True
    
    async def apply_command(self, cmd_data: dict):
        global TAKE_PROFIT_PERCENT, STOP_LOSS_PERCENT, MAX_POSITION_SIZE, TRADING_MODE

        cmd = (cmd_data or {}).get("command")
        if not cmd:
            return

        if cmd == "start_all":
            logger.info(f"🚀 Command: START ALL (TP={cmd_data.get('tp', TAKE_PROFIT_PERCENT)}, SL={cmd_data.get('sl', STOP_LOSS_PERCENT)}, Lev={cmd_data.get('leverage', 1)})")
            TAKE_PROFIT_PERCENT = float(cmd_data.get("tp", TAKE_PROFIT_PERCENT))
            STOP_LOSS_PERCENT = float(cmd_data.get("sl", STOP_LOSS_PERCENT))
            MAX_POSITION_SIZE = 5.0 * float(cmd_data.get("leverage", 1))

            if isinstance(self.engine, PaperTradingEngine):
                self.engine.max_trades_per_day = int(cmd_data.get("max_trades_per_day", self.engine.max_trades_per_day))
                self.engine.daily_tp_target_percent = float(cmd_data.get("daily_tp_target_percent", self.engine.daily_tp_target_percent))
                self.engine._roll_day_if_needed()
                logger.info(f"✅ Demo limits: max_trades_per_day={self.engine.max_trades_per_day}, daily_tp_target_percent={self.engine.daily_tp_target_percent}")

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

            self.strategy["min_ml_confidence"] = max(0.0, min(1.0, _as_float(cmd_data.get("min_ml_confidence"), self.strategy.get("min_ml_confidence", 0.70))))
            self.strategy["block_weak_signals"] = _as_bool(cmd_data.get("block_weak_signals"), self.strategy.get("block_weak_signals", True))
            self.strategy["cooldown_bars"] = max(0, _as_int(cmd_data.get("cooldown_bars"), self.strategy.get("cooldown_bars", 3)))
            self.strategy["use_atr_risk"] = _as_bool(cmd_data.get("use_atr_risk"), self.strategy.get("use_atr_risk", True))
            self.strategy["atr_sl_mult"] = max(0.1, _as_float(cmd_data.get("atr_sl_mult"), self.strategy.get("atr_sl_mult", 1.5)))
            self.strategy["atr_tp_mult"] = max(0.1, _as_float(cmd_data.get("atr_tp_mult"), self.strategy.get("atr_tp_mult", 2.5)))

            mode = (cmd_data.get("strategy_mode") or self.strategy_mode or "classic").strip().lower()
            if mode in ("classic", "pro"):
                self.strategy_mode = mode

            logger.info(
                "✅ Filters updated: "
                f"min_ml_confidence={self.strategy['min_ml_confidence']}, "
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

            # ML Prediction
            ml_pred = self.ml_loader.predict(df)
            if not ml_pred:
                logger.warning(f"⚠️ {symbol}: ML prediction failed")
                return

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

            # Only enter if no position
            if symbol not in self.engine.positions:
                min_conf = float(self.strategy.get("min_ml_confidence", 0.70))
                ml_conf = float((ml_pred or {}).get("confidence") or 0.0)
                if ml_conf < min_conf:
                    logger.info(f"⛔ {symbol}: ML confidence too low ({ml_conf:.1%} < {min_conf:.1%})")
                    return

                if self._in_cooldown(symbol, df):
                    logger.info(f"⏳ {symbol}: Cooldown active")
                    return

                logger.info(f"🔍 {symbol}: Requesting DeepSeek decision...")
                mode = (getattr(self, "strategy_mode", None) or STRATEGY_MODE or "classic").strip().lower()
                if mode == "pro":
                    decision = self.deepseek.get_pro_decision(symbol, df, ml_pred)
                else:
                    decision = self.deepseek.get_decision(symbol, df, ml_pred)

                decision["strategy_mode"] = mode

                atr = None
                try:
                    atr = float(df.iloc[-1].get("atr", 0.0))
                except Exception:
                    atr = None
                decision = self._decorate_decision_with_risk(symbol, float(current_price), ml_pred, decision, atr=atr)

                if bool(self.strategy.get("block_weak_signals", True)) and str(decision.get("signal_strength") or "").lower() == "weak":
                    logger.info(f"⛔ {symbol}: Weak signal blocked")
                    return

                if decision.get('trade_decision') == 'YES':
                    logger.info(f"✅ DeepSeek signal: {decision.get('side','long').upper()} {symbol} | TP={decision.get('tp_pct')}% SL={decision.get('sl_pct')}% | strength={decision.get('signal_strength')}")
                    position = self.engine.execute_entry(symbol, decision, current_price)
                    if position:
                        try:
                            self.last_action_ts[symbol] = df.index[-1]
                        except Exception:
                            pass

                        await self.notifier.send_signal("ENTRY", {
                            'asset': symbol,
                            'side': position.get('side', decision.get('side', 'long')),
                            'ml_forecast_pct': decision.get('ml_forecast_pct', '0%'),
                            'trade_decision': 'YES',
                            'analysis': decision.get('reasoning_short', ''),
                            'tp_pct': decision.get('tp_pct'),
                            'sl_pct': decision.get('sl_pct'),
                            'signal_strength': decision.get('signal_strength'),
                            'strategy_mode': decision.get('strategy_mode')
                        })
                    else:
                        logger.warning(f"❌ {symbol}: Entry execution failed (insufficient balance or engine error)")
                else:
                    logger.info(f"⏳ {symbol}: DeepSeek said NO. Reason: {decision.get('reasoning_short', 'No reason provided')}")
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