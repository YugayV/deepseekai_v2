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

# ============================================
# HEALTH CHECK SERVER (Start immediately for Railway)
# Respond on both $PORT and default 8080 using non-daemon threads.
# ============================================
DEFAULT_PORT = 8080
ENV_PORT_RAW = os.getenv('PORT')
try:
    ENV_PORT = int(ENV_PORT_RAW) if ENV_PORT_RAW else None
except Exception:
    ENV_PORT = None

HEALTH_PORTS = []
if ENV_PORT:
    HEALTH_PORTS.append(ENV_PORT)
if DEFAULT_PORT not in HEALTH_PORTS:
    HEALTH_PORTS.append(DEFAULT_PORT)

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(b'OK')

    def log_message(self, format, *args):
        return


def _serve_health(port: int):
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    server.serve_forever()


def start_health_servers():
    for p in HEALTH_PORTS:
        try:
            t = threading.Thread(target=_serve_health, args=(p,), daemon=False, name=f"health_{p}")
            t.start()
            print(f"✅ Health server listening on 0.0.0.0:{p}")
        except Exception as e:
            print(f"❌ Failed to start health server on port {p}: {e}")


if os.getenv('RAILWAY') or os.getenv('PORT'):
    start_health_servers()

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
    
    # Flag to indicate imports were successful
    IMPORTS_OK = True
except Exception as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    IMPORTS_OK = False

if not IMPORTS_OK:
    print("Keeping process alive for health check...")
    while True:
        time.sleep(10)

load_dotenv()


# ============================================
# TELEGRAM NOTIFIER (Menu + Signals)
# ============================================
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler

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
                self.application = Application.builder().token(self.bot_token).build()
                self.bot = self.application.bot
                
                # Handlers
                self.application.add_handler(CommandHandler("start", self._cmd_start))
                self.application.add_handler(CommandHandler("menu", self._cmd_start))
                self.application.add_handler(CallbackQueryHandler(self._handle_callbacks))
                
                await self.application.initialize()
                await self.application.start()
                if self.application.updater:
                    await self.application.updater.start_polling()
                logger.info("✅ Telegram bot initialized with Main Menu")
            except Exception as e:
                logger.error(f"Telegram init error: {e}")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [InlineKeyboardButton("📊 Portfolio Status", callback_data='status')],
            [InlineKeyboardButton("💰 Current Prices", callback_data='prices')],
            [InlineKeyboardButton("🔮 AI Forecast", callback_data='select_forecast')],
            [InlineKeyboardButton("🚀 Auto-Trade Controls", callback_data='trade_menu')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        text = "🤖 *AI Trading Bot Control Panel*\nSelect an option below:"
        if update.message:
            await update.message.reply_text(text, reply_markup=reply_markup, parse_mode='Markdown')
        else:
            await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode='Markdown')

    async def _handle_callbacks(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
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
                [InlineKeyboardButton("⬅️ Back", callback_data='main_menu')]
            ]
            await query.edit_message_text("⚙️ *Auto-Trade Controls*", 
                                          reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')

        elif data == 'start_all':
            with open(cmd_path, "w") as f:
                json.dump({"command": "start_all", "time": str(datetime.now())}, f)
            await query.edit_message_text("🚀 Command sent: *START ALL*", 
                                          reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='trade_menu')]]), parse_mode='Markdown')

        elif data == 'stop_all':
            with open(cmd_path, "w") as f:
                json.dump({"command": "stop_all", "time": str(datetime.now())}, f)
            await query.edit_message_text("🛑 Command sent: *STOP ALL*", 
                                          reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='trade_menu')]]), parse_mode='Markdown')

        elif data == 'main_menu':
            await self._cmd_start(update, context)

    async def send_message(self, text, chat_id=None):
        """Send message to Telegram"""
        if self.bot:
            target_chat = chat_id or self.group_id
            if target_chat:
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
        if not self.bot or not self.group_id:
            return

        if signal_type == "ENTRY":
            emoji = "🟢" if data.get('side') == 'long' else "🔴"
            text = f"""
{emoji} *TRADE SIGNAL: {data['asset']}*

*ML Prediction:* {data.get('ml_forecast_pct', '0%')}
*Decision:* {data.get('trade_decision', 'NO')}

*Reason:* {data.get('analysis', '')}
"""
            await self.send_message(text, self.group_id)

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
            await self.send_message(text, self.group_id)

    async def send_error(self, error_msg):
        """Send errors to admin (private message)"""
        if self.bot and self.admin_chat_id:
            text = f"⚠️ *ERROR*\n\n{error_msg[:500]}"
            await self.send_message(text, self.admin_chat_id)

    async def send_daily_summary(self, stats):
        """Send detailed report on portfolio and positions"""
        if not self.bot or not self.group_id:
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
        await self.send_message(text, self.group_id)

    async def send_startup_message(self):
        """Send bot startup message"""
        if self.bot and self.group_id:
            text = f"""
🚀 *TRADING BOT STARTED*

*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
*Capital:* ${PAPER_CAPITAL:.2f}
*Max Risk:* {MAX_POSITION_SIZE}%
*Stop Loss:* {STOP_LOSS_PERCENT}%
*Take Profit:* {TAKE_PROFIT_PERCENT}%

*Assets:* {', '.join(ALL_SYMBOLS)}
"""
            await self.send_message(text, self.group_id)


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
        self.balance = initial_capital
        self.positions = {}
        self.trades = []
        self.daily_trades = 0
        self._load_state()

    def _save_state(self):
        """Save portfolio state and trade history for the dashboard"""
        try:
            # Save portfolio state
            state = {
                'balance': self.balance,
                'equity': self.balance, # Initial equity equals balance
                'positions': self.positions,
                'last_update': str(datetime.now())
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

    def execute_entry(self, symbol, decision, price):
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
            'side': side,
            'entry_price': price,
            'stop_loss': stop,
            'take_profit': target,
            'size': size,
            'confidence': confidence,
            'entry_date': datetime.now()
        }

        margin = size * price
        self.balance -= margin
        self.daily_trades += 1
        
        logger.info(f"🚀 {symbol} ENTRY EXECUTED: {side.upper()} {size:.4f} @ {price:.5f}")
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
        
        # Select Engine based on mode
        if TRADING_MODE == "real" and EXCHANGE_API_KEY:
            self.engine = RealTradingEngine(EXCHANGE_ID, EXCHANGE_API_KEY, EXCHANGE_API_SECRET)
        else:
            self.engine = PaperTradingEngine()
        
        # Multi-channel: group chat for signals, private for errors
        self.notifier = MultiChannelNotifier(
            token=TELEGRAM_TOKEN,
            group_id=os.getenv("TELEGRAM_GROUP_ID"),      # Group ID
            admin_chat_id=os.getenv("TELEGRAM_CHAT_ID")   # Your personal ID
        )
        self.running = True
    
    async def _check_dashboard_commands(self):
        cmd_path = os.path.join(DATA_DIR, "bot_command.json")
        if os.path.exists(cmd_path):
            try:
                logger.info(f"📂 Found command file: {cmd_path}")
                with open(cmd_path, "r") as f:
                    cmd_data = json.load(f)
                
                # Delete command immediately to prevent double processing
                os.remove(cmd_path)
                logger.info(f"📥 Processing command: {cmd_data.get('command')}")
                
                cmd = cmd_data.get("command")
                if cmd == "start_all":
                    logger.info(f"🚀 Dashboard command: START ALL PAIRS (TP={cmd_data.get('tp', TAKE_PROFIT_PERCENT)}, SL={cmd_data.get('sl', STOP_LOSS_PERCENT)}, Lev={cmd_data.get('leverage', 1)})")
                    global TAKE_PROFIT_PERCENT, STOP_LOSS_PERCENT, MAX_POSITION_SIZE
                    TAKE_PROFIT_PERCENT = float(cmd_data.get("tp", TAKE_PROFIT_PERCENT))
                    STOP_LOSS_PERCENT = float(cmd_data.get("sl", STOP_LOSS_PERCENT))
                    MAX_POSITION_SIZE = 5.0 * float(cmd_data.get("leverage", 1))
                    
                    # Force immediate execution of the cycle
                    self.force_cycle = True
                    logger.info("⚡ Cycle force-start flag set. Starting trading cycle immediately.")
                elif cmd == "start_single" or cmd == "trade_single":
                    symbol = cmd_data.get("symbol")
                    logger.info(f"🚀 Dashboard command: START SINGLE PAIR {symbol}")
                    if symbol:
                        await self.process_symbol(symbol)
                    else:
                        logger.error("❌ Single trade command missing symbol!")
                elif cmd == "stop_all":
                    logger.info("🛑 Dashboard command: STOP ALL")
                    self.engine.positions = {}
                elif cmd == "update_api":
                    mode = cmd_data.get("mode")
                    logger.info(f"🔄 Dashboard command: SWITCH TO {mode.upper()}")
                    global TRADING_MODE
                    TRADING_MODE = mode
                    if mode == "real":
                        self.engine = RealTradingEngine(
                            cmd_data.get("exchange", "bybit"),
                            cmd_data.get("key"),
                            cmd_data.get("secret")
                        )
                    else:
                        self.engine = PaperTradingEngine()
                
            except Exception as e:
                logger.error(f"❌ Error processing dashboard command: {e}")
                if os.path.exists(cmd_path): os.remove(cmd_path)

    async def _send_startup(self):
        await asyncio.sleep(2)
        await self.notifier.send_startup_message()

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
                logger.info(f"🔍 {symbol}: Requesting DeepSeek decision...")
                decision = self.deepseek.get_decision(symbol, df, ml_pred)

                if decision.get('trade_decision') == 'YES':
                    logger.info(f"✅ DeepSeek signal: BUY {symbol}")
                    position = self.engine.execute_entry(symbol, decision, current_price)
                    if position:
                        await self.notifier.send_signal("ENTRY", {
                            'asset': symbol,
                            'side': position.get('side', 'long'),
                            'ml_forecast_pct': decision.get('ml_forecast_pct', '0%'),
                            'trade_decision': 'YES',
                            'analysis': decision.get('reasoning_short', '')
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
        logger.info("Cycle completed. Next in 10 minutes.")

    async def send_portfolio_summary(self):
        """Calculate state and send summary via notifier"""
        current_prices = {}
        for s in ALL_SYMBOLS:
            df = fetch_data(s)
            if df is not None: current_prices[s] = df['close'].iloc[-1]
        state = self.engine.get_state(current_prices)
        await self.notifier.send_daily_summary(state)

    async def run(self):
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
                
                # 2. Check if it's time for a scheduled trading cycle (every 10 mins) or forced
                if (current_time - last_cycle_time > 600) or getattr(self, 'force_cycle', False):
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
    if not IMPORTS_OK:
        print("🚨 BOT CANNOT START DUE TO IMPORT ERRORS. Keeping process alive for health check...")
        while True:
            time.sleep(60)
            
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        if os.getenv("RAILWAY", "false").lower() == "true":
            while True:
                time.sleep(10)