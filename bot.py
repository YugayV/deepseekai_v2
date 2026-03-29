"""
EURUSD AI Trading Bot with DeepSeek
Inspired by: https://github.com/tot-gromov/llm-deepseek-trading
Optimized models with Alligator + Fractals
"""

import os
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import time
import json
import asyncio
import pandas as pd
import numpy as np
import yfinance as yf
import talib
import joblib
from datetime import datetime
from dotenv import load_dotenv
import openai

load_dotenv()

# ============================================
# CONFIGURATION
# ============================================
RAILWAY = os.getenv("RAILWAY", "false").lower() == "true"
PORT = int(os.getenv("PORT", 8000))

# Assets
FOREX_SYMBOLS = os.getenv("SYMBOLS", "EURUSD=X,GBPUSD=X,USDJPY=X").split(',')
CRYPTO_SYMBOLS = os.getenv("CRYPTO_SYMBOLS", "").split(',') if os.getenv("CRYPTO_SYMBOLS") else []
ALL_SYMBOLS = FOREX_SYMBOLS + CRYPTO_SYMBOLS

# Trading
PAPER_CAPITAL = float(os.getenv("PAPER_START_CAPITAL", 10000))
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", 5.0))
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 2.0))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", 4.0))

# API Keys
DEEPSEEK_API_KEY = os.getenv("OPENROUTER_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_SIGNALS_CHAT_ID = os.getenv("TELEGRAM_SIGNALS_CHAT_ID", TELEGRAM_CHAT_ID)

# Paths
MODEL_PATH = "models/voting_ensemble.pkl"
SCALER_PATH = "models/feature_scaler.pkl"
METADATA_PATH = "models/model_metadata.json"
DATA_DIR = os.getenv("TRADEBOT_DATA_DIR", "data")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create data directory
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


# ============================================
# HEALTH CHECK SERVER (for Railway)
# ============================================
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, format, *args): return

def start_health_server():
    try:
        server = HTTPServer(('0.0.0.0', PORT), HealthHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        logger.info(f"✅ Health check server started on port {PORT}")
    except Exception as e:
        logger.error(f"❌ Failed to start health server: {e}")

if RAILWAY:
    start_health_server()


# ============================================
# TELEGRAM NOTIFIER (с поддержкой группы)
# ============================================
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ... существующий код ...

class TelegramNotifier:
    def __init__(self, token=None, group_id=None, admin_chat_id=None):
        self.bot_token = token
        self.group_id = group_id
        self.admin_chat_id = admin_chat_id
        self.application = None
        self.bot = None

    async def initialize(self):
        if self.bot_token:
            try:
                from telegram import Bot
                from telegram.ext import Application, CommandHandler
                self.application = Application.builder().token(self.bot_token).build()
                self.bot = self.application.bot
                
                self.application.add_handler(CommandHandler("start", self._cmd_start))
                self.application.add_handler(CommandHandler("status", self._cmd_status))
                self.application.add_handler(CommandHandler("pairs", self._cmd_pairs))
                
                await self.application.initialize()
                await self.application.start()
                if self.application.updater:
                    await self.application.updater.start_polling()
                logger.info("✅ Telegram bot initialized with command handlers")
            except Exception as e:
                logger.error(f"Telegram init error: {e}")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = "🚀 *AI Trading Bot is Active*\n\nCommands:\n/status - Portfolio summary\n/pairs - Tracked symbols"
        await update.message.reply_text(text, parse_mode='Markdown')

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("📊 Generating report... use /pairs for symbols.")

    async def _cmd_pairs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        pairs_text = "🎯 *Tracked Symbols:*\n\n"
        pairs_text += "\n".join([f"• `{s}`" for s in ALL_SYMBOLS])
        await update.message.reply_text(pairs_text, parse_mode='Markdown')

    async def send_message(self, text, chat_id=None):
        if not self.bot: return
        target_chat = chat_id or self.group_id
        if not target_chat: return
        try:
            await self.bot.send_message(chat_id=target_chat, text=text, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
# ... существующий код ...
        """Отправка сообщения в указанный чат"""
        if not self.bot:
            return
        
        # Если chat_id не указан, используем группу
        target_chat = chat_id or self.group_id
        if not target_chat:
            return
            
        try:
            await self.bot.send_message(
                chat_id=target_chat,
                text=text,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )
        except Exception as e:
            logger.error(f"Telegram send error to {target_chat}: {e}")

    async def send_signal(self, signal_type, data):
        """Отправка сигнала в группу с детальным анализом"""
        if not self.bot or not self.group_id:
            return

        if signal_type == "ENTRY":
            emoji = "🟢" if data.get('side') == 'long' else "🔴"
            text = f"""
{emoji} *MARKET ANALYSIS: {data['asset']}*

*Current Setup:*
{data.get('analysis', 'No detailed analysis provided.')}

*ML Signal:* {data.get('ml_regime', 'N/A')} (confidence: {data.get('confidence', 0):.1%})

*Recommendation:*
{data.get('recommendation', 'Wait for better setup.')}

*Risk Note:*
{data.get('risk_note', 'Standard position size.')}
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
        """Отправка ошибок админу (личное сообщение)"""
        if self.bot and self.admin_chat_id:
            text = f"⚠️ *ERROR*\n\n{error_msg[:500]}"
            await self.send_message(text, self.admin_chat_id)

    async def send_daily_summary(self, stats):
        """Отправка подробного отчета о портфеле и позициях"""
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
        """Отправка сообщения о запуске бота"""
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

        # Extract features
        features = pd.DataFrame(index=df.index)
        for col in self.feature_cols:
            if col in df.columns:
                features[col] = df[col]
            else:
                features[col] = 0

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
    df['ema_8'] = talib.EMA(df['close'], timeperiod=8)
    df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
    df['ema_50'] = talib.EMA(df['close'], timeperiod=50)

    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])

    # RSI
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)

    # ATR
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    # Bollinger
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])

    # Volume
    df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # EMA Cross
    df['ema_cross'] = 0
    df.loc[(df['ema_8'] > df['ema_21']) & (df['ema_8'].shift(1) <= df['ema_21'].shift(1)), 'ema_cross'] = 1
    df.loc[(df['ema_8'] < df['ema_21']) & (df['ema_8'].shift(1) >= df['ema_21'].shift(1)), 'ema_cross'] = -1

    # ALLIGATOR
    df['jaw'] = talib.SMA(df['close'], timeperiod=13).shift(8)
    df['teeth'] = talib.SMA(df['close'], timeperiod=8).shift(5)
    df['lips'] = talib.SMA(df['close'], timeperiod=5).shift(3)

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

        prompt = f"""
Act as a professional trader analyzing {symbol}. 
CONTEXT:
- PRICE: {latest['close']:.5f} (Change: {latest['returns']:.2f}%)
- VOLATILITY: {latest['volatility']:.4f}
- RSI: {latest['rsi']:.1f} ({'OVERBOUGHT' if latest['rsi'] > 70 else 'OVERSOLD' if latest['rsi'] < 30 else 'NEUTRAL'})
- MACD Hist: {latest['macd_hist']:.5f} (Trend: {'Up' if latest['macd_hist'] > prev['macd_hist'] else 'Down'})

INDICATORS:
- ALLIGATOR: {'BULLISH' if latest['alligator_bullish'] else 'BEARISH' if latest['alligator_bearish'] else 'ASLEEP' if latest['alligator_asleep'] else 'NEUTRAL'}
- FRACTALS: Bull={latest['fractal_bullish']}, Bear={latest['fractal_bearish']}
- ML REGIME: {ml_prediction['regime_name'].upper()} (Conf: {ml_prediction['confidence']:.1%})

DECISION CRITERIA:
1. Entry ONLY if ML confidence > 60% and technicals align.
2. Consider Stop Loss at {STOP_LOSS_PERCENT}% and Take Profit at {TAKE_PROFIT_PERCENT}%.

Respond ONLY with valid JSON:
{{"action": "entry/hold/close", "side": "long/short", "confidence": 0-100, "reasoning": "Be specific why now."}}
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
# PAPER TRADING ENGINE
# ============================================
class PaperTradingEngine:
    def __init__(self, initial_capital=PAPER_CAPITAL):
        self.balance = initial_capital
        self.positions = {}
        self.trades = []
        self.daily_trades = 0

    def execute_entry(self, symbol, decision, price):
        if decision.get('action') != 'entry':
            return None

        side = decision.get('side', 'long')
        stop = decision.get('stop_loss', price * (1 - STOP_LOSS_PERCENT/100 if side == 'long' else 1 + STOP_LOSS_PERCENT/100))
        target = decision.get('take_profit', price * (1 + TAKE_PROFIT_PERCENT/100 if side == 'long' else 1 - TAKE_PROFIT_PERCENT/100))
        confidence = decision.get('confidence', 50) / 100

        risk_amount = self.balance * (MAX_POSITION_SIZE / 100) * confidence
        risk_per_unit = abs(price - stop)
        if risk_per_unit <= 0:
            return None

        size = risk_amount / risk_per_unit
        size = min(size, self.balance * 0.3 / price)

        if size <= 0:
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
            pos['exit_date'] = datetime.now()

            self.trades.append(pos)
            del self.positions[symbol]

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
        self.engine = PaperTradingEngine()
        
        # Telegram: групповой чат для сигналов, личный для ошибок
        self.notifier = TelegramNotifier(
            token=TELEGRAM_TOKEN,
            group_id=os.getenv("TELEGRAM_GROUP_ID"),      # ID группы
            admin_chat_id=os.getenv("TELEGRAM_CHAT_ID")   # Ваш личный ID
        )
        self.running = True
    
    async def _check_dashboard_commands(self):
        cmd_path = "data/bot_command.json"
        if os.path.exists(cmd_path):
            try:
                with open(cmd_path, "r") as f:
                    cmd_data = json.load(f)
                
                cmd = cmd_data.get("command")
                if cmd == "start_all":
                    logger.info("🚀 Dashboard command: START ALL PAIRS")
                    # Update settings from dashboard
                    global TAKE_PROFIT_PERCENT, STOP_LOSS_PERCENT, MAX_POSITION_SIZE
                    TAKE_PROFIT_PERCENT = cmd_data.get("tp", TAKE_PROFIT_PERCENT)
                    STOP_LOSS_PERCENT = cmd_data.get("sl", STOP_LOSS_PERCENT)
                    # Use leverage to adjust position size
                    MAX_POSITION_SIZE = 5.0 * cmd_data.get("leverage", 1)
                elif cmd == "start_single":
                    symbol = cmd_data.get("symbol")
                    logger.info(f"🚀 Dashboard command: START SINGLE PAIR {symbol}")
                    # Force process only this symbol
                    await self.process_symbol(symbol)
                elif cmd == "stop_all":
                    logger.info("🛑 Dashboard command: STOP ALL")
                    self.engine.positions = {} # Emergency close all
                
                # Delete command after processing
                os.remove(cmd_path)
            except Exception as e:
                logger.error(f"Error processing dashboard command: {e}")

    async def _send_startup(self):
        await asyncio.sleep(2)
        await self.notifier.send_startup_message()

    async def process_symbol(self, symbol):
        try:
            df = fetch_data(symbol)
            if df is None or len(df) < 50:
                return

            current_price = df['close'].iloc[-1]

            # ML Prediction
            ml_pred = self.ml_loader.predict(df)
            if not ml_pred:
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
                decision = self.deepseek.get_decision(symbol, df, ml_pred)

                if decision.get('action') == 'entry':
                    position = self.engine.execute_entry(symbol, decision, current_price)
                    if position:
                        await self.notifier.send_signal("ENTRY", {
                            'asset': symbol,
                            'side': position['side'],
                            'entry_price': position['entry_price'],
                            'ml_regime': ml_pred['regime_name'],
                            'confidence': position['confidence'],
                            'analysis': decision.get('analysis', ''),
                            'recommendation': decision.get('recommendation', ''),
                            'risk_note': decision.get('risk_note', '')
                        })

            # Log
            logger.info(f"{symbol}: {current_price:.5f} | ML: {ml_pred['regime_name']} ({ml_pred['confidence']:.1%}) | AI: {decision.get('action', 'hold')}")

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

        # Send summary
        current_prices = {}
        positions_details = []
        for symbol in ALL_SYMBOLS:
            df = fetch_data(symbol)
            if df is not None:
                price = df['close'].iloc[-1]
                current_prices[symbol] = price
                
                if symbol in self.engine.positions:
                    pos = self.engine.positions[symbol]
                    pnl_usd = (price - pos['entry_price']) * pos['size'] if pos['side'] == 'long' else (pos['entry_price'] - price) * pos['size']
                    pnl_pct = (pnl_usd / (pos['size'] * pos['entry_price'])) * 100
                    positions_details.append({
                        'symbol': symbol,
                        'side': pos['side'],
                        'entry_price': pos['entry_price'],
                        'current_price': price,
                        'pnl_usd': pnl_usd,
                        'pnl_percent': pnl_pct
                    })

        state = self.engine.get_state(current_prices)
        await self.notifier.send_daily_summary({
            'balance': state['balance'],
            'equity': state['equity'],
            'pnl': state['pnl'],
            'trades_today': self.engine.daily_trades,
            'open_positions': state['positions'],
            'positions_details': positions_details
        })

        logger.info(f"Portfolio: Balance=${state['balance']:.2f}, Equity=${state['equity']:.2f}, PnL={state['pnl']:.2f}%")
        logger.info("Cycle completed. Next in 10 minutes.")

    async def run(self):
        # Инициализируем Telegram перед циклом
        await self.notifier.initialize()
        # Теперь безопасно отправляем сообщение о запуске
        await self._send_startup()
        
        while self.running:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error(f"Cycle error: {e}")
            await asyncio.sleep(10800)  # 3 hours (3 * 60 * 60)


# ============================================
# ENTRY POINT
# ============================================
async def main():
    try:
        bot = TradingBot()
        await bot.run()
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
        logger.error(f"FATAL ERROR: {e}")
        if os.getenv("RAILWAY", "false").lower() == "true":
            while True:
                time.sleep(10)