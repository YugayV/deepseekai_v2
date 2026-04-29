#!/usr/bin/env python3
"""
REAL-TIME TRADING SIMULATION BOT
=================================
Uses yfinance for real-time data
Simulates trading 24/7
Checks signals every 15 minutes
For Vitaliy Yugay | Optimus AI | 2026-04-28
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
import requests
import json
import time
from datetime import datetime

# ============== CONFIGURATION ==============
TELEGRAM_BOT_TOKEN = (os.getenv('TELEGRAM_BOT_TOKEN') or os.getenv('TELEGRAM_TOKEN') or '').strip()
TELEGRAM_CHAT_ID = (os.getenv('TELEGRAM_CHAT_ID') or '').strip()

INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', 1000))
LEVERAGE = float(os.getenv('LEVERAGE', 3))
STOP_LOSS = float(os.getenv('STOP_LOSS', 0.02))
TAKE_PROFIT = float(os.getenv('TAKE_PROFIT', 0.03))
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', 900))
# ==========================================


def send_telegram(message, photo_path=None):
    """Send message/photo to Telegram"""
    token = (TELEGRAM_BOT_TOKEN or '').strip()
    chat_id = (TELEGRAM_CHAT_ID or '').strip()

    if not token or not chat_id:
        return False

    base_url = f'https://api.telegram.org/bot{token}'

    try:
        if photo_path:
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {'chat_id': chat_id, 'caption': message}
                requests.post(f'{base_url}/sendPhoto', files=files, data=data, timeout=15)
        else:
            data = {'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}
            requests.post(f'{base_url}/sendMessage', data=data, timeout=15)
        return True
    except Exception:
        return False


def get_close(df):
    if df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            c = df['Close']
            return c.iloc[:, 0] if c.ndim > 1 else c
    return df['Close'] if 'Close' in df.columns else df.iloc[:, 0]


def compute_signal(d, g, e, idx):
    """DXY-EUR/USD correlation signal"""
    if idx < 20:
        return 0
    
    d_ret_3 = d[idx] / d[idx-3] - 1 if idx >= 3 else 0
    g_ret_3 = g[idx] / g[idx-3] - 1 if idx >= 3 else 0
    e_ma10 = np.mean(e[idx-10:idx]) if idx >= 10 else e[idx]
    
    score = 0
    
    if d_ret_3 < -0.01:
        score += 3
    elif d_ret_3 < -0.005:
        score += 2
    elif d_ret_3 < 0:
        score += 1
    
    if d_ret_3 > 0.01:
        score -= 3
    elif d_ret_3 > 0.005:
        score -= 2
    elif d_ret_3 > 0:
        score -= 1
    
    if g_ret_3 > 0:
        score += 1
    else:
        score -= 1
    
    if e[idx] > e_ma10 and score > 0:
        score += 1
    elif e[idx] < e_ma10 and score < 0:
        score -= 1
    
    return score


def get_realtime_price(ticker):
    """Get real-time price from yfinance"""
    try:
        data = yf.Ticker(ticker).history(period='1d')
        if not data.empty:
            return data['Close'].iloc[-1]
    except:
        pass
    return None


class RealTimeTrader:
    """Real-time trading simulator"""
    
    def __init__(self):
        self.balance = INITIAL_BALANCE
        self.initial_balance = INITIAL_BALANCE
        self.position = 0
        self.entry_price = 0
        self.entry_idx = 0
        self.trades = []
        self.equity_history = []
        self.wins = 0
        self.losses = 0
        self.last_check = None
        self.start_time = datetime.now()
        
    def check_and_trade(self, d, g, e, idx):
        """Check signal and execute trade"""
        if self.position != 0:
            current_price = e[idx]
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            if self.position == -1:
                pnl_pct = -pnl_pct
            
            hold_days = idx - self.entry_idx
            
            close = False
            reason = None
            
            if pnl_pct >= TAKE_PROFIT:
                close = True
                reason = 'TAKE_PROFIT'
            elif pnl_pct <= -STOP_LOSS:
                close = True
                reason = 'STOP_LOSS'
            elif hold_days >= 15:
                close = True
                reason = 'MAX_HOLD'
            
            if close:
                pnl_value = self.balance * pnl_pct * LEVERAGE
                self.balance += pnl_value
                
                self.trades.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'LONG' if self.position == 1 else 'SHORT',
                    'entry': self.entry_price,
                    'exit': current_price,
                    'pnl_pct': pnl_pct * 100,
                    'pnl_value': pnl_value,
                    'reason': reason
                })
                
                if pnl_value > 0:
                    self.wins += 1
                else:
                    self.losses += 1
                
                self.position = 0
                return {'action': 'close', 'pnl': pnl_value, 'reason': reason, 'price': current_price}
        
        # Check entry
        if self.position == 0:
            signal = compute_signal(d, g, e, idx)
            
            if signal >= 4:
                self.position = 1
                self.entry_price = e[idx]
                self.entry_idx = idx
                return {'action': 'open', 'direction': 'LONG', 'price': e[idx], 'signal': signal}
            elif signal <= -4:
                self.position = -1
                self.entry_price = e[idx]
                self.entry_idx = idx
                return {'action': 'open', 'direction': 'SHORT', 'price': e[idx], 'signal': signal}
        
        return None
    
    def get_status(self, current_price=None):
        """Get current status"""
        total_trades = len(self.trades)
        win_rate = self.wins / total_trades * 100 if total_trades > 0 else 0
        return_pct = (self.balance / self.initial_balance - 1) * 100
        
        uptime = (datetime.now() - self.start_time).total_seconds() / 3600
        
        msg = f"""
🤖 *REAL-TIME TRADING BOT*
{'='*35}

⏰ Uptime: {uptime:.1f} hours
💰 Balance: ${self.balance:.2f}
📊 Return: {return_pct:+.2f}%
📈 Win Rate: {win_rate:.1f}%

📋 Trades: {total_trades}
🟢 Wins: {self.wins} | 🔴 Losses: {self.losses}

📊 Position: {'OPEN' if self.position != 0 else 'NONE'}
"""
        
        if self.position != 0 and current_price:
            pnl = (current_price - self.entry_price) / self.entry_price * 100
            if self.position == -1:
                pnl = -pnl
            msg += f"Entry: ${self.entry_price:.5f}\n"
            msg += f"Current: ${current_price:.5f}\n"
            msg += f"PnL: {pnl:+.2f}%"
        
        return msg
    
    def plot_equity(self):
        """Plot equity curve"""
        if len(self.equity_history) < 2:
            return None
        
        df = pd.DataFrame(self.equity_history)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['equity'], 'b-', linewidth=2)
        ax.axhline(y=self.initial_balance, color='gray', linestyle='--', alpha=0.5)
        ax.fill_between(df.index, self.initial_balance, df['equity'], 
                       where=df['equity'] >= self.initial_balance, color='green', alpha=0.3)
        ax.fill_between(df.index, self.initial_balance, df['equity'], 
                       where=df['equity'] < self.initial_balance, color='red', alpha=0.3)
        ax.set_title('🤖 Real-Time Trading - Equity Curve', fontweight='bold', fontsize=14)
        ax.set_ylabel('Equity ($)')
        ax.set_xlabel('Check #')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs('data', exist_ok=True)
        path = os.path.join('data', 'realtime_equity.png')
        plt.savefig(path, dpi=150)
        plt.close()
        
        return path


def main():
    print("=" * 60)
    print("🤖 REAL-TIME TRADING SIMULATION BOT")
    print("24/7 Trading with yfinance")
    print("=" * 60)
    
    send_telegram("🤖 *Real-Time Trading Bot Started!*\n\n"
                  f"💰 Balance: ${INITIAL_BALANCE}\n"
                  f"📊 Leverage: {LEVERAGE}x\n"
                  f"⏱️ Check interval: {CHECK_INTERVAL//60} min\n\n"
                  "Strategy: DXY-EUR/USD Correlation\n\n"
                  "Bot will check signals every 15 minutes and send updates!")
    
    trader = RealTimeTrader()
    
    # Initial data fetch
    print("\n📥 Fetching initial data...")
    dxy = yf.download('UUP', period='2y', interval='1d', progress=False)
    eurusd = yf.download('EURUSD=X', period='2y', interval='1d', progress=False)
    gold = yf.download('GLD', period='2y', interval='1d', progress=False)
    
    d = get_close(dxy).values
    e = get_close(eurusd).values
    g = get_close(gold).values
    
    print(f"   DXY: {len(d)}, EUR/USD: {len(e)}, Gold: {len(g)}")
    
    # Find common length
    min_len = min(len(d), len(e), len(g))
    idx = min_len - 1
    
    # Check initial signal
    signal = compute_signal(d, g, e, idx)
    current_price = e[idx]
    
    print(f"\n📊 Current EUR/USD: ${current_price:.5f}")
    print(f"📊 Signal: {signal}")
    
    send_telegram(f"📊 *Initial Check*\n\nEUR/USD: ${current_price:.5f}\nSignal: {signal}\n\nWaiting for next check...")
    
    check_count = 0
    
    try:
        while True:
            check_count += 1
            print(f"\n{'='*60}")
            print(f"🔄 Check #{check_count} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Refresh data
            dxy = yf.download('UUP', period='2y', interval='1d', progress=False)
            eurusd = yf.download('EURUSD=X', period='2y', interval='1d', progress=False)
            gold = yf.download('GLD', period='2y', interval='1d', progress=False)
            
            d = get_close(dxy).values
            e = get_close(eurusd).values
            g = get_close(gold).values
            
            min_len = min(len(d), len(e), len(g))
            idx = min_len - 1
            current_price = e[idx]
            
            # Check for trade
            result = trader.check_and_trade(d, g, e, idx)
            
            if result:
                if result['action'] == 'open':
                    emoji = '📢'
                    print(f"{emoji} OPEN {result['direction']} @ {result['price']:.5f} (signal={result['signal']})")
                    send_telegram(f"📢 *NEW TRADE*\n\n"
                                 f"Type: {result['direction']}\n"
                                 f"Price: ${result['price']:.5f}\n"
                                 f"Signal: {result['signal']}")
                elif result['action'] == 'close':
                    emoji = '🟢' if result['pnl'] > 0 else '🔴'
                    print(f"{emoji} CLOSE {result['reason']}: ${result['pnl']:.2f}")
                    send_telegram(f"{emoji} *TRADE CLOSED*\n\n"
                                 f"Reason: {result['reason']}\n"
                                 f"PnL: ${result['pnl']:.2f}\n"
                                 f"Balance: ${trader.balance:.2f}")
            
            # Update equity history
            if trader.position != 0:
                u = (current_price - trader.entry_price) / trader.entry_price
                if trader.position == -1:
                    u = -u
                equity = trader.balance * (1 + u)
            else:
                equity = trader.balance
            
            trader.equity_history.append({
                'timestamp': datetime.now(),
                'equity': equity,
                'price': current_price
            })
            
            # Status update every 10 checks
            if check_count % 10 == 0:
                status = trader.get_status(current_price)
                print(status)
                send_telegram(status)
                
                # Plot and send equity
                path = trader.plot_equity()
                if path:
                    send_telegram("📈 Equity Curve:", path)
                
                # Save trades
                if trader.trades:
                    pd.DataFrame(trader.trades).to_csv('/data/workspace/notebooks/realtime_trades.csv', index=False)
            
            print(f"📊 Balance: ${trader.balance:.2f} | Position: {'OPEN' if trader.position != 0 else 'NONE'}")
            
            # Wait for next check
            print(f"⏱️ Next check in {CHECK_INTERVAL//60} minutes...")
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping bot...")
        send_telegram("🛑 *Trading Bot Stopped!*\n\n"
                      f"Final Balance: ${trader.balance:.2f}\n"
                      f"Total Trades: {len(trader.trades)}")
        
        # Save final results
        if trader.trades:
            pd.DataFrame(trader.trades).to_csv('/data/workspace/notebooks/realtime_trades.csv', index=False)
            print("✅ Saved: realtime_trades.csv")
        
        path = trader.plot_equity()
        if path:
            send_telegram("📈 Final Equity:", path)
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS:")
        print(f"Balance: ${trader.balance:.2f}")
        print(f"Trades: {len(trader.trades)}")
        print(f"Wins: {trader.wins} | Losses: {trader.losses}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()