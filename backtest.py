import pandas as pd
import numpy as np
import yfinance as yf
import talib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("BACKTEST WITH TIME SERIES CROSS-VALIDATION")
print("=" * 60)

# Load data
ticker = yf.Ticker("EURUSD=X")
df = ticker.history(period="5y", interval="1d")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.columns = ['open', 'high', 'low', 'close', 'volume']

# Calculate indicators (same as in training)
df['returns'] = df['close'].pct_change() * 100
df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
df['volatility'] = df['returns'].rolling(20).std()
df['ema_8'] = talib.EMA(df['close'], timeperiod=8)
df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
df['rsi'] = talib.RSI(df['close'], timeperiod=14)
df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])

# Alligator
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

# Fractals
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
df['bullish_fractal_alligator'] = ((df['fractal_bullish'] == 1) & ((df['alligator_bullish'] == 1) | (df['alligator_asleep'] == 1))).astype(int)
df['bearish_fractal_alligator'] = ((df['fractal_bearish'] == 1) & ((df['alligator_bearish'] == 1) | (df['alligator_asleep'] == 1))).astype(int)
df['strong_bullish'] = ((df['fractal_bullish'] == 1) & (df['alligator_bullish'] == 1) & (df['alligator_expanding'] == 1)).astype(int)
df['strong_bearish'] = ((df['fractal_bearish'] == 1) & (df['alligator_bearish'] == 1) & (df['alligator_expanding'] == 1)).astype(int)

# Lags
for lag in [1, 2, 3, 5]:
    df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

# Target
lookahead = 5
df['future_return'] = df['close'].pct_change(lookahead).shift(-lookahead) * 100
df['target'] = np.select([df['future_return'] < -0.5, df['future_return'].abs() <= 0.5, df['future_return'] > 0.5], [0, 1, 2], 1)

feature_cols = ['returns', 'log_returns', 'volatility', 'ema_8', 'ema_21', 'ema_50',
                'macd', 'macd_signal', 'macd_hist', 'rsi', 'atr', 'bb_upper', 'bb_lower',
                'jaw', 'teeth', 'lips', 'alligator_asleep', 'alligator_bullish', 'alligator_bearish',
                'alligator_expanding', 'fractal_bullish', 'fractal_bearish',
                'bullish_fractal_alligator', 'bearish_fractal_alligator', 'strong_bullish', 'strong_bearish']
for lag in [1, 2, 3, 5]:
    feature_cols.append(f'returns_lag_{lag}')

df_clean = df[feature_cols + ['target']].dropna()
X = df_clean[feature_cols]
y = df_clean['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = balanced_accuracy_score(y_train, model.predict(X_train))
    test_acc = balanced_accuracy_score(y_test, model.predict(X_test))
    scores.append(test_acc)
    print(f"Fold {fold}: Train={train_acc:.4f}, Test={test_acc:.4f}, Diff={train_acc - test_acc:.4f}")

print(f"\n✅ Average Balanced Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
print(f"Overfitting: {np.mean(train_acc - test_acc for _ in range(5)):.4f}")