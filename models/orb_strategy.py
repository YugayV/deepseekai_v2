#!/usr/bin/env python3
"""
ORB (Opening Range Breakout) Strategy - Complete Analysis
=========================================================
Pair: EUR/USD
Timeframe: H1 (Hourly)
Period: ~2 Years

Based on 24-hour Rolling Range Breakout
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import os
import pickle

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from xgboost import XGBClassifier

OUTPUT_DIR = '/data/workspace/notebooks'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('='*60)
print('📊 ORB (OPENING RANGE BREAKOUT) STRATEGY')
print('='*60)

# ============================================================
# 1. DATA
# ============================================================
print('\n📥 Downloading EUR/USD data...')

end_date = datetime.now()
start_date = end_date - timedelta(days=700)

data = yf.download('EURUSD=X', start=start_date, end=end_date, interval='1h', progress=False)
data.columns = [col[0] for col in data.columns]
data = data.dropna()

print(f'✅ Data: {len(data)} candles')
print(f'Range: {data.index[0]} to {data.index[-1]}')

# ============================================================
# 2. ORB (24-Hour Rolling Range)
# ============================================================
print('\n🔮 Calculating ORB...')

# Use 24-hour rolling high/low as the "opening range"
data['OR_High'] = data['High'].rolling(24).max()
data['OR_Low'] = data['Low'].rolling(24).min()
data['OR_Range'] = data['OR_High'] - data['OR_Low']

# ATR
hl = data['High'] - data['Low']
hc = np.abs(data['High'] - data['Close'].shift())
lc = np.abs(data['Low'] - data['Close'].shift())
tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
data['ATR'] = tr.rolling(14).mean()

# Breakouts: Close above/below previous 24h range
data['Breakout_Up'] = (data['Close'] > data['OR_High'].shift(1)).astype(int)
data['Breakout_Down'] = (data['Close'] < data['OR_Low'].shift(1)).astype(int)

bu = data['Breakout_Up'].sum()
bd = data['Breakout_Down'].sum()
print(f'   Breakout Up: {bu}')
print(f'   Breakout Down: {bd}')

# ============================================================
# 3. INDICATORS
# ============================================================
print('\n📊 Calculating indicators...')

# Hour
data['Hour'] = data.index.hour

# Trend (MA crossover)
ma20 = data['Close'].rolling(20).mean()
ma50 = data['Close'].rolling(50).mean()
data['Trend'] = np.where(data['Close'] > ma20, 1, -1).astype(int)
data['Trend_Strength'] = (data['Close'] - ma20) / ma20 * 100

# Volatility
data['Volatility'] = data['ATR'] / data['Close'] * 100

# RSI
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# MACD
ema12 = data['Close'].ewm(span=12).mean()
ema26 = data['Close'].ewm(span=26).mean()
macd = ema12 - ema26
macd_sig = macd.ewm(span=9).mean()
data['MACD'] = macd
data['MACD_Signal'] = macd_sig
data['MACD_Hist'] = macd - macd_sig

# Sessions
data['Is_EU_Session'] = ((data['Hour'] >= 7) & (data['Hour'] <= 15)).astype(int)
data['Is_US_Session'] = ((data['Hour'] >= 13) & (data['Hour'] <= 21)).astype(int)

# ============================================================
# 4. LABELS
# ============================================================
print('\n🏷️ Generating labels...')

labels = []
lookforward = 4

for i in range(len(data)):
    if data['Breakout_Up'].iloc[i] == 1:
        entry = data['Close'].iloc[i]
        if i + lookforward < len(data):
            future_close = data['Close'].iloc[i + lookforward]
            if future_close > entry * 1.001:
                labels.append(1)
            elif future_close < entry * 0.999:
                labels.append(-1)
            else:
                labels.append(0)
        else:
            labels.append(0)
    elif data['Breakout_Down'].iloc[i] == 1:
        entry = data['Close'].iloc[i]
        if i + lookforward < len(data):
            future_close = data['Close'].iloc[i + lookforward]
            if future_close < entry * 0.999:
                labels.append(1)
            elif future_close > entry * 1.001:
                labels.append(-1)
            else:
                labels.append(0)
        else:
            labels.append(0)
    else:
        labels.append(0)

data['Label'] = labels

counts = pd.Series(labels).value_counts()
print(f'   Win(1): {counts.get(1, 0)}')
print(f'   Neutral(0): {counts.get(0, 0)}')
print(f'   Loss(-1): {counts.get(-1, 0)}')

# ============================================================
# 5. FEATURES
# ============================================================
print('\n⚙️ Creating features...')

features = pd.DataFrame(index=data.index)
features['OR_Range_Pct'] = data['OR_Range'] / data['Close'] * 100
features['Breakout_Up'] = data['Breakout_Up']
features['Breakout_Down'] = data['Breakout_Down']
features['Trend'] = data['Trend']
features['Trend_Strength'] = data['Trend_Strength']
features['Volatility'] = data['Volatility']
features['RSI'] = data['RSI']
features['MACD'] = data['MACD']
features['MACD_Hist'] = data['MACD_Hist']
features['Hour'] = data['Hour']
features['Is_EU_Session'] = data['Is_EU_Session']
features['Is_US_Session'] = data['Is_US_Session']
features['Momentum_4'] = data['Close'].pct_change(4)
features['Momentum_12'] = data['Close'].pct_change(12)
features['Return_1H'] = data['Close'].pct_change(1)
features['ATR_Pct'] = data['ATR'] / data['Close'] * 100

features = features.replace([np.inf, -np.inf], np.nan).dropna()
data_aligned = data.loc[features.index]

print(f'   Features: {features.shape}')

# ============================================================
# 6. TRAIN/TEST SPLIT
# ============================================================
print('\n📅 Splitting data...')

split_idx = int(len(features) * 0.6)
train_idx = features.index[:split_idx]
test_idx = features.index[split_idx:]

X_train = features.loc[train_idx]
X_test = features.loc[test_idx]
y_train = data_aligned['Label'].loc[train_idx]
y_test = data_aligned['Label'].loc[test_idx]

# Binary: breakout signals only
train_mask = y_train != 0
test_mask = y_test != 0

X_train_bin = X_train[train_mask]
y_train_bin = (y_train[train_mask] == 1).astype(int)
X_test_bin = X_test[test_mask]
y_test_bin = (y_test[test_mask] == 1).astype(int)

print(f'   Train: {len(X_train_bin)}, Test: {len(X_test_bin)}')

if len(X_train_bin) < 50:
    print('   ⚠️ Few samples, using all data...')
    X_train_bin = X_train
    X_test_bin = X_test
    y_train_bin = (y_train == 1).astype(int)
    y_test_bin = (y_test == 1).astype(int)

# ============================================================
# 7. TRAINING
# ============================================================
print('\n🚀 Training models...')

xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
xgb.fit(X_train_bin, y_train_bin)
pred_xgb = xgb.predict(X_test_bin)
proba_xgb = xgb.predict_proba(X_test_bin)[:, 1]
try:
    auc_xgb = roc_auc_score(y_test_bin, proba_xgb)
except:
    auc_xgb = 0.5
print(f'   XGBoost ROC-AUC: {auc_xgb:.4f}')

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_bin, y_train_bin)
pred_rf = rf.predict(X_test_bin)
proba_rf = rf.predict_proba(X_test_bin)[:, 1]
try:
    auc_rf = roc_auc_score(y_test_bin, proba_rf)
except:
    auc_rf = 0.5
print(f'   Random Forest ROC-AUC: {auc_rf:.4f}')

gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)
gb.fit(X_train_bin, y_train_bin)
pred_gb = gb.predict(X_test_bin)
proba_gb = gb.predict_proba(X_test_bin)[:, 1]
try:
    auc_gb = roc_auc_score(y_test_bin, proba_gb)
except:
    auc_gb = 0.5
print(f'   Gradient Boosting ROC-AUC: {auc_gb:.4f}')

best_model = xgb
best_pred = pred_xgb
best_proba = proba_xgb
best_auc = auc_xgb

if auc_rf > best_auc:
    best_model = rf
    best_pred = pred_rf
    best_proba = proba_rf
    best_auc = auc_rf
if auc_gb > best_auc:
    best_model = gb
    best_pred = pred_gb
    best_proba = proba_gb
    best_auc = auc_gb

print(f'\n   🏆 Best: {type(best_model).__name__} with ROC-AUC {best_auc:.4f}')

# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================
print('\n📊 Feature Importance:')

feat_imp = pd.DataFrame({'feature': features.columns, 'importance': best_model.feature_importances_})
feat_imp = feat_imp.sort_values('importance', ascending=False)

for _, row in feat_imp.head(10).iterrows():
    print(f'   {row["feature"]:<20} {row["importance"]:.4f}')

fig, ax = plt.subplots(figsize=(10, 6))
top = feat_imp.head(12)
ax.barh(range(len(top)), top['importance'].values)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top['feature'].values)
ax.invert_yaxis()
ax.set_title('ORB Strategy - Feature Importance (XGBoost)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/orb_feature_importance.png', dpi=150)
plt.close()

# ============================================================
# 9. BACKTEST
# ============================================================
print('\n📈 Backtesting...')

trades = []
balance = 10000

for i in range(len(data_aligned)):
    if data_aligned['Breakout_Up'].iloc[i] == 1 or data_aligned['Breakout_Down'].iloc[i] == 1:
        entry = data_aligned['Close'].iloc[i]
        direction = 1 if data_aligned['Breakout_Up'].iloc[i] == 1 else -1
        
        if i + 4 < len(data_aligned):
            exit_price = data_aligned['Close'].iloc[i+4]
            if direction == 1:
                pnl = (exit_price / entry - 1) * balance
            else:
                pnl = (1 - exit_price / entry) * balance
            
            balance += pnl
            trades.append({
                'time': data_aligned.index[i],
                'direction': 'Long' if direction == 1 else 'Short',
                'entry': entry,
                'exit': exit_price,
                'pnl': pnl,
                'balance': balance
            })

trades_df = pd.DataFrame(trades)

if len(trades_df) > 0:
    wr = (trades_df['pnl'] > 0).mean() * 100
    print(f'   Trades: {len(trades_df)}')
    print(f'   Win Rate: {wr:.1f}%')
    print(f'   Final: ${balance:.2f}')
    print(f'   Return: {(balance/10000-1)*100:.1f}%')
    
    # Equity curve
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    recent = data_aligned.iloc[-500:]
    axes[0].plot(recent.index, recent['Close'], 'b-', label='EUR/USD')
    axes[0].plot(recent.index, recent['OR_High'], 'g--', alpha=0.7, label='24h High')
    axes[0].plot(recent.index, recent['OR_Low'], 'r--', alpha=0.7, label='24h Low')
    
    bu = recent[recent['Breakout_Up'] == 1]
    bd = recent[recent['Breakout_Down'] == 1]
    axes[0].scatter(bu.index, bu['Close'], color='green', s=30, marker='^', zorder=5, label='Buy')
    axes[0].scatter(bd.index, bd['Close'], color='red', s=30, marker='v', zorder=5, label='Sell')
    axes[0].set_title('EUR/USD H1 - ORB Strategy (Last 500 Hours)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(trades_df.index, trades_df['balance'], 'b-', linewidth=2)
    axes[1].axhline(10000, color='gray', linestyle='--', alpha=0.5)
    above = trades_df['balance'] >= 10000
    axes[1].fill_between(trades_df.index, 10000, trades_df['balance'], 
                         where=above, color='green', alpha=0.3)
    axes[1].fill_between(trades_df.index, 10000, trades_df['balance'], 
                         where=~above, color='red', alpha=0.3)
    axes[1].set_title('Equity Curve')
    axes[1].set_ylabel('Balance ($)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/orb_backtest.png', dpi=150)
    plt.close()
    
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test_bin, best_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/orb_confusion_matrix.png', dpi=150)
    plt.close()
    
    trades_df.to_csv(f'{OUTPUT_DIR}/orb_trades.csv', index=False)

# ============================================================
# 10. SAVE
# ============================================================
models = {'xgb': xgb, 'rf': rf, 'gb': gb}
with open(f'{OUTPUT_DIR}/orb_models.pkl', 'wb') as f:
    pickle.dump(models, f)

data.to_csv(f'{OUTPUT_DIR}/orb_data.csv')
features.to_csv(f'{OUTPUT_DIR}/orb_features.csv')

print(f'\n💾 Files saved:')
print(f'   - {OUTPUT_DIR}/orb_models.pkl')
print(f'   - {OUTPUT_DIR}/orb_data.csv')
print(f'   - {OUTPUT_DIR}/orb_features.csv')
print(f'   - {OUTPUT_DIR}/orb_trades.csv')
print(f'   - {OUTPUT_DIR}/orb_*.png')

print('\n' + '='*60)
print('✅ ORB STRATEGY COMPLETE!')
print('='*60)