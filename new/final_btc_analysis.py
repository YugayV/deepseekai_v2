#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

pd.set_option('display.width', 200)
plt.rcParams['font.family'] = 'DejaVu Sans'

print("=" * 80)
print("ФИНАЛЬНЫЙ АНАЛИЗ BITCOIN")
print("=" * 80)

# Загрузка правильных данных
df = pd.read_csv('/home/ubuntu/btc_daily_ohlcv_final.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"\n✓ Загружено {len(df)} дневных свечей")
print(f"  Период: {df['date'].min().strftime('%Y-%m-%d')} - {df['date'].max().strftime('%Y-%m-%d')}")

# Технические индикаторы
def calc_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['MA20'] = df['close'].rolling(20).mean()
df['MA50'] = df['close'].rolling(50).mean()
df['MA200'] = df['close'].rolling(200).mean()
df['RSI'] = calc_rsi(df['close'])
df['EMA12'] = df['close'].ewm(span=12).mean()
df['EMA26'] = df['close'].ewm(span=26).mean()
df['MACD'] = df['EMA12'] - df['EMA26']
df['Signal'] = df['MACD'].ewm(span=9).mean()

current = df.iloc[-1]
print(f"\n📊 ТЕКУЩИЕ ИНДИКАТОРЫ (на {current['date'].strftime('%Y-%m-%d')}):")
print(f"  Цена: ${current['close']:,.2f}")
print(f"  MA20: ${current['MA20']:,.2f}, MA50: ${current['MA50']:,.2f}, MA200: ${current['MA200']:,.2f}")
print(f"  RSI(14): {current['RSI']:.1f}")
print(f"  MACD: {current['MACD']:.2f}")

# Уровни поддержки/сопротивления (из локальных экстремумов)
from scipy.signal import argrelextrema
highs_idx = argrelextrema(df['high'].values, np.greater, order=10)[0]
lows_idx = argrelextrema(df['low'].values, np.less, order=10)[0]

recent_highs = df.iloc[highs_idx[highs_idx >= len(df)-90]]['high'].values if len(highs_idx) > 0 else []
recent_lows = df.iloc[lows_idx[lows_idx >= len(df)-90]]['low'].values if len(lows_idx) > 0 else []

resistances = sorted([x for x in recent_highs if x > current['close']])[:3] if len(recent_highs) > 0 else [80000, 85000, 90000]
supports = sorted([x for x in recent_lows if x < current['close']], reverse=True)[:3] if len(recent_lows) > 0 else [75000, 70000, 65000]

print(f"\n🎯 КЛЮЧЕВЫЕ УРОВНИ:")
print(f"  Сопротивление: {', '.join([f'${x:,.0f}' for x in resistances])}")
print(f"  Поддержка: {', '.join([f'${x:,.0f}' for x in supports])}")

# Сохранение результатов
results = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
    'current_price': float(current['close']),
    'indicators': {
        'MA20': float(current['MA20']),
        'MA50': float(current['MA50']),
        'MA200': float(current['MA200']),
        'RSI': float(current['RSI']),
        'MACD': float(current['MACD'])
    },
    'levels': {
        'resistance': [float(x) for x in resistances],
        'support': [float(x) for x in supports]
    }
}

with open('/home/ubuntu/final_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

df.to_csv('/home/ubuntu/btc_with_indicators.csv', index=False)
print("\n✓ Результаты сохранены")
print("=" * 80)
