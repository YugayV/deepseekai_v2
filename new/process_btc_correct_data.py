#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("ОБРАБОТКА ПРАВИЛЬНЫХ BTC ДАННЫХ")
print("=" * 80)

# Загрузка 90-дневных данных с почасовым разрешением
with open('/home/ubuntu/.external_service_outputs/fetch_crypto_data_output_1779316873.json', 'r') as f:
    data_90d = json.load(f)

# Загрузка 540-дневных данных (дневное разрешение)
with open('/home/ubuntu/.external_service_outputs/fetch_crypto_data_output_1779316842.json', 'r') as f:
    data_540d = json.load(f)

# ============================================================================
# 1. Обработка 90-дневных данных (для точных OHLCV)
# ============================================================================

print("\n[1/3] Обработка 90-дневных данных (почасовое разрешение)...")

df_hourly = pd.DataFrame(data_90d['prices'], columns=['timestamp', 'price'])
df_hourly['datetime'] = pd.to_datetime(df_hourly['timestamp'], unit='ms')
df_hourly['date'] = df_hourly['datetime'].dt.date

# Volume data
df_volume = pd.DataFrame(data_90d['total_volumes'], columns=['timestamp', 'volume'])
df_volume['datetime'] = pd.to_datetime(df_volume['timestamp'], unit='ms')
df_volume['date'] = df_volume['datetime'].dt.date

print(f"Всего почасовых точек: {len(df_hourly)}")
print(f"Период: {df_hourly['datetime'].min()} - {df_hourly['datetime'].max()}")

# Интервал между точками
intervals = df_hourly['timestamp'].diff() / 1000 / 3600
print(f"Средний интервал: {intervals.mean():.2f} часов")

# Агрегация в дневные OHLCV
daily_90d = df_hourly.groupby('date').agg({
    'price': ['first', 'max', 'min', 'last', 'count']
}).reset_index()
daily_90d.columns = ['date', 'open', 'high', 'low', 'close', 'ticks']

# Добавление объемов
volume_daily = df_volume.groupby('date')['volume'].mean().reset_index()
daily_90d = daily_90d.merge(volume_daily, on='date', how='left')

daily_90d = daily_90d.sort_values('date')

print(f"\nДневных свечей (90д): {len(daily_90d)}")
print(f"Средне точек на свечу: {daily_90d['ticks'].mean():.1f}")
print("\nПервые 3 дня:")
print(daily_90d.head(3))
print("\nПоследние 3 дня:")
print(daily_90d.tail(3))

# ============================================================================
# 2. Объединение с 540-дневными данными
# ============================================================================

print("\n[2/3] Объединение с 540-дневными данными...")

df_daily_540 = pd.DataFrame(data_540d['prices'], columns=['timestamp', 'close'])
df_daily_540['datetime'] = pd.to_datetime(df_daily_540['timestamp'], unit='ms')
df_daily_540['date'] = df_daily_540['datetime'].dt.date

df_volume_540 = pd.DataFrame(data_540d['total_volumes'], columns=['timestamp', 'volume'])
df_volume_540['datetime'] = pd.to_datetime(df_volume_540['timestamp'], unit='ms')
df_volume_540['date'] = df_volume_540['datetime'].dt.date
df_volume_540 = df_volume_540[['date', 'volume']]

# Для 540-дневных данных O=H=L=C (недостаточно внутри-дневных точек)
df_daily_540['open'] = df_daily_540['close']
df_daily_540['high'] = df_daily_540['close']
df_daily_540['low'] = df_daily_540['close']

df_daily_540 = df_daily_540.merge(df_volume_540, on='date', how='left')
df_daily_540 = df_daily_540[['date', 'open', 'high', 'low', 'close', 'volume']]

# Объединение: приоритет 90-дневным данным (точнее), затем 540-дневные
daily_90d['date'] = pd.to_datetime(daily_90d['date'])
df_daily_540['date'] = pd.to_datetime(df_daily_540['date'])

# Используем 90-дневные данные где они есть
df_final = df_daily_540.copy()
dates_90d = set(daily_90d['date'])

for idx, row in df_final.iterrows():
    if row['date'] in dates_90d:
        match = daily_90d[daily_90d['date'] == row['date']].iloc[0]
        df_final.at[idx, 'open'] = match['open']
        df_final.at[idx, 'high'] = match['high']
        df_final.at[idx, 'low'] = match['low']
        df_final.at[idx, 'close'] = match['close']
        if pd.notna(match['volume']):
            df_final.at[idx, 'volume'] = match['volume']

print(f"Финальный датасет: {len(df_final)} дневных свечей")
print(f"Период: {df_final['date'].min()} - {df_final['date'].max()}")

# ============================================================================
# 3. Расчет базовых метрик
# ============================================================================

print("\n[3/3] Расчет базовых метрик...")

df_final['daily_return'] = df_final['close'].pct_change() * 100
df_final['log_return'] = np.log(df_final['close'] / df_final['close'].shift(1))

# Текущие метрики
current_price = df_final['close'].iloc[-1]
price_30d_ago = df_final['close'].iloc[-30] if len(df_final) >= 30 else df_final['close'].iloc[0]
price_90d_ago = df_final['close'].iloc[-90] if len(df_final) >= 90 else df_final['close'].iloc[0]
price_180d_ago = df_final['close'].iloc[-180] if len(df_final) >= 180 else df_final['close'].iloc[0]
price_365d_ago = df_final['close'].iloc[-365] if len(df_final) >= 365 else df_final['close'].iloc[0]

perf_30d = ((current_price / price_30d_ago) - 1) * 100
perf_90d = ((current_price / price_90d_ago) - 1) * 100
perf_180d = ((current_price / price_180d_ago) - 1) * 100
perf_365d = ((current_price / price_365d_ago) - 1) * 100

# Волатильность
vol_30d = df_final['daily_return'].tail(30).std() * np.sqrt(365)
vol_90d = df_final['daily_return'].tail(90).std() * np.sqrt(365)

# Max drawdown
running_max = df_final['close'].expanding().max()
drawdown = (df_final['close'] - running_max) / running_max
max_dd = drawdown.min() * 100

print("\n📊 КЛЮЧЕВЫЕ МЕТРИКИ:")
print(f"Текущая цена: ${current_price:,.2f}")
print(f"ATH: ${df_final['high'].max():,.2f}")
print(f"ATL (за период): ${df_final['low'].min():,.2f}")
print(f"\nПроизводительность:")
print(f"  30 дней: {perf_30d:+.2f}%")
print(f"  90 дней: {perf_90d:+.2f}%")
print(f"  180 дней: {perf_180d:+.2f}%")
print(f"  365 дней: {perf_365d:+.2f}%")
print(f"\nВолатильность (годовая):")
print(f"  30 дней: {vol_30d:.2f}%")
print(f"  90 дней: {vol_90d:.2f}%")
print(f"\nМаксимальная просадка: {max_dd:.2f}%")

# Сохранение
df_final.to_csv('/home/ubuntu/btc_daily_ohlcv_final.csv', index=False)
print("\n✓ Финальные данные сохранены в btc_daily_ohlcv_final.csv")

# Сохранение метрик
metrics = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
    'data_points': len(df_final),
    'period_start': df_final['date'].min().strftime('%Y-%m-%d'),
    'period_end': df_final['date'].max().strftime('%Y-%m-%d'),
    'current_price': float(current_price),
    'ath': float(df_final['high'].max()),
    'atl': float(df_final['low'].min()),
    'performance': {
        '30d': float(perf_30d),
        '90d': float(perf_90d),
        '180d': float(perf_180d),
        '365d': float(perf_365d)
    },
    'volatility': {
        '30d': float(vol_30d),
        '90d': float(vol_90d)
    },
    'max_drawdown': float(max_dd)
}

with open('/home/ubuntu/btc_metrics_correct.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("✓ Метрики сохранены в btc_metrics_correct.json")

print("\n" + "=" * 80)
print("ОБРАБОТКА ЗАВЕРШЕНА")
print("=" * 80)

