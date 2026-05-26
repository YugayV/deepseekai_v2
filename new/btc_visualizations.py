#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация анализа Bitcoin
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Настройка русского шрифта
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

print("=" * 80)
print("СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ ДЛЯ ОТЧЕТА")
print("=" * 80)

# Загрузка данных
df_btc = pd.read_csv('/home/ubuntu/btc_ohlc_indicators.csv')
df_btc['date'] = pd.to_datetime(df_btc['date'])

with open('/home/ubuntu/btc_analysis_results.json', 'r') as f:
    results = json.load(f)

# Функция для конвертации графика в base64
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

# ============================================================================
# ГРАФИК 1: ЦЕНА И СКОЛЬЗЯЩИЕ СРЕДНИЕ
# ============================================================================

print("\n[1/5] Создание графика: Цена и скользящие средние...")

fig1 = plt.figure(figsize=(14, 8))
gs = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)

# Основной график цены
ax1 = fig1.add_subplot(gs[0])
ax1.plot(df_btc['date'], df_btc['close'], label='BTC/USD', linewidth=2, color='#F7931A')
ax1.plot(df_btc['date'], df_btc['MA_20'], label='MA20', linewidth=1.5, linestyle='--', alpha=0.8)
ax1.plot(df_btc['date'], df_btc['MA_50'], label='MA50', linewidth=1.5, linestyle='--', alpha=0.8)
ax1.plot(df_btc['date'], df_btc['MA_200'], label='MA200', linewidth=1.5, linestyle='--', alpha=0.8)

# Уровни поддержки и сопротивления
for level in results['key_levels']['resistance'][:3]:
    ax1.axhline(y=level, color='r', linestyle=':', alpha=0.5, linewidth=1)
    ax1.text(df_btc['date'].iloc[-1], level, f'  ${level:,.0f}', fontsize=8, va='center', color='red')

for level in results['key_levels']['support'][:3]:
    ax1.axhline(y=level, color='g', linestyle=':', alpha=0.5, linewidth=1)
    ax1.text(df_btc['date'].iloc[-1], level, f'  ${level:,.0f}', fontsize=8, va='center', color='green')

ax1.set_title('Bitcoin (BTC/USD): Цена и скользящие средние', fontsize=14, fontweight='bold')
ax1.set_ylabel('Цена (USD)', fontsize=11)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# RSI
ax2 = fig1.add_subplot(gs[1], sharex=ax1)
ax2.plot(df_btc['date'], df_btc['RSI_14'], linewidth=1.5, color='purple')
ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, linewidth=1)
ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, linewidth=1)
ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.3, linewidth=1)
ax2.fill_between(df_btc['date'], 70, 100, alpha=0.1, color='red')
ax2.fill_between(df_btc['date'], 0, 30, alpha=0.1, color='green')
ax2.set_ylabel('RSI(14)', fontsize=10)
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3)

# MACD
ax3 = fig1.add_subplot(gs[2], sharex=ax1)
ax3.plot(df_btc['date'], df_btc['MACD'], label='MACD', linewidth=1.5)
ax3.plot(df_btc['date'], df_btc['MACD_Signal'], label='Signal', linewidth=1.5)
ax3.bar(df_btc['date'], df_btc['MACD_Histogram'], label='Histogram', alpha=0.3, color='gray')
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
ax3.set_ylabel('MACD', fontsize=10)
ax3.set_xlabel('Дата', fontsize=10)
ax3.legend(loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.tight_layout()
fig1.savefig('/home/ubuntu/chart1_price_indicators.png', bbox_inches='tight', dpi=150)
chart1_base64 = fig_to_base64(fig1)
plt.close(fig1)

print("✓ График 1 создан: chart1_price_indicators.png")

# ============================================================================
# ГРАФИК 2: ВОЛАТИЛЬНОСТЬ И ПРОИЗВОДИТЕЛЬНОСТЬ
# ============================================================================

print("[2/5] Создание графика: Волатильность и производительность...")

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

# Волатильность
ax1.plot(df_btc['date'], df_btc['Volatility_30d'], linewidth=2, color='orange')
ax1.set_title('Bitcoin: 30-дневная реализованная волатильность', fontsize=14, fontweight='bold')
ax1.set_ylabel('Волатильность (%)', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Кумулятивная доходность
df_btc['cumulative_return'] = (1 + df_btc['daily_return']/100).cumprod() - 1
ax2.plot(df_btc['date'], df_btc['cumulative_return'] * 100, linewidth=2, color='#F7931A')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
ax2.fill_between(df_btc['date'], 0, df_btc['cumulative_return'] * 100, 
                 where=(df_btc['cumulative_return'] * 100 >= 0), alpha=0.3, color='green')
ax2.fill_between(df_btc['date'], 0, df_btc['cumulative_return'] * 100, 
                 where=(df_btc['cumulative_return'] * 100 < 0), alpha=0.3, color='red')
ax2.set_title('Bitcoin: Кумулятивная доходность', fontsize=14, fontweight='bold')
ax2.set_ylabel('Доходность (%)', fontsize=11)
ax2.set_xlabel('Дата', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.tight_layout()
fig2.savefig('/home/ubuntu/chart2_volatility_returns.png', bbox_inches='tight', dpi=150)
chart2_base64 = fig_to_base64(fig2)
plt.close(fig2)

print("✓ График 2 создан: chart2_volatility_returns.png")

# ============================================================================
# ГРАФИК 3: КОРРЕЛЯЦИИ С ТРАДИЦИОННЫМИ АКТИВАМИ
# ============================================================================

print("[3/5] Создание графика: Корреляции с традиционными активами...")

df_macro = pd.read_csv('/home/ubuntu/btc_macro_correlations.csv')
df_macro['date'] = pd.to_datetime(df_macro['date'])

# Нормализация цен (100 = начальное значение)
for col in ['btc_close', 'spx_close', 'qqq_close', 'gld_close']:
    df_macro[f'{col}_norm'] = (df_macro[col] / df_macro[col].iloc[0]) * 100

fig3 = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)

# Нормализованные цены
ax1 = fig3.add_subplot(gs[0, :])
ax1.plot(df_macro['date'], df_macro['btc_close_norm'], label='BTC', linewidth=2, color='#F7931A')
ax1.plot(df_macro['date'], df_macro['spx_close_norm'], label='S&P 500', linewidth=2, alpha=0.7)
ax1.plot(df_macro['date'], df_macro['qqq_close_norm'], label='QQQ (Nasdaq)', linewidth=2, alpha=0.7)
ax1.plot(df_macro['date'], df_macro['gld_close_norm'], label='Gold', linewidth=2, alpha=0.7)
ax1.axhline(y=100, color='black', linestyle='--', alpha=0.3)
ax1.set_title('Сравнение производительности: BTC vs традиционные активы (нормализовано)', 
              fontsize=14, fontweight='bold')
ax1.set_ylabel('Индекс (100 = начало)', fontsize=11)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Корреляционная матрица
corr_data = {
    'Актив': ['S&P 500', 'QQQ', 'Gold'],
    '30 дней': [
        results['correlations_30d']['spx'],
        results['correlations_30d']['qqq'],
        results['correlations_30d']['gold']
    ]
}
corr_df = pd.DataFrame(corr_data)

ax2 = fig3.add_subplot(gs[1, 0])
colors = ['green' if x > 0.3 else 'orange' if x > 0 else 'red' for x in corr_df['30 дней']]
bars = ax2.barh(corr_df['Актив'], corr_df['30 дней'], color=colors, alpha=0.7)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.axvline(x=0.3, color='green', linestyle='--', alpha=0.3)
ax2.axvline(x=-0.3, color='red', linestyle='--', alpha=0.3)
ax2.set_xlabel('Корреляция', fontsize=10)
ax2.set_title('Корреляция BTC (30 дней)', fontsize=12, fontweight='bold')
ax2.set_xlim(-1, 1)
ax2.grid(True, alpha=0.3, axis='x')

# Добавление значений на графике
for i, (bar, val) in enumerate(zip(bars, corr_df['30 дней'])):
    ax2.text(val + 0.05 if val >= 0 else val - 0.05, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)

# Производительность сравнение
perf_data = {
    'Актив': ['BTC'],
    '30d': [results['performance'].get('30d', 0)],
    '90d': [results['performance'].get('90d', 0)]
}
perf_df = pd.DataFrame(perf_data)

ax3 = fig3.add_subplot(gs[1, 1])
x = np.arange(len(perf_df))
width = 0.35
bars1 = ax3.bar(x - width/2, perf_df['30d'], width, label='30 дней', alpha=0.8)
bars2 = ax3.bar(x + width/2, perf_df['90d'], width, label='90 дней', alpha=0.8)

# Раскраска баров
for bars in [bars1, bars2]:
    for bar in bars:
        if bar.get_height() < 0:
            bar.set_color('red')
        else:
            bar.set_color('green')

ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.set_ylabel('Доходность (%)', fontsize=10)
ax3.set_title('Производительность BTC', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(perf_df['Актив'])
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# Добавление значений на графике
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -1),
                f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

plt.tight_layout()
fig3.savefig('/home/ubuntu/chart3_correlations.png', bbox_inches='tight', dpi=150)
chart3_base64 = fig_to_base64(fig3)
plt.close(fig3)

print("✓ График 3 создан: chart3_correlations.png")

# ============================================================================
# ГРАФИК 4: BOLLINGER BANDS
# ============================================================================

print("[4/5] Создание графика: Bollinger Bands...")

fig4, ax = plt.subplots(figsize=(14, 8))

# Цена и Bollinger Bands
ax.plot(df_btc['date'], df_btc['close'], label='BTC/USD', linewidth=2, color='#F7931A')
ax.plot(df_btc['date'], df_btc['BB_Upper'], label='BB Upper', linewidth=1, linestyle='--', color='red', alpha=0.7)
ax.plot(df_btc['date'], df_btc['BB_Middle'], label='BB Middle (MA20)', linewidth=1, linestyle='--', color='gray', alpha=0.7)
ax.plot(df_btc['date'], df_btc['BB_Lower'], label='BB Lower', linewidth=1, linestyle='--', color='green', alpha=0.7)
ax.fill_between(df_btc['date'], df_btc['BB_Lower'], df_btc['BB_Upper'], alpha=0.1, color='gray')

ax.set_title('Bitcoin: Bollinger Bands (20, 2)', fontsize=14, fontweight='bold')
ax.set_ylabel('Цена (USD)', fontsize=11)
ax.set_xlabel('Дата', fontsize=11)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.tight_layout()
fig4.savefig('/home/ubuntu/chart4_bollinger.png', bbox_inches='tight', dpi=150)
chart4_base64 = fig_to_base64(fig4)
plt.close(fig4)

print("✓ График 4 создан: chart4_bollinger.png")

# ============================================================================
# ГРАФИК 5: РАСПРЕДЕЛЕНИЕ ДНЕВНЫХ ДОХОДНОСТЕЙ
# ============================================================================

print("[5/5] Создание графика: Распределение доходностей...")

fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Гистограмма дневных доходностей
ax1.hist(df_btc['daily_return'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='#F7931A')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax1.axvline(x=df_btc['daily_return'].mean(), color='blue', linestyle='--', linewidth=2, label=f"Среднее: {df_btc['daily_return'].mean():.2f}%")
ax1.set_title('Распределение дневных доходностей BTC', fontsize=14, fontweight='bold')
ax1.set_xlabel('Дневная доходность (%)', fontsize=11)
ax1.set_ylabel('Частота', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Drawdown
running_max = df_btc['close'].expanding().max()
drawdown = (df_btc['close'] - running_max) / running_max * 100

ax2.fill_between(df_btc['date'], 0, drawdown, alpha=0.5, color='red')
ax2.plot(df_btc['date'], drawdown, linewidth=1.5, color='darkred')
ax2.set_title('Bitcoin: Просадка от ATH', fontsize=14, fontweight='bold')
ax2.set_ylabel('Просадка (%)', fontsize=11)
ax2.set_xlabel('Дата', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.tight_layout()
fig5.savefig('/home/ubuntu/chart5_distribution.png', bbox_inches='tight', dpi=150)
chart5_base64 = fig_to_base64(fig5)
plt.close(fig5)

print("✓ График 5 создан: chart5_distribution.png")

# Сохранение base64 строк для встраивания в отчет
charts_base64 = {
    'chart1': chart1_base64,
    'chart2': chart2_base64,
    'chart3': chart3_base64,
    'chart4': chart4_base64,
    'chart5': chart5_base64
}

with open('/home/ubuntu/charts_base64.json', 'w') as f:
    json.dump(charts_base64, f)

print("\n" + "=" * 80)
print("ВСЕ ВИЗУАЛИЗАЦИИ СОЗДАНЫ УСПЕШНО")
print("=" * 80)
