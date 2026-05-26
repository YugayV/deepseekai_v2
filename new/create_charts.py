import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import base64
from io import BytesIO
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Создание визуализаций...")

# Загрузка данных
with open('/home/ubuntu/analysis_data.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['df']
current_data = data['current_data']
technical_data = data['technical_data']
analysis_summary = data['analysis_summary']
returns_data = data['returns_data']

# Функция для сохранения графика как base64
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64

# Список для хранения base64 графиков
charts_base64 = {}

# 1. График сравнения цен (нормализованный)
print("1. График сравнения цен...")
fig, ax = plt.subplots(figsize=(14, 7))

for symbol in current_data.keys():
    symbol_df = df[df['symbol'] == symbol].sort_values('ts_event')
    if len(symbol_df) > 0:
        # Нормализуем к 100 от первой цены
        prices = symbol_df['close'].values
        normalized = (prices / prices[0]) * 100
        ax.plot(symbol_df['ts_event'], normalized, label=symbol, linewidth=2)

ax.set_title('Сравнение динамики цен акций (нормализовано к 100)', fontsize=16, fontweight='bold')
ax.set_xlabel('Дата', fontsize=12)
ax.set_ylabel('Относительная цена (база = 100)', fontsize=12)
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Сохранение
plt.savefig('/home/ubuntu/chart_price_comparison.png', dpi=150, bbox_inches='tight')
charts_base64['price_comparison'] = fig_to_base64(fig)
plt.close()

# 2. График доходности за разные периоды
print("2. График доходности...")
fig, ax = plt.subplots(figsize=(12, 6))

symbols = list(current_data.keys())
periods = ['1M %', '3M %', '6M %', '12M %']
x = np.arange(len(symbols))
width = 0.2

returns_matrix = []
for period in periods:
    period_key = period.replace(' %', '')
    returns_matrix.append([returns_data[s][period_key] for s in symbols])

for i, period in enumerate(periods):
    ax.bar(x + i * width, returns_matrix[i], width, label=period)

ax.set_title('Доходность акций за разные периоды', fontsize=16, fontweight='bold')
ax.set_xlabel('Акция', fontsize=12)
ax.set_ylabel('Доходность (%)', fontsize=12)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(symbols)
ax.legend(loc='best', fontsize=10)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

plt.savefig('/home/ubuntu/chart_returns.png', dpi=150, bbox_inches='tight')
charts_base64['returns'] = fig_to_base64(fig)
plt.close()

# 3. RSI и технические индикаторы
print("3. График RSI...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, symbol in enumerate(current_data.keys()):
    df_tech = technical_data[symbol].tail(126)  # Последние 6 месяцев
    
    axes[idx].plot(df_tech['ts_event'], df_tech['RSI'], color='blue', linewidth=1.5)
    axes[idx].axhline(y=70, color='red', linestyle='--', label='Перекупленность (70)')
    axes[idx].axhline(y=30, color='green', linestyle='--', label='Перепроданность (30)')
    axes[idx].set_title(f'{symbol} - RSI', fontsize=12, fontweight='bold')
    axes[idx].set_ylim(0, 100)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend(loc='best', fontsize=8)
    plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.suptitle('Индекс относительной силы (RSI) для всех акций', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()

plt.savefig('/home/ubuntu/chart_rsi.png', dpi=150, bbox_inches='tight')
charts_base64['rsi'] = fig_to_base64(fig)
plt.close()

# 4. Сравнение фундаментальных показателей
print("4. График фундаментальных показателей...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# P/E Ratio
symbols = list(current_data.keys())
pe_ratios = [current_data[s]['pe_ratio'] for s in symbols]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

ax1.barh(symbols, pe_ratios, color=colors)
ax1.set_title('P/E Ratio (Цена/Прибыль)', fontsize=14, fontweight='bold')
ax1.set_xlabel('P/E Ratio', fontsize=12)
ax1.grid(True, alpha=0.3, axis='x')

# Market Cap
market_caps = [current_data[s]['market_cap'] for s in symbols]
ax2.barh(symbols, market_caps, color=colors)
ax2.set_title('Рыночная капитализация', fontsize=14, fontweight='bold')
ax2.set_xlabel('Капитализация (трлн $)', fontsize=12)
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()

plt.savefig('/home/ubuntu/chart_fundamentals.png', dpi=150, bbox_inches='tight')
charts_base64['fundamentals'] = fig_to_base64(fig)
plt.close()

# 5. Детальный график цен с MA для каждой акции (2x3 сетка)
print("5. Детальные графики с MA...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, symbol in enumerate(current_data.keys()):
    df_tech = technical_data[symbol].tail(252)  # Последний год
    
    ax = axes[idx]
    ax.plot(df_tech['ts_event'], df_tech['close'], label='Цена', linewidth=2, color='black')
    ax.plot(df_tech['ts_event'], df_tech['MA50'], label='MA50', linewidth=1.5, color='blue', alpha=0.7)
    ax.plot(df_tech['ts_event'], df_tech['MA200'], label='MA200', linewidth=1.5, color='red', alpha=0.7)
    
    # Поддержка и сопротивление
    support = analysis_summary[symbol]['support']
    resistance = analysis_summary[symbol]['resistance']
    ax.axhline(y=support, color='green', linestyle=':', label=f'Поддержка (${support:.1f})', alpha=0.6)
    ax.axhline(y=resistance, color='red', linestyle=':', label=f'Сопротивление (${resistance:.1f})', alpha=0.6)
    
    ax.set_title(f'{symbol} - Цена и скользящие средние', fontsize=12, fontweight='bold')
    ax.set_ylabel('Цена ($)', fontsize=10)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

plt.suptitle('Технический анализ: Цена, скользящие средние и уровни поддержки/сопротивления', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()

plt.savefig('/home/ubuntu/chart_price_ma.png', dpi=150, bbox_inches='tight')
charts_base64['price_ma'] = fig_to_base64(fig)
plt.close()

# 6. Матрица потенциала роста и риска
print("6. Матрица риск/доходность...")
fig, ax = plt.subplots(figsize=(12, 8))

# Рассчитаем "потенциал" и "риск"
# Потенциал = средняя доходность + тренд
# Риск = волатильность + высокий P/E

potentials = []
risks = []
labels = []

for symbol in current_data.keys():
    # Потенциал: взвешенная доходность (больший вес на недавние периоды)
    ret = returns_data[symbol]
    potential = (ret['1M'] * 0.4 + ret['3M'] * 0.3 + ret['6M'] * 0.2 + ret['12M'] * 0.1)
    
    # Риск: волатильность + нормализованный P/E
    volatility = analysis_summary[symbol]['volatility']
    pe_normalized = current_data[symbol]['pe_ratio'] / 50  # Нормализация P/E
    risk = volatility + pe_normalized
    
    potentials.append(potential)
    risks.append(risk)
    labels.append(symbol)

# Scatter plot
scatter = ax.scatter(risks, potentials, s=500, alpha=0.6, c=range(len(symbols)), cmap='viridis')

# Добавляем метки
for i, label in enumerate(labels):
    ax.annotate(label, (risks[i], potentials[i]), fontsize=14, fontweight='bold', 
                ha='center', va='center')

ax.set_title('Матрица Риск-Потенциал роста', fontsize=16, fontweight='bold')
ax.set_xlabel('Риск (Волатильность + Оценка)', fontsize=12)
ax.set_ylabel('Потенциал роста (Взвешенная доходность)', fontsize=12)
ax.grid(True, alpha=0.3)

# Квадранты
ax.axhline(y=np.mean(potentials), color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=np.mean(risks), color='gray', linestyle='--', alpha=0.5)

# Аннотации квадрантов
ax.text(ax.get_xlim()[1] * 0.05, ax.get_ylim()[1] * 0.95, 'Высокий потенциал\nНизкий риск', 
        fontsize=10, alpha=0.5, style='italic')
ax.text(ax.get_xlim()[1] * 0.75, ax.get_ylim()[1] * 0.95, 'Высокий потенциал\nВысокий риск', 
        fontsize=10, alpha=0.5, style='italic')

plt.tight_layout()

plt.savefig('/home/ubuntu/chart_risk_potential.png', dpi=150, bbox_inches='tight')
charts_base64['risk_potential'] = fig_to_base64(fig)
plt.close()

# Сохранение base64 данных
with open('/home/ubuntu/charts_base64.pkl', 'wb') as f:
    pickle.dump(charts_base64, f)

print("\n✓ Все графики созданы и сохранены!")
print("  - chart_price_comparison.png")
print("  - chart_returns.png")
print("  - chart_rsi.png")
print("  - chart_fundamentals.png")
print("  - chart_price_ma.png")
print("  - chart_risk_potential.png")

