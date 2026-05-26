import pandas as pd
import numpy as np

print("=" * 80)
print("ФИНАЛЬНЫЙ КОРРЕКТИРОВАННЫЙ АНАЛИЗ")
print("=" * 80)

# ТОЧНЫЕ ДАННЫЕ НА 22 МАЯ 2026
stocks_data = {
    'NVDA': {
        'price': 214.28,
        'forward_eps_fy27': 8.47,
        'market_cap_t': 5.21,
        'growth': 'очень высокий (74% YoY)',
        'quality': 'высокое (gross margin 75%)'
    },
    'GOOGL': {
        'price': 382.26,
        'eps_ttm': 13.11,
        'forward_pe': 30.60,
        'market_cap_t': 4.64,
        'growth': 'высокий (22% revenue, 63% Cloud)',
        'quality': 'отличное (margin 32.7%, FCF $64B)'
    },
    'AMZN': {
        'price': 265.32,
        'forward_eps_2027': 10.25,
        'market_cap_t': 2.65,
        'growth': 'умеренный (11% CAGR)',
        'quality': 'хорошее (AWS лидер, но FCF -$19B в 2026)'
    },
    'AAPL': {
        'price': 308.40,
        'forward_eps_fy26': 8.51,
        'forward_pe': 33.94,
        'market_cap_t': 4.54,
        'growth': 'умеренный (14% EPS)',
        'quality': 'отличное (margin 32.6%, FCF $137B)'
    },
    'MSFT': {
        'price': 417.77,
        'forward_eps_fy27': 19.82,
        'forward_pe': 21.12,
        'market_cap_t': 3.11,
        'growth': 'высокий (19% revenue, 25% EPS)',
        'quality': 'отличное (лидер cloud/AI)'
    },
    'TSLA': {
        'price': 423.67,
        'eps_ttm': 1.09,
        'forward_eps_2026': 1.33,
        'current_pe': 390.83,
        'market_cap_t': 1.60,
        'growth': 'умеренный (16% revenue)',
        'quality': 'смешанное (высокий риск, margin pressure)'
    }
}

print("\n1. СЦЕНАРИИ ОЦЕНКИ (6-12 месяцев):\n")

# ПРАВИЛЬНАЯ МЕТОДОЛОГИЯ: forward EPS × realistic P/E ranges

valuations = {}

# NVDA
nvda_eps = stocks_data['NVDA']['forward_eps_fy27']
valuations['NVDA'] = {
    'bull': {'price': nvda_eps * 32, 'pe': 32, 'desc': 'AI boom продолжается'},
    'base': {'price': nvda_eps * 27, 'pe': 27, 'desc': 'Стабильный рост'},
    'bear': {'price': nvda_eps * 20, 'pe': 20, 'desc': 'Замедление/конкуренция'},
    'prob': [0.35, 0.45, 0.20],
    'reasoning': 'AI infra лидер, но высокая база сравнения'
}

# GOOGL  
googl_eps_proj = stocks_data['GOOGL']['eps_ttm'] * 1.22  # 22% рост
valuations['GOOGL'] = {
    'bull': {'price': googl_eps_proj * 36, 'pe': 36, 'desc': 'Cloud ускорение + AI'},
    'base': {'price': googl_eps_proj * 31, 'pe': 31, 'desc': 'Устойчивый рост'},
    'bear': {'price': googl_eps_proj * 26, 'pe': 26, 'desc': 'Регуляторное давление'},
    'prob': [0.35, 0.45, 0.20],
    'reasoning': 'Сильная Cloud динамика, но антимонопольные риски'
}

# MSFT
msft_eps = stocks_data['MSFT']['forward_eps_fy27']
valuations['MSFT'] = {
    'bull': {'price': msft_eps * 30, 'pe': 30, 'desc': 'Azure AI монетизация'},
    'base': {'price': msft_eps * 26, 'pe': 26, 'desc': 'Стабильное выполнение'},
    'bear': {'price': msft_eps * 22, 'pe': 22, 'desc': 'Слабый momentum'},
    'prob': [0.30, 0.45, 0.25],
    'reasoning': 'Качество высокое, но weak recent trend'
}

# AMZN
amzn_eps = stocks_data['AMZN']['forward_eps_2027']
valuations['AMZN'] = {
    'bull': {'price': amzn_eps * 34, 'pe': 34, 'desc': 'AWS AI + margin expansion'},
    'base': {'price': amzn_eps * 29, 'pe': 29, 'desc': 'План выполнен'},
    'bear': {'price': amzn_eps * 24, 'pe': 24, 'desc': 'Capex concerns'},
    'prob': [0.30, 0.45, 0.25],
    'reasoning': 'AWS лидер, но negative FCF в 2026'
}

# AAPL
aapl_eps = stocks_data['AAPL']['forward_eps_fy26']
valuations['AAPL'] = {
    'bull': {'price': aapl_eps * 37, 'pe': 37, 'desc': 'Services + AI Intelligence'},
    'base': {'price': aapl_eps * 32, 'pe': 32, 'desc': 'Умеренный рост'},
    'bear': {'price': aapl_eps * 27, 'pe': 27, 'desc': 'China risks'},
    'prob': [0.25, 0.50, 0.25],
    'reasoning': 'Mature бизнес, ограниченный upside'
}

# TSLA - особый подход из-за extreme valuation
tsla_eps_optimistic = 2.50  # если выполнит прогнозы роста
valuations['TSLA'] = {
    'bull': {'price': tsla_eps_optimistic * 100, 'pe': 100, 'desc': 'FSD breakthrough'},
    'base': {'price': tsla_eps_optimistic * 60, 'pe': 60, 'desc': 'Постепенный рост'},
    'bear': {'price': tsla_eps_optimistic * 35, 'pe': 35, 'desc': 'Margin compression'},
    'prob': [0.15, 0.35, 0.50],
    'reasoning': 'Extreme current P/E 391, высочайший риск'
}

# Расчет взвешенных целей
results = []
for symbol in stocks_data.keys():
    val = valuations[symbol]
    current = stocks_data[symbol]['price']
    
    bull_price = val['bull']['price']
    base_price = val['base']['price']
    bear_price = val['bear']['price']
    
    prob = val['prob']
    weighted = bull_price * prob[0] + base_price * prob[1] + bear_price * prob[2]
    
    upside_weighted = ((weighted - current) / current) * 100
    upside_bull = ((bull_price - current) / current) * 100
    upside_base = ((base_price - current) / current) * 100
    upside_bear = ((bear_price - current) / current) * 100
    
    results.append({
        'symbol': symbol,
        'current': current,
        'bull': bull_price,
        'base': base_price,
        'bear': bear_price,
        'weighted': weighted,
        'upside_weighted': upside_weighted,
        'upside_bull': upside_bull,
        'upside_base': upside_base,
        'upside_bear': upside_bear
    })
    
    print(f"{symbol}: ${current:.2f} → ${weighted:.2f} ({upside_weighted:+.1f}%)")
    print(f"  Bull: ${bull_price:.2f} ({upside_bull:+.1f}%) - {val['bull']['desc']}")
    print(f"  Base: ${base_price:.2f} ({upside_base:+.1f}%) - {val['base']['desc']}")
    print(f"  Bear: ${bear_price:.2f} ({upside_bear:+.1f}%) - {val['bear']['desc']}")
    print(f"  Вероятности: {prob[0]:.0%}/{prob[1]:.0%}/{prob[2]:.0%}")
    print(f"  Обоснование: {val['reasoning']}")
    print()

print("\n2. РЕЙТИНГ ПО ПОТЕНЦИАЛУ:\n")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('upside_weighted', ascending=False)

ranking = []
for idx, row in results_df.iterrows():
    symbol = row['symbol']
    data = stocks_data[symbol]
    ranking.append({
        'Акция': symbol,
        'Цена': f"${row['current']:.2f}",
        'Цель': f"${row['weighted']:.2f}",
        'Потенциал': f"{row['upside_weighted']:+.1f}%",
        'Рост': data['growth'],
        'Качество': data['quality']
    })

ranking_df = pd.DataFrame(ranking)
print(ranking_df.to_string(index=False))

top_symbol = results_df.iloc[0]['symbol']
top_upside = results_df.iloc[0]['upside_weighted']

print(f"\n🏆 ЛУЧШАЯ ВОЗМОЖНОСТЬ: {top_symbol} ({top_upside:+.1f}%)")

print("\n\n3. РЕКОМЕНДОВАННОЕ РАСПРЕДЕЛЕНИЕ $25,000:\n")

# Консервативное распределение с учетом рисков
total_capital = 25000

# Топ-3 получают основной вес
weights = {
    results_df.iloc[0]['symbol']: 0.30,
    results_df.iloc[1]['symbol']: 0.25,
    results_df.iloc[2]['symbol']: 0.20,
    results_df.iloc[3]['symbol']: 0.15,
    results_df.iloc[4]['symbol']: 0.07,
    results_df.iloc[5]['symbol']: 0.03,  # минимальный вес для высокорисковых
}

portfolio = []
total_invested = 0
total_target = 0

print(f"{'Акция':<8} {'Вес':>6} {'Инвестиция':>13} {'Акций':>7} {'Цена':>10} {'Цель':>10} {'Доход':>10}")
print("-" * 85)

for symbol, weight in weights.items():
    amount = total_capital * weight
    price = stocks_data[symbol]['price']
    shares = int(amount / price)
    invested = shares * price
    
    row = results_df[results_df['symbol'] == symbol].iloc[0]
    target_price = row['weighted']
    target_value = shares * target_price
    gain_pct = ((target_value - invested) / invested) * 100
    
    total_invested += invested
    total_target += target_value
    
    portfolio.append({
        'symbol': symbol,
        'shares': shares,
        'invested': invested,
        'target': target_value
    })
    
    print(f"{symbol:<8} {weight*100:>5.1f}%  ${invested:>11,.0f}  {shares:>7}  ${price:>9.2f}  ${target_price:>9.2f}  {gain_pct:>+8.1f}%")

total_gain = total_target - total_invested
total_return_pct = (total_gain / total_invested) * 100
cash = total_capital - total_invested

print("-" * 85)
print(f"{'ИТОГО':<8}        ${total_invested:>11,.0f}           ${total_target:>23,.0f}  {total_return_pct:>+8.1f}%")
print(f"\nОстаток cash: ${cash:,.2f}")

print("\n\n4. ПЛАН ВХОДА И УПРАВЛЕНИЕ РИСКАМИ:\n")
print("""
📅 СТРАТЕГИЯ ВХОДА (Постепенный вход - РЕКОМЕНДУЕТСЯ):

Транш 1 (40% позиций): Неделя 1-2
  - Первичный вход во все позиции
  - Особое внимание: RSI и локальные уровни

Транш 2 (35% позиций): Неделя 3-4
  - Докупка при коррекции или стабилизации
  - Оценка первичной реакции рынка

Транш 3 (25% позиций): Неделя 5-6
  - Завершение формирования позиций
  - Итоговое усреднение

⚠️  СТОП-ЛОССЫ (индивидуальные по волатильности):
""")

for symbol in weights.keys():
    price = stocks_data[symbol]['price']
    if symbol in ['NVDA', 'TSLA']:
        stop_pct = -15
    elif symbol in ['GOOGL', 'AMZN']:
        stop_pct = -12
    else:
        stop_pct = -10
    
    stop_price = price * (1 + stop_pct/100)
    print(f"  {symbol}: ${stop_price:.2f} ({stop_pct}%)")

print("""

🎯 КЛЮЧЕВЫЕ КАТАЛИЗАТОРЫ:
  - GOOGL: Google Cloud рост, Gemini AI adoption
  - MSFT: Azure AI монетизация, Copilot uptake
  - NVDA: Q2 FY27 earnings (август), Blackwell ramp
  - AMZN: FCF recovery после peak capex 2026
  - AAPL: Apple Intelligence adoption, Services рост
  - TSLA: Q2/Q3 earnings, FSD/robotaxi прогресс

⚠️  МАКРО-РИСКИ:
  - Ставки ФРС (высокие ставки → давление на tech)
  - Геополитика (Ближний Восток → NVDA supply, China → AAPL/TSLA)
  - Регуляция (антимонопольные иски → GOOGL)
  - AI конкуренция (custom chips → риск для NVDA)
""")

print("\n5. ИТОГОВЫЕ РЕКОМЕНДАЦИИ:\n")

top3 = list(weights.keys())[:3]
print(f"""
✅ ТОП-3 ВЫБОРА:
1. {top3[0]}: Наибольший потенциал при приемлемом риске
2. {top3[1]}: Сильный second choice
3. {top3[2]}: Качественная диверсификация

Ожидаемая доходность портфеля: {total_return_pct:+.1f}%
Горизонт: 6-12 месяцев
Уровень риска: Умеренно-агрессивный

⚠️  НЕ РЕКОМЕНДУЕТСЯ единовременный вход - используйте транши!
""")

# Сохранение
import pickle
with open('/home/ubuntu/final_analysis.pkl', 'wb') as f:
    pickle.dump({
        'stocks_data': stocks_data,
        'valuations': valuations,
        'results': results_df,
        'portfolio': portfolio,
        'total_return': total_return_pct,
        'top_symbol': top_symbol
    }, f)

print("✓ Финальный анализ завершен и сохранен.")
print("=" * 80)

