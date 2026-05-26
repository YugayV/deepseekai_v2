import pandas as pd
import numpy as np
import json

print("=" * 80)
print("КОРРЕКТИРОВАННЫЙ АНАЛИЗ 6 ТЕХНОЛОГИЧЕСКИХ АКЦИЙ")
print("=" * 80)

# ТОЧНЫЕ ДАННЫЕ НА 22 МАЯ 2026
stocks_data = {
    'NVDA': {
        'price': 214.28,
        'forward_pe': 25.43,
        'forward_eps_fy27': 8.47,
        'analyst_target': 294.22,
        'analyst_rating': 'Strong Buy',
        'gross_margin': 75.0,
        'revenue_growth_yoy': 74.0,
        'datacenter_pct': 92,
        'market_cap_t': 5.21
    },
    'GOOGL': {
        'price': 382.26,
        'forward_pe': 30.60,
        'eps_ttm': 13.11,
        'analyst_target': 429.12,
        'analyst_rating': 'Strong Buy',
        'operating_margin': 32.69,
        'revenue_growth_q1': 22.0,
        'cloud_growth_yoy': 63.0,
        'fcf_ltm': 64.43,
        'market_cap_t': 4.64
    },
    'AMZN': {
        'price': 265.32,
        'forward_pe': 29.59,
        'forward_eps_2026': 9.00,
        'forward_eps_2027': 10.25,
        'analyst_target': 312.63,
        'analyst_rating': 'Strong Buy',
        'aws_growth_yoy': 24.0,
        'fcf_2026': -18.79,  # negative due to capex
        'capex_2026': 200.0,
        'market_cap_t': 2.65
    },
    'AAPL': {
        'price': 308.40,
        'forward_pe': 33.94,
        'forward_eps_fy26': 8.51,
        'analyst_target': 295.44,
        'analyst_rating': 'Buy',
        'operating_margin': 32.64,
        'fcf_fy26': 137.49,
        'eps_growth_fy26': 14.1,
        'market_cap_t': 4.54
    },
    'MSFT': {
        'price': 417.77,
        'forward_pe_fy26': 24.46,
        'forward_pe_fy27': 21.12,
        'forward_eps_fy26': 17.11,
        'forward_eps_fy27': 19.82,
        'analyst_rating': 'Strong Buy',
        'revenue_growth_fy26': 19.3,
        'eps_growth_fy26': 25.46,
        'market_cap_t': 3.11
    },
    'TSLA': {
        'price': 423.67,
        'forward_pe': 390.83,  # extreme
        'eps_ttm': 1.09,
        'forward_eps_2026': 1.33,
        'analyst_target': None,
        'analyst_rating': 'Mixed',
        'eps_growth_forecast': 42.5,
        'revenue_growth_q1': 15.8,
        'market_cap_t': 1.60
    }
}

print("\n1. ТЕКУЩЕЕ СОСТОЯНИЕ (22 МАЯ 2026):\n")

summary = []
for symbol, data in stocks_data.items():
    summary.append({
        'Акция': symbol,
        'Цена': f"${data['price']:.2f}",
        'Forward P/E': f"{data.get('forward_pe', data.get('forward_pe_fy26', 'N/A'))}",
        'Кап.(трлн)': f"${data['market_cap_t']:.2f}",
        'Рейтинг': data['analyst_rating']
    })

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

print("\n\n2. МЕТОДОЛОГИЯ ЦЕЛЕВЫХ ЦЕН:\n")
print("""
Используем три подхода:
1. Forward EPS × разумный диапазон P/E multiples
2. Текущие analyst consensus targets
3. Сценарный анализ с вероятностными взвешиваниями

Для каждой акции:
- Bull: благоприятное развитие + multiple expansion
- Base: выполнение ожиданий + текущие multiples
- Bear: разочарование + multiple compression
""")

print("\n3. ЦЕЛЕВЫЕ ЦЕНЫ НА 6-12 МЕСЯЦЕВ:\n")

targets = {}

# NVDA - используем forward EPS FY27
nvda_base_pe = 25.0  # разумный для mega-cap growth
nvda_bull_pe = 30.0  # при продолжении AI boom
nvda_bear_pe = 18.0  # при замедлении / конкуренции

targets['NVDA'] = {
    'bull': stocks_data['NVDA']['forward_eps_fy27'] * nvda_bull_pe,
    'base': stocks_data['NVDA']['forward_eps_fy27'] * nvda_base_pe,
    'bear': stocks_data['NVDA']['forward_eps_fy27'] * nvda_bear_pe,
    'analyst': stocks_data['NVDA']['analyst_target'],
    'prob': {'bull': 0.40, 'base': 0.45, 'bear': 0.15}
}

# GOOGL - используем текущий EPS с ростом 20%
googl_eps_projected = stocks_data['GOOGL']['eps_ttm'] * 1.20
googl_base_pe = 30.0
googl_bull_pe = 35.0
googl_bear_pe = 25.0

targets['GOOGL'] = {
    'bull': googl_eps_projected * googl_bull_pe,
    'base': googl_eps_projected * googl_base_pe,
    'bear': googl_eps_projected * googl_bear_pe,
    'analyst': stocks_data['GOOGL']['analyst_target'],
    'prob': {'bull': 0.35, 'base': 0.45, 'bear': 0.20}
}

# AMZN - используем forward EPS 2027
amzn_base_pe = 28.0
amzn_bull_pe = 33.0
amzn_bear_pe = 23.0

targets['AMZN'] = {
    'bull': stocks_data['AMZN']['forward_eps_2027'] * amzn_bull_pe,
    'base': stocks_data['AMZN']['forward_eps_2027'] * amzn_base_pe,
    'bear': stocks_data['AMZN']['forward_eps_2027'] * amzn_bear_pe,
    'analyst': stocks_data['AMZN']['analyst_target'],
    'prob': {'bull': 0.30, 'base': 0.45, 'bear': 0.25}
}

# AAPL - используем forward EPS FY26
aapl_base_pe = 33.0
aapl_bull_pe = 38.0
aapl_bear_pe = 28.0

targets['AAPL'] = {
    'bull': stocks_data['AAPL']['forward_eps_fy26'] * aapl_bull_pe,
    'base': stocks_data['AAPL']['forward_eps_fy26'] * aapl_base_pe,
    'bear': stocks_data['AAPL']['forward_eps_fy26'] * aapl_bear_pe,
    'analyst': stocks_data['AAPL']['analyst_target'],
    'prob': {'bull': 0.30, 'base': 0.45, 'bear': 0.25}
}

# MSFT - используем forward EPS FY27
msft_base_pe = 26.0
msft_bull_pe = 30.0
msft_bear_pe = 22.0

targets['MSFT'] = {
    'bull': stocks_data['MSFT']['forward_eps_fy27'] * msft_bull_pe,
    'base': stocks_data['MSFT']['forward_eps_fy27'] * msft_base_pe,
    'bear': stocks_data['MSFT']['forward_eps_fy27'] * msft_bear_pe,
    'analyst': None,
    'prob': {'bull': 0.30, 'base': 0.45, 'bear': 0.25}
}

# TSLA - самый сложный, используем консервативный подход
tsla_eps_projected = stocks_data['TSLA']['forward_eps_2026'] * 1.20  # небольшой рост
tsla_base_pe = 60.0  # снижение с текущего extreme
tsla_bull_pe = 100.0
tsla_bear_pe = 40.0

targets['TSLA'] = {
    'bull': tsla_eps_projected * tsla_bull_pe,
    'base': tsla_eps_projected * tsla_base_pe,
    'bear': tsla_eps_projected * tsla_bear_pe,
    'analyst': None,
    'prob': {'bull': 0.20, 'base': 0.35, 'bear': 0.45}  # высокий риск
}

# Расчет взвешенных целей и потенциала
for symbol in stocks_data.keys():
    target = targets[symbol]
    prob = target['prob']
    
    weighted = (target['bull'] * prob['bull'] + 
               target['base'] * prob['base'] + 
               target['bear'] * prob['bear'])
    
    current = stocks_data[symbol]['price']
    upside_bull = ((target['bull'] - current) / current) * 100
    upside_base = ((target['base'] - current) / current) * 100
    downside_bear = ((target['bear'] - current) / current) * 100
    upside_weighted = ((weighted - current) / current) * 100
    
    targets[symbol]['weighted'] = weighted
    targets[symbol]['upside_bull'] = upside_bull
    targets[symbol]['upside_base'] = upside_base
    targets[symbol]['downside_bear'] = downside_bear
    targets[symbol]['upside_weighted'] = upside_weighted
    
    print(f"{symbol}:")
    print(f"  Текущая цена: ${current:.2f}")
    print(f"  Bull:  ${target['bull']:.2f} ({upside_bull:+.1f}%) [вероятность {prob['bull']*100:.0f}%]")
    print(f"  Base:  ${target['base']:.2f} ({upside_base:+.1f}%) [вероятность {prob['base']*100:.0f}%]")
    print(f"  Bear:  ${target['bear']:.2f} ({downside_bear:+.1f}%) [вероятность {prob['bear']*100:.0f}%]")
    print(f"  Взвешенная цель: ${weighted:.2f} ({upside_weighted:+.1f}%)")
    if target['analyst']:
        print(f"  Analyst consensus: ${target['analyst']:.2f}")
    print()

print("\n4. РЕЙТИНГ И РЕКОМЕНДАЦИИ:\n")

# Простой рейтинг на основе взвешенного потенциала и качества
ranking_data = []

for symbol in stocks_data.keys():
    score = targets[symbol]['upside_weighted']
    
    # Корректировки
    if symbol == 'NVDA':
        quality_adj = 1.2  # лидер в AI
    elif symbol == 'GOOGL':
        quality_adj = 1.15  # сильный фундамент
    elif symbol == 'AMZN':
        quality_adj = 1.05  # AWS лидер
    elif symbol == 'AAPL':
        quality_adj = 1.0  # качественная, но mature
    elif symbol == 'MSFT':
        quality_adj = 1.1  # качество, но weak momentum
    else:  # TSLA
        quality_adj = 0.7  # высокий риск
    
    adjusted_score = score * quality_adj
    
    ranking_data.append({
        'Акция': symbol,
        'Текущая цена': f"${stocks_data[symbol]['price']:.2f}",
        'Цель(взв.)': f"${targets[symbol]['weighted']:.2f}",
        'Потенциал': f"{targets[symbol]['upside_weighted']:+.1f}%",
        'Скор': round(adjusted_score, 1),
        'Рейтинг': stocks_data[symbol]['analyst_rating']
    })

ranking_df = pd.DataFrame(ranking_data)
ranking_df = ranking_df.sort_values('Скор', ascending=False)

print(ranking_df.to_string(index=False))

top_symbol = ranking_df.iloc[0]['Акция']
print(f"\n🏆 ЛУЧШАЯ ВОЗМОЖНОСТЬ: {top_symbol}")
print(f"Потенциал: {targets[top_symbol]['upside_weighted']:+.1f}%")

print("\n5. ПЛАН РАСПРЕДЕЛЕНИЯ $25,000:\n")

# Более консервативное распределение после корректировки
allocations = {
    ranking_df.iloc[0]['Акция']: 0.30,  # ТОП-1: 30%
    ranking_df.iloc[1]['Акция']: 0.25,  # ТОП-2: 25%
    ranking_df.iloc[2]['Акция']: 0.20,  # ТОП-3: 20%
    ranking_df.iloc[3]['Акция']: 0.15,  # ТОП-4: 15%
    ranking_df.iloc[4]['Акция']: 0.07,  # ТОП-5: 7%
    ranking_df.iloc[5]['Акция']: 0.03,  # ТОП-6: 3%
}

total_capital = 25000.0
portfolio = []
total_value = 0
total_target_value = 0

print(f"{'Акция':<8} {'Вес':>6} {'Инвестиция':>12} {'Акций':>6} {'Цена':>9} {'Целевая':>10} {'Потенциал':>11}")
print("-" * 80)

for symbol, weight in allocations.items():
    amount = total_capital * weight
    price = stocks_data[symbol]['price']
    shares = int(amount / price)
    actual_amount = shares * price
    target_price = targets[symbol]['weighted']
    target_value = shares * target_price
    potential = ((target_value - actual_amount) / actual_amount) * 100
    
    total_value += actual_amount
    total_target_value += target_value
    
    portfolio.append({
        'symbol': symbol,
        'shares': shares,
        'amount': actual_amount,
        'target_value': target_value
    })
    
    print(f"{symbol:<8} {weight*100:5.1f}%  ${actual_amount:>10,.0f}  {shares:>6}  ${price:>8.2f}  ${target_price:>9.2f}  {potential:>+9.1f}%")

total_return = ((total_target_value - total_value) / total_value) * 100
cash_remaining = total_capital - total_value

print("-" * 80)
print(f"{'ИТОГО':<8}        ${total_value:>10,.0f}          ${total_target_value:>21,.0f}  {total_return:>+9.1f}%")
print(f"\nОстаток cash: ${cash_remaining:,.2f}")
print(f"Ожидаемая доходность портфеля: {total_return:+.1f}%")

print("\n\n6. ВАЖНЫЕ ЗАМЕЧАНИЯ:\n")
print("""
⚠️  КРИТИЧЕСКИЕ МОМЕНТЫ:

1. КАЧЕСТВО ДАННЫХ И УВЕРЕННОСТЬ:
   - Анализ основан на публичных данных по состоянию на 22.05.2026
   - Forward multiples - это консенсус аналитиков, не гарантия
   - Уровень уверенности: УМЕРЕННЫЙ (не high-conviction)

2. ОСНОВНЫЕ РИСКИ:
   - NVDA: Высокая волатильность, конкуренция custom chips
   - GOOGL: Антимонопольные риски, YouTube headwinds
   - AMZN: Negative FCF в 2026 из-за capex $200B
   - AAPL: Mature бизнес, China exposure
   - MSFT: Weak recent momentum
   - TSLA: Extreme valuation (P/E 391), execution risk

3. УПРАВЛЕНИЕ РИСКАМИ:
   - Не рекомендуется единовременный вход
   - Постепенный вход траншами по 33% каждые 2-3 недели
   - Стоп-лоссы индивидуальны по волатильности:
     * NVDA, TSLA: -15%
     * GOOGL, AMZN: -12%
     * AAPL, MSFT: -10%

4. МОНИТОРИНГ:
   - Ежемесячная ревизия портфеля
   - Отслеживание earnings dates и guidance
   - Макро-факторы: ставки ФРС, геополитика, регуляция

5. ГОРИЗОНТ И ОЖИДАНИЯ:
   - Целевой горизонт: 6-12 месяцев
   - Ожидаемая доходность: умеренно оптимистична
   - Возможная просадка портфеля: до -20% краткосрочно
""")

# Сохранение
with open('/home/ubuntu/corrected_data.pkl', 'wb') as f:
    import pickle
    pickle.dump({
        'stocks_data': stocks_data,
        'targets': targets,
        'ranking_df': ranking_df,
        'portfolio': portfolio,
        'total_return': total_return
    }, f)

print("\n✓ Корректированный анализ завершен.")
print("=" * 80)

