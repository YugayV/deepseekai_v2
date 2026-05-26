import pickle
import pandas as pd
import numpy as np

print("=" * 80)
print("ПЛАН РАСПРЕДЕЛЕНИЯ ПОРТФЕЛЯ $25,000")
print("=" * 80)

# Загрузка данных
with open('/home/ubuntu/analysis_data.pkl', 'rb') as f:
    data = pickle.load(f)

with open('/home/ubuntu/forecasts_ratings.pkl', 'rb') as f:
    forecast_data = pickle.load(f)

current_data = data['current_data']
forecasts = forecast_data['forecasts']
scores = forecast_data['scores']
rating_df = forecast_data['rating_df']
top_symbol = forecast_data['top_symbol']

# Портфель
total_capital = 25000

# Стратегия распределения: 
# - Концентрация на топ-3 акции (70%)
# - Диверсификация в остальные (30%)
# - Больший вес лидеру

print("\n📊 РЕКОМЕНДУЕМАЯ СТРАТЕГИЯ:")
print("Умеренно-агрессивный рост с концентрацией на лидеров\n")

# Получаем топ-3
top3_symbols = rating_df.head(3)['Акция'].tolist()
other_symbols = rating_df.tail(3)['Акция'].tolist()

# Распределение весов
allocations = {}

# ТОП-1: 35% (максимальный потенциал, но не слишком рискованно)
allocations[top3_symbols[0]] = 0.35

# ТОП-2: 20%
allocations[top3_symbols[1]] = 0.20

# ТОП-3: 15%
allocations[top3_symbols[2]] = 0.15

# Остальные: по 10% каждая
for symbol in other_symbols:
    allocations[symbol] = 0.10

# Расчет размеров позиций
positions = []

print("ДЕТАЛЬНОЕ РАСПРЕДЕЛЕНИЕ:")
print("-" * 80)

for symbol in rating_df['Акция'].tolist():
    weight = allocations[symbol]
    amount = total_capital * weight
    price = current_data[symbol]['price']
    shares = int(amount / price)
    actual_amount = shares * price
    actual_weight = (actual_amount / total_capital) * 100
    
    target_price = forecasts[symbol]['weighted_target']
    upside = forecasts[symbol]['weighted_upside']
    
    positions.append({
        'symbol': symbol,
        'weight': weight * 100,
        'amount': actual_amount,
        'shares': shares,
        'current_price': price,
        'target_price': target_price,
        'upside': upside,
        'potential_value': shares * target_price
    })
    
    print(f"\n{symbol}:")
    print(f"  Вес портфеля: {weight*100:.1f}% (${actual_amount:,.2f})")
    print(f"  Количество акций: {shares}")
    print(f"  Цена входа: ${price:.2f}")
    print(f"  Целевая цена: ${target_price:.2f}")
    print(f"  Потенциальный доход: ${shares * target_price - actual_amount:,.2f} ({upside:.1f}%)")

# Итоговая таблица
portfolio_df = pd.DataFrame(positions)
total_invested = portfolio_df['amount'].sum()
total_shares_value = portfolio_df['potential_value'].sum()
total_gain = total_shares_value - total_invested
total_return_pct = (total_gain / total_invested) * 100

print("\n\n" + "=" * 80)
print("СВОДНАЯ ТАБЛИЦА ПОРТФЕЛЯ")
print("=" * 80)

summary_table = pd.DataFrame({
    'Акция': portfolio_df['symbol'],
    'Акций': portfolio_df['shares'],
    'Цена входа': [f"${x:.2f}" for x in portfolio_df['current_price']],
    'Инвестиция': [f"${x:,.0f}" for x in portfolio_df['amount']],
    'Вес %': [f"{(x/total_invested)*100:.1f}%" for x in portfolio_df['amount']],
    'Цель': [f"${x:.2f}" for x in portfolio_df['target_price']],
    'Потенциал': [f"+{x:.1f}%" for x in portfolio_df['upside']],
    'Целевая стоимость': [f"${x:,.0f}" for x in portfolio_df['potential_value']]
})

print("\n")
print(summary_table.to_string(index=False))

print(f"\n\nИТОГО:")
print(f"  Всего инвестировано:      ${total_invested:,.2f}")
print(f"  Целевая стоимость:        ${total_shares_value:,.2f}")
print(f"  Потенциальная прибыль:    ${total_gain:,.2f}")
print(f"  Общая доходность:         +{total_return_pct:.1f}%")

print("\n\n📈 ПЛАН ВХОДА В ПОЗИЦИИ:")
print("-" * 80)
print("""
Учитывая рыночные условия (май 2026), рекомендуется:

ВАРИАНТ 1: Постепенный вход (РЕКОМЕНДУЕТСЯ)
  • Неделя 1-2: Купить 50% каждой позиции (особенно AAPL - RSI=90)
  • Неделя 3-4: Докупить 30% при коррекции или стабилизации
  • Неделя 5-6: Завершить позиции оставшимися 20%
  
  Преимущества: снижение риска входа на локальных максимумах,
                усреднение цены входа

ВАРИАНТ 2: Единовременный вход
  • Если готовы принять краткосрочную волатильность
  • Подходит при сильном убеждении в bull-сценарии
  • Требует психологической готовности к просадкам 10-15%

ВНИМАНИЕ НА:
  • AAPL: RSI=90 - вероятна коррекция, лучше входить траншами
  • NVDA: Высокая волатильность - стоп-лосс рекомендуется на уровне -12%
  • TSLA: Высокий риск (P/E=391) - самая маленькая позиция оправдана
""")

print("\n🎯 КЛЮЧЕВЫЕ УРОВНИ ДЛЯ МОНИТОРИНГА:")
print("-" * 80)

for idx, row in portfolio_df.iterrows():
    symbol = row['symbol']
    current_price = row['current_price']
    # Стоп-лосс: -12% от текущей цены
    stop_loss = current_price * 0.88
    # Первый тейк-профит: +25%
    take_profit_1 = current_price * 1.25
    # Финальная цель
    target = row['target_price']
    
    print(f"\n{symbol}:")
    print(f"  🔴 Стоп-лосс:        ${stop_loss:.2f} (-12%)")
    print(f"  🟡 Тейк-профит 1:    ${take_profit_1:.2f} (+25% - зафиксировать 30-50% позиции)")
    print(f"  🟢 Целевая цена:     ${target:.2f} ({forecasts[symbol]['weighted_upside']:.1f}%)")

print("\n\n💡 ДОПОЛНИТЕЛЬНЫЕ РЕКОМЕНДАЦИИ:")
print("-" * 80)
print("""
1. РЕБАЛАНСИРОВКА:
   - Проверять портфель каждый месяц
   - При достижении +25% любой позицией - зафиксировать частично
   - При превышении веса 40% одной акции - ребалансировать

2. УПРАВЛЕНИЕ РИСКАМИ:
   - Не добавлять в позиции при RSI > 75
   - Соблюдать стоп-лоссы, особенно для NVDA и TSLA
   - При общей просадке портфеля >15% - пересмотр тезисов

3. КАТАЛИЗАТОРЫ ДЛЯ ОТСЛЕЖИВАНИЯ:
   - NVDA: Результаты Q2 FY27 (август 2026)
   - GOOGL: Развитие Gemini AI, рост Cloud
   - AMZN: FCF после пика capex, развитие AWS
   - AAPL: Цикл iPhone, развитие Apple Intelligence
   - MSFT: Монетизация Azure AI
   - TSLA: Q2-Q3 2026 earnings, FSD/robotaxi прогресс

4. МАКРО-РИСКИ:
   - Процентные ставки ФРС
   - Геополитическая напряженность
   - Регуляторные риски (особенно GOOGL, TSLA)
   - Конкуренция в AI (риск для NVDA, MSFT, GOOGL)
""")

# Сохранение портфеля
with open('/home/ubuntu/portfolio_allocation.pkl', 'wb') as f:
    pickle.dump({
        'positions': positions,
        'portfolio_df': portfolio_df,
        'total_invested': total_invested,
        'total_target_value': total_shares_value,
        'expected_return': total_return_pct
    }, f)

print("\n✓ План распределения портфеля создан и сохранен.")
print("=" * 80)

