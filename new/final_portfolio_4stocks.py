import pandas as pd

print("=" * 80)
print("ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ: ПОРТФЕЛЬ ИЗ 4 АКЦИЙ")
print("=" * 80)

# Результаты анализа
stocks = {
    'GOOGL': {
        'price': 382.26,
        'target': 507.82,
        'upside': 32.8,
        'scenario': 'Bull: +50.6%, Base: +29.7%, Bear: +8.8%',
        'reasoning': 'Cloud ускоряется (+63% YoY), Gemini AI, backlog $460B',
        'risks': 'Антимонопольные риски, YouTube headwinds'
    },
    'MSFT': {
        'price': 417.77,
        'target': 519.28,
        'upside': 24.3,
        'scenario': 'Bull: +42.3%, Base: +23.4%, Bear: +4.4%',
        'reasoning': 'Azure AI лидер, Copilot monetization, качество 97/100',
        'risks': 'Weak recent momentum, конкуренция в AI'
    },
    'AMZN': {
        'price': 265.32,
        'target': 299.81,
        'upside': 13.0,
        'scenario': 'Bull: +31.4%, Base: +12.0%, Bear: -7.3%',
        'reasoning': 'AWS доминирование, margin expansion тренд',
        'risks': 'Negative FCF -$19B в 2026 (capex $200B)'
    },
    'NVDA': {
        'price': 214.28,
        'target': 231.65,
        'upside': 8.1,
        'scenario': 'Bull: +26.5%, Base: +6.7%, Bear: -20.9%',
        'reasoning': 'Data Center 92% revenue, Blackwell ramp, gross margin 75%',
        'risks': 'Custom chips конкуренция, высокая база'
    }
}

print("\n📊 РЕКОМЕНДОВАННЫЕ 4 АКЦИИ:\n")

for symbol, data in stocks.items():
    print(f"{symbol}: ${data['price']:.2f} → ${data['target']:.2f} (+{data['upside']:.1f}%)")
    print(f"  Сценарии: {data['scenario']}")
    print(f"  Почему: {data['reasoning']}")
    print(f"  Риски: {data['risks']}")
    print()

print("\n💼 РАСПРЕДЕЛЕНИЕ ПОРТФЕЛЯ $25,000:\n")

# Новое распределение на 4 акции
weights = {
    'GOOGL': 0.35,  # максимальный вес лидеру
    'MSFT': 0.30,
    'AMZN': 0.20,
    'NVDA': 0.15
}

total_capital = 25000
portfolio = []
total_invested = 0
total_target = 0

print(f"{'Акция':<8} {'Вес':>6} {'Инвестиция':>13} {'Акций':>7} {'Цена':>10} {'Цель':>10} {'Доход':>10}")
print("-" * 85)

for symbol, weight in weights.items():
    data = stocks[symbol]
    amount = total_capital * weight
    price = data['price']
    shares = int(amount / price)
    invested = shares * price
    target_price = data['target']
    target_value = shares * target_price
    gain_pct = ((target_value - invested) / invested) * 100
    
    total_invested += invested
    total_target += target_value
    
    print(f"{symbol:<8} {weight*100:>5.1f}%  ${invested:>11,.0f}  {shares:>7}  ${price:>9.2f}  ${target_price:>9.2f}  {gain_pct:>+8.1f}%")

total_gain = total_target - total_invested
total_return = (total_gain / total_invested) * 100
cash = total_capital - total_invested

print("-" * 85)
print(f"{'ИТОГО':<8}        ${total_invested:>11,.0f}           ${total_target:>23,.0f}  {total_return:>+8.1f}%")
print(f"\nОстаток cash: ${cash:,.2f} (резерв для докупок)")

print(f"\n\n✅ ИТОГО ПО ПОРТФЕЛЮ:")
print(f"  Инвестировано: ${total_invested:,.2f}")
print(f"  Целевая стоимость: ${total_target:,.2f}")
print(f"  Ожидаемая прибыль: ${total_gain:,.2f}")
print(f"  Ожидаемая доходность: +{total_return:.1f}%")
print(f"  Горизонт: 6-12 месяцев")

print("\n\n❌ ПОЧЕМУ НЕ ВКЛЮЧЕНЫ AAPL И TSLA:")
print("""
AAPL ($308.40):
  ✗ Текущий P/E 33.9 выше исторического среднего
  ✗ Forward EPS $8.51 → справедливая цена $272-281
  ✗ Ожидаемый потенциал: -11.7% (downside)
  ✗ RSI=90 - сильная перекупленность
  ✗ Mature бизнес с ограниченным ростом (14%)
  → Вердикт: ПЕРЕОЦЕНЕНА, избегать или короткая позиция

TSLA ($423.67):
  ✗ Экстремальный P/E 390.8 (в 10-15х выше разумного)
  ✗ Forward EPS $1.33-2.50 → справедливая цена $87-150
  ✗ Ожидаемый потенциал: -68.4% (massive downside!)
  ✗ Высокий риск: margin pressure, execution, sentiment
  ✗ Вероятность bull-сценария всего 15%
  → Вердикт: КРАЙНЕ ПЕРЕОЦЕНЕНА, строго избегать

Оба тикера исключены для защиты капитала.
""")

print("\n📅 ПЛАН ВХОДА (3 транша):\n")
print("""
Транш 1 (40%): Неделя 1
  GOOGL: купить 12 акций (~$4,587)
  MSFT:  купить  7 акций (~$2,925)
  AMZN:  купить 10 акций (~$2,653)
  NVDA:  купить 10 акций (~$2,143)
  Итого: ~$12,308

Транш 2 (35%): Неделя 3
  GOOGL: купить  8 акций (~$3,058)
  MSFT:  купить  5 акций (~$2,089)
  AMZN:  купить  7 акций (~$1,857)
  NVDA:  купить  7 акций (~$1,500)
  Итого: ~$8,504

Транш 3 (25%): Неделя 5
  GOOGL: купить  4 акций (~$1,529)
  MSFT:  купить  3 акций (~$1,253)
  AMZN:  купить  4 акций (~$1,061)
  NVDA:  купить  4 акций (~$857)
  Итого: ~$4,700

TOTAL: $25,512 (округление)
Средняя цена входа будет усреднена через 3 транша.
""")

print("\n🎯 СТОП-ЛОССЫ И МОНИТОРИНГ:\n")

for symbol in weights.keys():
    price = stocks[symbol]['price']
    if symbol in ['NVDA']:
        stop_pct = -15
    elif symbol in ['GOOGL', 'AMZN']:
        stop_pct = -12
    else:
        stop_pct = -10
    
    stop_price = price * (1 + stop_pct/100)
    
    if symbol == 'NVDA':
        take1 = price * 1.15
        take2 = price * 1.25
    else:
        take1 = price * 1.20
        take2 = stocks[symbol]['target']
    
    print(f"{symbol}:")
    print(f"  🔴 Стоп-лосс:     ${stop_price:>7.2f} ({stop_pct:>3}%)")
    print(f"  🟡 Тейк-профит 1: ${take1:>7.2f} (зафиксировать 30%)")
    print(f"  🟢 Финальная цель: ${take2:>7.2f}")
    print()

print("\n⚠️  КЛЮЧЕВЫЕ РИСКИ:")
print("""
1. МАКРО: Высокие ставки ФРС давят на tech valuations
2. ГЕОПОЛИТИКА: Ближний Восток (NVDA supply), China (trade)
3. РЕГУЛЯЦИЯ: Антимонопольные расследования (GOOGL)
4. AI КОНКУРЕНЦИЯ: Custom chips угрожают NVDA margin
5. КАПЕКС: AMZN negative FCF в 2026 может волновать рынок

Мониторинг обязателен! Ежемесячная ревизия портфеля.
""")

print("\n✓ Финальные рекомендации подготовлены.")
print("=" * 80)

# Сохранение
import pickle
with open('/home/ubuntu/final_portfolio_4stocks.pkl', 'wb') as f:
    pickle.dump({
        'stocks': stocks,
        'weights': weights,
        'total_return': total_return,
        'total_invested': total_invested,
        'total_target': total_target
    }, f)

