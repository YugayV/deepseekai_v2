import pickle
import pandas as pd
import numpy as np
import json

print("=" * 80)
print("ПРОГНОЗ ЦЕЛЕВЫХ ЦЕН И СЦЕНАРИИ НА 6-12 МЕСЯЦЕВ")
print("=" * 80)

# Загрузка данных
with open('/home/ubuntu/analysis_data.pkl', 'rb') as f:
    data = pickle.load(f)

current_data = data['current_data']
analysis_summary = data['analysis_summary']
returns_data = data['returns_data']

# Фундаментальные катализаторы и данные (из веб-поиска)
fundamental_catalysts = {
    'MSFT': {
        'revenue_growth': 10,  # % YoY
        'margin_trend': 'stable',
        'ai_catalyst': 'Azure AI, enterprise solutions',
        'risk': 'AI monetization, competition',
        'sector_pe': 34.55,
        'quality_score': 97
    },
    'AMZN': {
        'revenue_growth': 11,  # CAGR through 2035
        'margin_trend': 'expanding',  # from 2.6% to 11.2%
        'ai_catalyst': 'AWS AI infrastructure, $200B capex',
        'risk': 'negative FCF in 2026 due to capex',
        'aws_growth': 24,  # % YoY Q4 2025
        'quality_score': 85
    },
    'AAPL': {
        'revenue_growth': 5,  # conservative estimate
        'margin_trend': 'stable',
        'ai_catalyst': 'Apple Intelligence, services growth',
        'risk': 'hardware cycle dependency, China exposure',
        'analyst_buy': 67,  # % buy ratings
        'quality_score': 90
    },
    'GOOGL': {
        'revenue_growth': 22,  # Q1 2026 YoY
        'margin_trend': 'expanding',  # 36.1% operating margin
        'ai_catalyst': 'Gemini AI, Cloud 63% growth, $460B backlog',
        'risk': 'antitrust, YouTube ad miss',
        'cloud_acceleration': 63,  # % YoY
        'quality_score': 95
    },
    'NVDA': {
        'revenue_growth': 85,  # Q1 FY27 YoY
        'margin_trend': 'stable',  # 75% gross margin
        'ai_catalyst': 'Data center dominance, Blackwell, Vera CPU',
        'risk': 'competition (custom chips), supply chain, valuation',
        'datacenter_share': 92,  # % of revenue
        'quality_score': 88
    },
    'TSLA': {
        'revenue_growth': 15.8,  # Q1 2026 YoY
        'margin_trend': 'compressed',
        'ai_catalyst': 'FSD, robotaxi, energy storage',
        'risk': 'extreme sentiment, margin pressure, execution',
        'eps_growth_forecast': 42.5,  # % next year
        'quality_score': 65
    }
}

# Функция для расчета целевых цен по сценариям
def calculate_target_prices(symbol, current_price, pe_ratio, fundamentals, technical):
    """
    Расчет целевых цен на 6-12 месяцев по трем сценариям
    """
    
    # Базовые параметры
    revenue_growth = fundamentals['revenue_growth'] / 100
    quality = fundamentals['quality_score']
    
    # Историческая волатильность
    volatility = technical['volatility'] / 100
    
    # RSI для оценки перекупленности/перепроданности
    rsi = technical['rsi']
    
    # Тренд
    trend = technical['trend']
    
    # === BULL SCENARIO (оптимистичный) ===
    # Предполагает: выполнение/превышение ожиданий, multiple expansion
    if trend == "Сильный восходящий":
        trend_multiplier = 1.3
    elif trend == "Восходящий":
        trend_multiplier = 1.2
    else:
        trend_multiplier = 1.1
    
    # Multiple expansion (зависит от качества и роста)
    if quality > 90 and revenue_growth > 0.15:
        multiple_expansion = 1.25
    elif quality > 85:
        multiple_expansion = 1.15
    else:
        multiple_expansion = 1.10
    
    # Рост прибыли (предполагаем 1.2x от роста выручки для bull case)
    earnings_growth_bull = 1 + (revenue_growth * 1.2)
    
    bull_target = current_price * earnings_growth_bull * multiple_expansion * trend_multiplier
    
    # === BASE SCENARIO (базовый) ===
    # Предполагает: выполнение ожиданий, стабильные multiples
    multiple_stable = 1.00
    earnings_growth_base = 1 + revenue_growth
    
    if trend == "Сильный восходящий" or trend == "Восходящий":
        trend_adj_base = 1.05
    else:
        trend_adj_base = 1.00
    
    base_target = current_price * earnings_growth_base * multiple_stable * trend_adj_base
    
    # === BEAR SCENARIO (пессимистичный) ===
    # Предполагает: невыполнение ожиданий, multiple compression
    multiple_compression = 0.85
    earnings_growth_bear = 1 + (revenue_growth * 0.5)  # Половина ожидаемого роста
    
    # Дополнительный риск для высоких P/E
    if pe_ratio > 100:
        pe_risk = 0.90
    elif pe_ratio > 50:
        pe_risk = 0.95
    else:
        pe_risk = 1.00
    
    bear_target = current_price * earnings_growth_bear * multiple_compression * pe_risk
    
    # Корректировки на основе RSI
    if rsi > 80:  # Сильно перекуплена
        bull_target *= 0.95
        base_target *= 0.97
    elif rsi < 30:  # Перепродана
        bull_target *= 1.05
        base_target *= 1.03
    
    # Вероятности сценариев
    if quality > 90 and trend == "Сильный восходящий":
        probabilities = {'bull': 0.40, 'base': 0.45, 'bear': 0.15}
    elif quality > 85 and trend in ["Сильный восходящий", "Восходящий"]:
        probabilities = {'bull': 0.35, 'base': 0.45, 'bear': 0.20}
    elif quality < 70 or pe_ratio > 200:
        probabilities = {'bull': 0.20, 'base': 0.40, 'bear': 0.40}
    else:
        probabilities = {'bull': 0.30, 'base': 0.45, 'bear': 0.25}
    
    # Взвешенная целевая цена
    weighted_target = (bull_target * probabilities['bull'] + 
                      base_target * probabilities['base'] + 
                      bear_target * probabilities['bear'])
    
    # Расчет потенциала (upside/downside)
    upside_bull = ((bull_target - current_price) / current_price) * 100
    upside_base = ((base_target - current_price) / current_price) * 100
    downside_bear = ((bear_target - current_price) / current_price) * 100
    weighted_upside = ((weighted_target - current_price) / current_price) * 100
    
    return {
        'bull_target': round(bull_target, 2),
        'base_target': round(base_target, 2),
        'bear_target': round(bear_target, 2),
        'weighted_target': round(weighted_target, 2),
        'upside_bull': round(upside_bull, 1),
        'upside_base': round(upside_base, 1),
        'downside_bear': round(downside_bear, 1),
        'weighted_upside': round(weighted_upside, 1),
        'probabilities': probabilities
    }

# Расчет прогнозов для всех акций
forecasts = {}

print("\n")
for symbol in current_data.keys():
    current_price = current_data[symbol]['price']
    pe_ratio = current_data[symbol]['pe_ratio']
    fundamentals = fundamental_catalysts[symbol]
    technical = analysis_summary[symbol]
    
    forecast = calculate_target_prices(symbol, current_price, pe_ratio, fundamentals, technical)
    forecasts[symbol] = forecast
    
    print(f"\n{symbol} ({current_data[symbol]['description']})")
    print("-" * 70)
    print(f"Текущая цена: ${current_price:.2f}")
    print(f"P/E Ratio: {pe_ratio:.1f}")
    print(f"Тренд: {technical['trend']}")
    print(f"RSI: {technical['rsi']:.0f}")
    print(f"\nПрогноз на 6-12 месяцев:")
    print(f"  🟢 Bull (оптимистичный):  ${forecast['bull_target']:>8.2f} (+{forecast['upside_bull']:>5.1f}%) - вероятность {forecast['probabilities']['bull']*100:.0f}%")
    print(f"  🟡 Base (базовый):        ${forecast['base_target']:>8.2f} (+{forecast['upside_base']:>5.1f}%) - вероятность {forecast['probabilities']['base']*100:.0f}%")
    print(f"  🔴 Bear (пессимистичный): ${forecast['bear_target']:>8.2f} ({forecast['downside_bear']:>5.1f}%) - вероятность {forecast['probabilities']['bear']*100:.0f}%")
    print(f"  ⚖️  Взвешенная цель:      ${forecast['weighted_target']:>8.2f} (+{forecast['weighted_upside']:>5.1f}%)")
    print(f"\nКлючевой катализатор: {fundamentals['ai_catalyst']}")
    print(f"Основной риск: {fundamentals['risk']}")

# === РЕЙТИНГОВАЯ СИСТЕМА ===
print("\n\n" + "=" * 80)
print("КОМПЛЕКСНЫЙ РЕЙТИНГ АКЦИЙ")
print("=" * 80)

# Функция расчета комплексного скора
def calculate_composite_score(symbol, forecast, fundamentals, technical, returns):
    """
    Комплексный скор на основе:
    1. Взвешенного потенциала роста (40%)
    2. Качества компании (25%)
    3. Технического подтверждения (20%)
    4. Оценки риска (15%)
    """
    
    # 1. Потенциал роста (0-100)
    upside_score = min(forecast['weighted_upside'], 100)
    
    # 2. Качество компании (0-100)
    quality_score = fundamentals['quality_score']
    
    # 3. Технический скор (0-100)
    trend_scores = {
        "Сильный восходящий": 90,
        "Восходящий": 75,
        "Боковой": 50,
        "Нисходящий": 30,
        "Сильный нисходящий": 10
    }
    trend_score = trend_scores.get(technical['trend'], 50)
    
    # RSI adjustment (перекупленность снижает скор)
    rsi = technical['rsi']
    if rsi > 80:
        rsi_adj = 0.9
    elif rsi < 30:
        rsi_adj = 1.1
    else:
        rsi_adj = 1.0
    
    technical_score = trend_score * rsi_adj
    
    # 4. Риск (0-100, обратный - меньше риска = выше скор)
    pe_ratio = current_data[symbol]['pe_ratio']
    volatility = technical['volatility']
    
    # Нормализация P/E (высокий P/E = выше риск)
    if pe_ratio > 200:
        pe_risk = 20
    elif pe_ratio > 100:
        pe_risk = 40
    elif pe_ratio > 50:
        pe_risk = 60
    elif pe_ratio > 30:
        pe_risk = 75
    else:
        pe_risk = 90
    
    # Нормализация волатильности
    vol_risk = max(0, 100 - volatility * 2)
    
    risk_score = (pe_risk + vol_risk) / 2
    
    # Взвешенный композитный скор
    composite = (upside_score * 0.40 + 
                quality_score * 0.25 + 
                technical_score * 0.20 + 
                risk_score * 0.15)
    
    return {
        'composite_score': round(composite, 1),
        'upside_score': round(upside_score, 1),
        'quality_score': round(quality_score, 1),
        'technical_score': round(technical_score, 1),
        'risk_score': round(risk_score, 1)
    }

# Расчет скоров для всех акций
scores = {}
for symbol in current_data.keys():
    scores[symbol] = calculate_composite_score(
        symbol, 
        forecasts[symbol],
        fundamental_catalysts[symbol],
        analysis_summary[symbol],
        returns_data[symbol]
    )

# Создание рейтинговой таблицы
rating_data = []
for symbol in current_data.keys():
    rating_data.append({
        'Акция': symbol,
        'Текущая цена': f"${current_data[symbol]['price']:.2f}",
        'Целевая (взв.)': f"${forecasts[symbol]['weighted_target']:.2f}",
        'Потенциал': f"{forecasts[symbol]['weighted_upside']:.1f}%",
        'Композитный скор': scores[symbol]['composite_score'],
        'Качество': scores[symbol]['quality_score'],
        'Технический': scores[symbol]['technical_score'],
        'Риск': scores[symbol]['risk_score']
    })

rating_df = pd.DataFrame(rating_data)
rating_df = rating_df.sort_values('Композитный скор', ascending=False)

print("\n")
print(rating_df.to_string(index=False))

# Определение ТОП-1 акции
top_symbol = rating_df.iloc[0]['Акция']
print(f"\n\n{'='*80}")
print(f"🏆 АКЦИЯ С НАИБОЛЬШИМ ПОТЕНЦИАЛОМ: {top_symbol}")
print(f"{'='*80}")
print(f"Композитный скор: {scores[top_symbol]['composite_score']:.1f}/100")
print(f"Взвешенный потенциал роста: {forecasts[top_symbol]['weighted_upside']:.1f}%")
print(f"Целевая цена (6-12 мес): ${forecasts[top_symbol]['weighted_target']:.2f}")

# Сохранение данных
with open('/home/ubuntu/forecasts_ratings.pkl', 'wb') as f:
    pickle.dump({
        'forecasts': forecasts,
        'scores': scores,
        'rating_df': rating_df,
        'top_symbol': top_symbol,
        'fundamental_catalysts': fundamental_catalysts
    }, f)

print("\n✓ Прогнозы и рейтинги рассчитаны и сохранены.")
print("=" * 80)

