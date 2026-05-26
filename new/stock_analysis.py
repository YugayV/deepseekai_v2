import json
import pandas as pd
from datetime import datetime, timedelta

# Загрузка данных
with open('/home/ubuntu/.external_service_outputs/fetch_market_data_output_1779527489.json', 'r') as f:
    data = json.load(f)

# Создание DataFrame
df = pd.DataFrame(data['data'])
df['ts_event'] = pd.to_datetime(df['ts_event'])
df = df.sort_values(['symbol', 'ts_event'])

# Текущая дата
current_date = pd.to_datetime('2026-05-22').tz_localize('UTC')

# Расчет метрик для каждой акции
results = {}

for symbol in df['symbol'].unique():
    symbol_df = df[df['symbol'] == symbol].copy()
    
    # Текущая цена (последняя)
    current_price = symbol_df.iloc[-1]['close']
    
    # 52-week high/low
    high_52w = symbol_df['high'].max()
    low_52w = symbol_df['low'].min()
    
    # Расстояние до 52-week high/low
    dist_to_high = ((current_price - high_52w) / high_52w) * 100
    dist_to_low = ((current_price - low_52w) / low_52w) * 100
    
    # Доходность за периоды
    def get_return(days):
        target_date = current_date - timedelta(days=days)
        historical = symbol_df[symbol_df['ts_event'] <= target_date]
        if len(historical) > 0:
            old_price = historical.iloc[-1]['close']
            return ((current_price - old_price) / old_price) * 100
        return None
    
    ret_1m = get_return(30)
    ret_3m = get_return(90)
    ret_6m = get_return(180)
    ret_12m = get_return(365)
    
    # Волатильность (30D и 90D)
    symbol_df['returns'] = symbol_df['close'].pct_change()
    vol_30d = symbol_df.tail(30)['returns'].std() * (252 ** 0.5) * 100  # Annualized
    vol_90d = symbol_df.tail(90)['returns'].std() * (252 ** 0.5) * 100  # Annualized
    
    results[symbol] = {
        'current_price': round(current_price, 2),
        'high_52w': round(high_52w, 2),
        'low_52w': round(low_52w, 2),
        'dist_to_high_pct': round(dist_to_high, 2),
        'dist_to_low_pct': round(dist_to_low, 2),
        'return_1m_pct': round(ret_1m, 2) if ret_1m else None,
        'return_3m_pct': round(ret_3m, 2) if ret_3m else None,
        'return_6m_pct': round(ret_6m, 2) if ret_6m else None,
        'return_12m_pct': round(ret_12m, 2) if ret_12m else None,
        'volatility_30d_pct': round(vol_30d, 2),
        'volatility_90d_pct': round(vol_90d, 2),
        'volume_latest': int(symbol_df.iloc[-1]['volume'])
    }

# Вывод результатов
print(json.dumps(results, indent=2))
