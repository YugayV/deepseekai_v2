import json
import pandas as pd
from datetime import datetime, timedelta

# Загрузка данных
with open('/home/ubuntu/.external_service_outputs/fetch_crypto_data_output_1779527524.json', 'r') as f:
    data = json.load(f)

# Создание DataFrame для цен
prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms')

# Текущая цена
current_price = 74557.64

# 52-week high/low
high_52w = prices_df['price'].max()
low_52w = prices_df['price'].min()

# Расстояние до 52-week high/low
dist_to_high = ((current_price - high_52w) / high_52w) * 100
dist_to_low = ((current_price - low_52w) / low_52w) * 100

# Цены для расчета доходности
current_date = pd.to_datetime('2026-05-22')

def get_price_on_date(target_date):
    """Получить ближайшую цену к целевой дате"""
    closest = prices_df.iloc[(prices_df['timestamp'] - target_date).abs().argsort()[:1]]
    return closest['price'].values[0]

# Доходность за периоды
price_1m_ago = get_price_on_date(current_date - timedelta(days=30))
price_3m_ago = get_price_on_date(current_date - timedelta(days=90))
price_6m_ago = get_price_on_date(current_date - timedelta(days=180))
price_12m_ago = get_price_on_date(current_date - timedelta(days=365))

ret_1m = ((current_price - price_1m_ago) / price_1m_ago) * 100
ret_3m = ((current_price - price_3m_ago) / price_3m_ago) * 100
ret_6m = ((current_price - price_6m_ago) / price_6m_ago) * 100
ret_12m = ((current_price - price_12m_ago) / price_12m_ago) * 100

# Волатильность (30D и 90D)
prices_df['returns'] = prices_df['price'].pct_change()
vol_30d = prices_df.tail(30)['returns'].std() * (365 ** 0.5) * 100  # Annualized
vol_90d = prices_df.tail(90)['returns'].std() * (365 ** 0.5) * 100  # Annualized

results = {
    'symbol': 'BTC',
    'current_price': round(current_price, 2),
    'high_52w': round(high_52w, 2),
    'low_52w': round(low_52w, 2),
    'dist_to_high_pct': round(dist_to_high, 2),
    'dist_to_low_pct': round(dist_to_low, 2),
    'return_1m_pct': round(ret_1m, 2),
    'return_3m_pct': round(ret_3m, 2),
    'return_6m_pct': round(ret_6m, 2),
    'return_12m_pct': round(ret_12m, 2),
    'volatility_30d_pct': round(vol_30d, 2),
    'volatility_90d_pct': round(vol_90d, 2)
}

# Вывод результатов
print(json.dumps(results, indent=2))
