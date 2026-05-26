"""
Модуль загрузки и обработки данных для торговой системы GLD
Автор: Trading System
Дата: 2026-05-23

Функционал:
- Загрузка исторических данных OHLCV
- Создание множественных таймфреймов (1h, 4h, 1d)
- Фильтрация по торговым сессиям
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz


class DataLoader:
    """Класс для загрузки и обработки рыночных данных"""
    
    def __init__(self, hourly_file, daily_file):
        """
        Инициализация загрузчика данных
        
        Args:
            hourly_file: путь к JSON файлу с часовыми данными
            daily_file: путь к JSON файлу с дневными данными
        """
        self.hourly_file = hourly_file
        self.daily_file = daily_file
        self.df_1h = None
        self.df_4h = None
        self.df_1d = None
        
    def load_data(self):
        """Загрузить все данные и создать таймфреймы"""
        print("Загрузка данных...")
        
        # Загрузка часовых данных
        with open(self.hourly_file, 'r') as f:
            hourly_data = json.load(f)
        
        # Загрузка дневных данных
        with open(self.daily_file, 'r') as f:
            daily_data = json.load(f)
        
        # Создание DataFrame для 1h
        df_1h_raw = pd.DataFrame(hourly_data['data'])
        df_1h_raw['ts_event'] = pd.to_datetime(df_1h_raw['ts_event'])
        df_1h_raw.set_index('ts_event', inplace=True)
        
        # Фильтрация по regular hours (09:30-16:00 ET = 13:30-20:00 UTC)
        # С учетом летнего времени EDT (09:30-16:00 EDT = 13:30-20:00 UTC)
        # и зимнего времени EST (09:30-16:00 EST = 14:30-21:00 UTC)
        self.df_1h = self._filter_regular_hours(df_1h_raw)
        
        print(f"Загружено {len(self.df_1h)} часовых баров (regular hours)")
        
        # Создание 4h через resample
        self.df_4h = self._resample_to_4h(self.df_1h)
        print(f"Создано {len(self.df_4h)} 4-часовых баров")
        
        # Создание DataFrame для 1d
        df_1d_raw = pd.DataFrame(daily_data['data'])
        df_1d_raw['ts_event'] = pd.to_datetime(df_1d_raw['ts_event'])
        df_1d_raw.set_index('ts_event', inplace=True)
        self.df_1d = df_1d_raw[['open', 'high', 'low', 'close', 'volume']].copy()
        
        print(f"Загружено {len(self.df_1d)} дневных баров")
        
        # Расчет дополнительных полезных полей
        for df in [self.df_1h, self.df_4h, self.df_1d]:
            df['hl_range'] = df['high'] - df['low']
            df['oc_change'] = df['close'] - df['open']
            df['oc_pct'] = (df['close'] - df['open']) / df['open'] * 100
        
        return self.df_1h, self.df_4h, self.df_1d
    
    def _filter_regular_hours(self, df):
        """
        Фильтровать данные по regular trading hours (09:30-16:00 ET)
        
        Args:
            df: DataFrame с часовыми данными
            
        Returns:
            DataFrame с данными только в торговые часы
        """
        # Конвертируем в ET (Eastern Time)
        et = pytz.timezone('US/Eastern')
        df_et = df.copy()
        df_et.index = df_et.index.tz_convert(et)
        
        # Фильтруем по времени: 09:30 - 16:00 ET
        # Включаем бары с 09:00 (начало часа, который включает 09:30)
        # до 15:00 (последний час торговой сессии)
        regular_hours = df_et.between_time('09:00', '15:59')
        
        # Возвращаем с UTC индексом
        regular_hours.index = regular_hours.index.tz_convert('UTC')
        
        return regular_hours[['open', 'high', 'low', 'close', 'volume']].copy()
    
    def _resample_to_4h(self, df_1h):
        """
        Создать 4-часовой таймфрейм из часовых данных
        
        Args:
            df_1h: DataFrame с часовыми данными
            
        Returns:
            DataFrame с 4-часовыми барами
        """
        # Resample с агрегацией
        df_4h = df_1h.resample('4H', label='right', closed='right').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return df_4h
    
    def get_data_summary(self):
        """Получить сводку по загруженным данным"""
        summary = {
            '1h': {
                'bars': len(self.df_1h),
                'start': self.df_1h.index[0],
                'end': self.df_1h.index[-1],
                'price_range': f"${self.df_1h['low'].min():.2f} - ${self.df_1h['high'].max():.2f}"
            },
            '4h': {
                'bars': len(self.df_4h),
                'start': self.df_4h.index[0],
                'end': self.df_4h.index[-1],
                'price_range': f"${self.df_4h['low'].min():.2f} - ${self.df_4h['high'].max():.2f}"
            },
            '1d': {
                'bars': len(self.df_1d),
                'start': self.df_1d.index[0],
                'end': self.df_1d.index[-1],
                'price_range': f"${self.df_1d['low'].min():.2f} - ${self.df_1d['high'].max():.2f}"
            }
        }
        return summary


if __name__ == "__main__":
    # Тестовый запуск
    loader = DataLoader(
        hourly_file='/home/ubuntu/.external_service_outputs/fetch_market_data_output_1779529162.json',
        daily_file='/home/ubuntu/.external_service_outputs/fetch_market_data_output_1779529148.json'
    )
    
    df_1h, df_4h, df_1d = loader.load_data()
    
    print("\n=== Сводка по данным ===")
    summary = loader.get_data_summary()
    for tf, info in summary.items():
        print(f"\nТаймфрейм {tf}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
