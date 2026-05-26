"""
МОДУЛЬ FEATURE ENGINEERING
Создание признаков для машинного обучения
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Класс для создания признаков"""
    
    def __init__(self, prices_df):
        """
        Args:
            prices_df: DataFrame с ценами OHLCV
        """
        self.prices_df = prices_df.copy()
        self.features_df = pd.DataFrame(index=prices_df.index)
        
    def create_price_features(self):
        """Ценовые признаки"""
        print("\n📊 Создание ценовых признаков...")
        
        # Основные ценовые данные
        self.features_df['Close'] = self.prices_df['EURUSD_Close']
        self.features_df['Open'] = self.prices_df['EURUSD_Open']
        self.features_df['High'] = self.prices_df['EURUSD_High']
        self.features_df['Low'] = self.prices_df['EURUSD_Low']
        
        # Доходности
        self.features_df['Return_1d'] = self.prices_df['EURUSD_Close'].pct_change() * 100
        self.features_df['Return_3d'] = self.prices_df['EURUSD_Close'].pct_change(3) * 100
        self.features_df['Return_5d'] = self.prices_df['EURUSD_Close'].pct_change(5) * 100
        
        # Диапазон High-Low
        self.features_df['HL_Range'] = (self.prices_df['EURUSD_High'] - 
                                        self.prices_df['EURUSD_Low'])
        self.features_df['HL_Range_Pct'] = (self.features_df['HL_Range'] / 
                                            self.prices_df['EURUSD_Close'] * 100)
        
        # Body и тени свечи
        self.features_df['Body'] = abs(self.prices_df['EURUSD_Close'] - 
                                       self.prices_df['EURUSD_Open'])
        self.features_df['Upper_Shadow'] = (self.prices_df['EURUSD_High'] - 
                                           self.prices_df[['EURUSD_Close', 'EURUSD_Open']].max(axis=1))
        self.features_df['Lower_Shadow'] = (self.prices_df[['EURUSD_Close', 'EURUSD_Open']].min(axis=1) - 
                                           self.prices_df['EURUSD_Low'])
        
        print(f"  ✅ Создано {9} ценовых признаков")
        
    def create_technical_indicators(self):
        """Технические индикаторы"""
        print("\n📈 Создание технических индикаторов...")
        
        close = self.prices_df['EURUSD_Close']
        high = self.prices_df['EURUSD_High']
        low = self.prices_df['EURUSD_Low']
        
        # RSI (14, 21)
        self.features_df['RSI_14'] = RSIIndicator(close, window=14).rsi()
        self.features_df['RSI_21'] = RSIIndicator(close, window=21).rsi()
        
        # MACD
        macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
        self.features_df['MACD'] = macd.macd()
        self.features_df['MACD_Signal'] = macd.macd_signal()
        self.features_df['MACD_Diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(close, window=20, window_dev=2)
        self.features_df['BB_High'] = bb.bollinger_hband()
        self.features_df['BB_Low'] = bb.bollinger_lband()
        self.features_df['BB_Mid'] = bb.bollinger_mavg()
        self.features_df['BB_Width'] = ((bb.bollinger_hband() - bb.bollinger_lband()) / 
                                        bb.bollinger_mavg() * 100)
        self.features_df['BB_Position'] = ((close - bb.bollinger_lband()) / 
                                          (bb.bollinger_hband() - bb.bollinger_lband()) * 100)
        
        # ATR (Average True Range) - волатильность
        atr = AverageTrueRange(high, low, close, window=14)
        self.features_df['ATR'] = atr.average_true_range()
        self.features_df['ATR_Pct'] = (self.features_df['ATR'] / close * 100)
        
        # ADX (Average Directional Index) - сила тренда
        adx = ADXIndicator(high, low, close, window=14)
        self.features_df['ADX'] = adx.adx()
        self.features_df['ADX_Pos'] = adx.adx_pos()
        self.features_df['ADX_Neg'] = adx.adx_neg()
        
        # Stochastic
        stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
        self.features_df['Stoch_K'] = stoch.stoch()
        self.features_df['Stoch_D'] = stoch.stoch_signal()
        
        print(f"  ✅ Создано {17} технических индикаторов")
        
    def create_moving_averages(self):
        """Скользящие средние"""
        print("\n📊 Создание скользящих средних...")
        
        close = self.prices_df['EURUSD_Close']
        
        # SMA (Simple Moving Average)
        for period in [10, 20, 50, 100, 200]:
            sma = SMAIndicator(close, window=period)
            self.features_df[f'SMA_{period}'] = sma.sma_indicator()
            # Расстояние до SMA
            self.features_df[f'Distance_SMA_{period}'] = ((close - self.features_df[f'SMA_{period}']) / 
                                                          self.features_df[f'SMA_{period}'] * 100)
        
        # EMA (Exponential Moving Average)
        for period in [12, 26, 50]:
            ema = EMAIndicator(close, window=period)
            self.features_df[f'EMA_{period}'] = ema.ema_indicator()
        
        # Кроссоверы MA
        self.features_df['SMA_Cross_50_200'] = (self.features_df['SMA_50'] > 
                                                self.features_df['SMA_200']).astype(int)
        
        print(f"  ✅ Создано {14} признаков скользящих средних")
        
    def create_momentum_features(self):
        """Моментум признаки"""
        print("\n⚡ Создание моментум признаков...")
        
        close = self.prices_df['EURUSD_Close']
        
        # Momentum (разница цен)
        for period in [5, 10, 20]:
            self.features_df[f'Momentum_{period}'] = close - close.shift(period)
            self.features_df[f'Momentum_Pct_{period}'] = ((close - close.shift(period)) / 
                                                          close.shift(period) * 100)
        
        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            self.features_df[f'ROC_{period}'] = ((close - close.shift(period)) / 
                                                 close.shift(period) * 100)
        
        print(f"  ✅ Создано {9} моментум признаков")
        
    def create_lagged_features(self):
        """Лаговые признаки (прошлые значения)"""
        print("\n⏱️  Создание лаговых признаков...")
        
        # Лаги цены закрытия
        for lag in [1, 2, 3, 5, 10]:
            self.features_df[f'Close_Lag_{lag}'] = self.prices_df['EURUSD_Close'].shift(lag)
        
        # Лаги доходности
        for lag in [1, 2, 3, 5]:
            self.features_df[f'Return_Lag_{lag}'] = self.features_df['Return_1d'].shift(lag)
        
        print(f"  ✅ Создано {9} лаговых признаков")
        
    def create_correlation_features(self):
        """Признаки на основе коррелирующих инструментов"""
        print("\n🔗 Создание признаков корреляций...")
        
        # Доходности коррелирующих инструментов
        for instrument in ['DXY', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'XAUUSD']:
            col_name = f'{instrument}_Close'
            if col_name in self.prices_df.columns:
                # Доходность инструмента
                self.features_df[f'{instrument}_Return'] = (self.prices_df[col_name].pct_change() * 100)
                
                # Лаг 1 день (для ведущих индикаторов)
                self.features_df[f'{instrument}_Return_Lag1'] = (self.features_df[f'{instrument}_Return'].shift(1))
        
        # Спред между коррелирующими парами
        if 'GBPUSD_Close' in self.prices_df.columns:
            self.features_df['Spread_EUR_GBP'] = (self.prices_df['EURUSD_Close'] - 
                                                  self.prices_df['GBPUSD_Close'])
        
        print(f"  ✅ Создано {13} признаков корреляций")
        
    def create_calendar_features(self):
        """Календарные признаки"""
        print("\n📅 Создание календарных признаков...")
        
        # День недели (0=Monday, 4=Friday)
        self.features_df['DayOfWeek'] = self.prices_df.index.dayofweek
        
        # Месяц
        self.features_df['Month'] = self.prices_df.index.month
        
        # Квартал
        self.features_df['Quarter'] = self.prices_df.index.quarter
        
        # День месяца
        self.features_df['DayOfMonth'] = self.prices_df.index.day
        
        # Начало/конец месяца
        self.features_df['IsMonthStart'] = self.prices_df.index.is_month_start.astype(int)
        self.features_df['IsMonthEnd'] = self.prices_df.index.is_month_end.astype(int)
        
        # Циклические признаки (sin/cos для периодичности)
        self.features_df['DayOfWeek_Sin'] = np.sin(2 * np.pi * self.features_df['DayOfWeek'] / 7)
        self.features_df['DayOfWeek_Cos'] = np.cos(2 * np.pi * self.features_df['DayOfWeek'] / 7)
        self.features_df['Month_Sin'] = np.sin(2 * np.pi * self.features_df['Month'] / 12)
        self.features_df['Month_Cos'] = np.cos(2 * np.pi * self.features_df['Month'] / 12)
        
        print(f"  ✅ Создано {10} календарных признаков")
        
    def create_target_variables(self):
        """Целевые переменные для ML"""
        print("\n🎯 Создание целевых переменных...")
        
        # 1. Классификация: направление движения на следующий день
        self.features_df['Target_Direction'] = (self.prices_df['EURUSD_Close'].shift(-1) > 
                                                self.prices_df['EURUSD_Close']).astype(int)
        
        # 2. Регрессия: доходность следующего дня
        self.features_df['Target_Return'] = ((self.prices_df['EURUSD_Close'].shift(-1) - 
                                              self.prices_df['EURUSD_Close']) / 
                                             self.prices_df['EURUSD_Close'] * 100)
        
        # 3. Регрессия: цена следующего дня
        self.features_df['Target_Price'] = self.prices_df['EURUSD_Close'].shift(-1)
        
        print(f"  ✅ Создано 3 целевые переменные")
        
    def create_all_features(self):
        """Создание всех признаков"""
        print("\n" + "="*70)
        print("🏗️  СОЗДАНИЕ ВСЕХ ПРИЗНАКОВ")
        print("="*70)
        
        self.create_price_features()
        self.create_technical_indicators()
        self.create_moving_averages()
        self.create_momentum_features()
        self.create_lagged_features()
        self.create_correlation_features()
        self.create_calendar_features()
        self.create_target_variables()
        
        # Удаляем строки с NaN
        initial_rows = len(self.features_df)
        self.features_df = self.features_df.dropna()
        final_rows = len(self.features_df)
        
        print("\n" + "="*70)
        print(f"✅ FEATURE ENGINEERING ЗАВЕРШЕН")
        print(f"📊 Всего признаков: {len(self.features_df.columns)}")
        print(f"📅 Строк данных: {final_rows} (удалено {initial_rows - final_rows} с NaN)")
        print("="*70)
        
        return self.features_df
    
    def save_features(self, output_path='/home/ubuntu/eurusd_analysis/data/features.csv'):
        """Сохранение признаков"""
        self.features_df.to_csv(output_path)
        print(f"\n💾 Признаки сохранены: {output_path}")
        
    def get_feature_groups(self):
        """Получение групп признаков для анализа"""
        groups = {
            'price': [col for col in self.features_df.columns if 'Close' in col or 'Open' in col or 
                     'High' in col or 'Low' in col or 'Body' in col or 'Shadow' in col or 
                     'HL_Range' in col],
            'technical': [col for col in self.features_df.columns if any(ind in col for ind in 
                         ['RSI', 'MACD', 'BB', 'ATR', 'ADX', 'Stoch'])],
            'moving_avg': [col for col in self.features_df.columns if 'SMA' in col or 'EMA' in col],
            'momentum': [col for col in self.features_df.columns if 'Momentum' in col or 'ROC' in col],
            'lagged': [col for col in self.features_df.columns if 'Lag' in col],
            'correlation': [col for col in self.features_df.columns if any(inst in col for inst in 
                           ['DXY', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'XAUUSD', 'Spread'])],
            'calendar': [col for col in self.features_df.columns if any(cal in col for cal in 
                        ['Day', 'Month', 'Quarter', 'Sin', 'Cos'])],
            'target': ['Target_Direction', 'Target_Return', 'Target_Price']
        }
        
        return groups


def main():
    """Главная функция"""
    # Загрузка данных
    prices_df = pd.read_csv('/home/ubuntu/eurusd_analysis/data/master_data.csv',
                           index_col=0, parse_dates=True)
    
    print(f"\n📊 Загружено {len(prices_df)} дней данных")
    
    # Создание признаков
    engineer = FeatureEngineer(prices_df)
    features_df = engineer.create_all_features()
    
    # Сохранение
    engineer.save_features()
    
    # Анализ групп признаков
    groups = engineer.get_feature_groups()
    print("\n📋 ГРУППЫ ПРИЗНАКОВ:")
    print("-" * 70)
    for group_name, features in groups.items():
        print(f"  {group_name:15}: {len(features):3} признаков")
    
    # Статистика по признакам
    print("\n📊 СТАТИСТИКА ПО ПРИЗНАКАМ:")
    print("-" * 70)
    print(features_df.describe().T[['mean', 'std', 'min', 'max']].head(20))
    
    return features_df


if __name__ == "__main__":
    features_df = main()
