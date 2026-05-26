"""
МОДУЛЬ ЗАГРУЗКИ ДАННЫХ
Загрузка EUR/USD и коррелирующих инструментов за 2 года (май 2024 - май 2026)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Класс для загрузки исторических данных"""
    
    def __init__(self, start_date='2024-05-01', end_date='2026-05-23'):
        """
        Инициализация загрузчика данных
        
        Args:
            start_date: Начальная дата (по умолчанию май 2024)
            end_date: Конечная дата (по умолчанию май 2026)
        """
        self.start_date = start_date
        self.end_date = end_date
        
        # Символы инструментов для загрузки
        self.symbols = {
            'EURUSD': 'EURUSD=X',      # EUR/USD
            'DXY': 'DX-Y.NYB',         # Индекс доллара
            'GBPUSD': 'GBPUSD=X',      # GBP/USD
            'USDJPY': 'USDJPY=X',      # USD/JPY
            'USDCHF': 'USDCHF=X',      # USD/CHF
            'AUDUSD': 'AUDUSD=X',      # AUD/USD
            'XAUUSD': 'GC=F',          # Золото (фьючерс)
            'WTI': 'CL=F',             # Нефть WTI
            'SPX': '^GSPC',            # S&P 500
            'DAX': '^GDAXI',           # DAX
            'VIX': '^VIX'              # Индекс волатильности
        }
        
        self.data = {}
        
    def load_all_data(self):
        """Загрузка всех инструментов"""
        print(f"🔄 Загрузка данных с {self.start_date} по {self.end_date}...")
        
        for name, symbol in self.symbols.items():
            try:
                print(f"  📊 Загрузка {name} ({symbol})...")
                df = yf.download(symbol, start=self.start_date, end=self.end_date, 
                               progress=False, auto_adjust=False)
                
                if not df.empty:
                    # Убираем многоуровневый индекс если есть
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)
                    
                    # Сохраняем OHLCV данные
                    self.data[name] = pd.DataFrame({
                        'Open': df['Open'],
                        'High': df['High'],
                        'Low': df['Low'],
                        'Close': df['Close'],
                        'Volume': df['Volume'] if 'Volume' in df.columns else 0
                    })
                    print(f"    ✅ {name}: {len(self.data[name])} баров")
                else:
                    print(f"    ⚠️ {name}: данные не получены")
                    
            except Exception as e:
                print(f"    ❌ Ошибка загрузки {name}: {e}")
                
        return self.data
    
    def create_master_dataframe(self):
        """
        Создание единого DataFrame со всеми инструментами
        Выравнивание по датам и заполнение пропусков
        """
        print("\n🔧 Создание мастер-таблицы...")
        
        if not self.data:
            print("❌ Нет загруженных данных")
            return None
            
        # Создаем базовый DataFrame на основе EUR/USD
        if 'EURUSD' not in self.data or self.data['EURUSD'].empty:
            print("❌ EUR/USD не загружен")
            return None
            
        master_df = pd.DataFrame(index=self.data['EURUSD'].index)
        
        # Добавляем данные по каждому инструменту
        for name, df in self.data.items():
            if not df.empty:
                # Добавляем Close цену
                master_df[f'{name}_Close'] = df['Close']
                # Добавляем OHLC для основной пары
                if name == 'EURUSD':
                    master_df['EURUSD_Open'] = df['Open']
                    master_df['EURUSD_High'] = df['High']
                    master_df['EURUSD_Low'] = df['Low']
                    master_df['EURUSD_Volume'] = df['Volume']
        
        # Заполнение пропусков методом forward fill
        master_df = master_df.ffill().bfill()
        
        print(f"✅ Мастер-таблица создана: {len(master_df)} строк, {len(master_df.columns)} столбцов")
        print(f"📅 Период: {master_df.index[0].date()} - {master_df.index[-1].date()}")
        
        return master_df
    
    def calculate_returns(self, master_df):
        """
        Расчет доходностей для корреляционного анализа
        
        Args:
            master_df: Мастер-таблица с ценами
            
        Returns:
            DataFrame с доходностями
        """
        returns_df = pd.DataFrame(index=master_df.index)
        
        # Рассчитываем дневные доходности для каждого инструмента
        for col in master_df.columns:
            if '_Close' in col:
                returns_df[col.replace('_Close', '_Return')] = master_df[col].pct_change() * 100
        
        # Удаляем первую строку с NaN
        returns_df = returns_df.dropna()
        
        print(f"✅ Доходности рассчитаны: {len(returns_df)} строк")
        
        return returns_df
    
    def save_data(self, master_df, returns_df, output_dir='data'):
        """Сохранение данных в CSV"""
        import os
        
        output_path = f"/home/ubuntu/eurusd_analysis/{output_dir}"
        os.makedirs(output_path, exist_ok=True)
        
        # Сохраняем мастер-таблицу
        master_df.to_csv(f'{output_path}/master_data.csv')
        print(f"💾 Сохранено: {output_path}/master_data.csv")
        
        # Сохраняем доходности
        returns_df.to_csv(f'{output_path}/returns_data.csv')
        print(f"💾 Сохранено: {output_path}/returns_data.csv")
        
        # Сохраняем отдельные файлы для каждого инструмента
        for name, df in self.data.items():
            if not df.empty:
                df.to_csv(f'{output_path}/{name}_raw.csv')
        
        print(f"✅ Все данные сохранены в {output_path}/")


def main():
    """Главная функция для тестирования загрузки данных"""
    loader = DataLoader(start_date='2024-05-01', end_date='2026-05-23')
    
    # Загрузка всех данных
    data = loader.load_all_data()
    
    if data:
        # Создание мастер-таблицы
        master_df = loader.create_master_dataframe()
        
        if master_df is not None:
            # Расчет доходностей
            returns_df = loader.calculate_returns(master_df)
            
            # Сохранение
            loader.save_data(master_df, returns_df)
            
            print("\n" + "="*60)
            print("📊 СТАТИСТИКА ЗАГРУЖЕННЫХ ДАННЫХ")
            print("="*60)
            print(f"\nФорма мастер-таблицы: {master_df.shape}")
            print(f"\nПервые 3 строки:")
            print(master_df.head(3))
            print(f"\nПоследние 3 строки:")
            print(master_df.tail(3))
            print(f"\nОписательная статистика EUR/USD:")
            print(master_df[['EURUSD_Open', 'EURUSD_High', 'EURUSD_Low', 'EURUSD_Close']].describe())
            
            return master_df, returns_df
    
    return None, None


if __name__ == "__main__":
    master_df, returns_df = main()
