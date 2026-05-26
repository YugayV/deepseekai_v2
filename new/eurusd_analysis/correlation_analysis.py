"""
МОДУЛЬ КОРРЕЛЯЦИОННОГО АНАЛИЗА
Анализ связей EUR/USD с ключевыми инструментами
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Настройки для matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

class CorrelationAnalyzer:
    """Класс для корреляционного анализа"""
    
    def __init__(self, returns_df, prices_df):
        """
        Args:
            returns_df: DataFrame с доходностями
            prices_df: DataFrame с ценами
        """
        self.returns_df = returns_df
        self.prices_df = prices_df
        self.results = {}
        
    def static_correlation(self):
        """Статическая корреляция за весь период"""
        print("\n" + "="*60)
        print("📊 СТАТИЧЕСКАЯ КОРРЕЛЯЦИЯ (весь период)")
        print("="*60)
        
        # Выбираем только столбцы с доходностями
        corr_matrix = self.returns_df.corr()
        
        # Фокус на EUR/USD
        eurusd_corr = corr_matrix['EURUSD_Return'].sort_values(ascending=False)
        
        print("\n🎯 Корреляция с EUR/USD (отсортировано):")
        print("-" * 60)
        for idx, value in eurusd_corr.items():
            if idx != 'EURUSD_Return':
                instrument = idx.replace('_Return', '')
                # Оценка силы корреляции
                if abs(value) >= 0.7:
                    strength = "ОЧЕНЬ СИЛЬНАЯ"
                elif abs(value) >= 0.5:
                    strength = "СИЛЬНАЯ"
                elif abs(value) >= 0.3:
                    strength = "УМЕРЕННАЯ"
                else:
                    strength = "СЛАБАЯ"
                
                direction = "прямая" if value > 0 else "обратная"
                print(f"{instrument:12} : {value:+.4f}  [{strength:15} {direction}]")
        
        self.results['static_correlation'] = eurusd_corr
        return corr_matrix, eurusd_corr
    
    def rolling_correlation(self, windows=[30, 60, 90]):
        """
        Скользящая корреляция для разных окон
        
        Args:
            windows: Список окон для расчета (дни)
        """
        print("\n" + "="*60)
        print("📈 СКОЛЬЗЯЩАЯ КОРРЕЛЯЦИЯ")
        print("="*60)
        
        rolling_corrs = {}
        
        for window in windows:
            print(f"\n🔍 Окно {window} дней:")
            
            # Рассчитываем скользящую корреляцию для каждого инструмента
            temp_corrs = {}
            for col in self.returns_df.columns:
                if col != 'EURUSD_Return' and '_Return' in col:
                    rolling_corr = self.returns_df['EURUSD_Return'].rolling(
                        window=window
                    ).corr(self.returns_df[col])
                    temp_corrs[col] = rolling_corr
                    
                    # Статистика по скользящей корреляции
                    instrument = col.replace('_Return', '')
                    mean_corr = rolling_corr.mean()
                    std_corr = rolling_corr.std()
                    print(f"  {instrument:12}: mean={mean_corr:+.4f}, std={std_corr:.4f}")
            
            rolling_corrs[f'window_{window}'] = pd.DataFrame(temp_corrs)
        
        self.results['rolling_correlation'] = rolling_corrs
        return rolling_corrs
    
    def lagged_correlation(self, max_lag=5):
        """
        Лаговая корреляция (с задержкой 1-5 дней)
        Определяет ведущие индикаторы
        
        Args:
            max_lag: Максимальная задержка в днях
        """
        print("\n" + "="*60)
        print("⏱️  ЛАГОВАЯ КОРРЕЛЯЦИЯ (поиск ведущих индикаторов)")
        print("="*60)
        
        lagged_results = {}
        
        instruments = [col for col in self.returns_df.columns 
                      if col != 'EURUSD_Return' and '_Return' in col]
        
        for instrument in instruments:
            name = instrument.replace('_Return', '')
            lag_corrs = []
            
            # Корреляция без лага (текущий день)
            corr_0 = self.returns_df['EURUSD_Return'].corr(
                self.returns_df[instrument]
            )
            lag_corrs.append(corr_0)
            
            # Корреляция с лагами 1-5 дней
            for lag in range(1, max_lag + 1):
                # Инструмент с задержкой -> EUR/USD сегодня
                corr_lag = self.returns_df['EURUSD_Return'].corr(
                    self.returns_df[instrument].shift(lag)
                )
                lag_corrs.append(corr_lag)
            
            lagged_results[name] = lag_corrs
            
            # Находим максимальную корреляцию и соответствующий лаг
            max_corr_idx = np.argmax(np.abs(lag_corrs))
            max_corr = lag_corrs[max_corr_idx]
            
            if max_corr_idx == 0:
                status = "синхронный"
            else:
                status = f"ведущий ({max_corr_idx} дн.)"
            
            print(f"\n{name:12}:")
            print(f"  Лаги 0-{max_lag}: {[f'{c:+.3f}' for c in lag_corrs]}")
            print(f"  🎯 Оптимальный лаг: {max_corr_idx} дней | Корреляция: {max_corr:+.4f} | Статус: {status}")
        
        # Создаем DataFrame с результатами
        lag_df = pd.DataFrame(lagged_results, 
                             index=[f'Lag_{i}' for i in range(max_lag + 1)])
        
        self.results['lagged_correlation'] = lag_df
        return lag_df
    
    def correlation_stability(self):
        """
        Анализ стабильности корреляций во времени
        Разбивает период на кварталы и сравнивает
        """
        print("\n" + "="*60)
        print("🔬 СТАБИЛЬНОСТЬ КОРРЕЛЯЦИЙ ПО ПЕРИОДАМ")
        print("="*60)
        
        # Разбиваем данные на кварталы
        quarters = self.returns_df.resample('Q')
        
        quarterly_corrs = []
        quarter_names = []
        
        for quarter_end, quarter_data in quarters:
            if len(quarter_data) >= 30:  # Минимум 30 наблюдений
                corr_vector = quarter_data.corr()['EURUSD_Return'].drop('EURUSD_Return')
                quarterly_corrs.append(corr_vector)
                quarter_names.append(quarter_end.strftime('%Y-Q%q'))
        
        # Создаем DataFrame
        quarterly_df = pd.DataFrame(quarterly_corrs, index=quarter_names)
        
        print("\n📊 Корреляции по кварталам:")
        print(quarterly_df.T.round(3))
        
        # Анализ стабильности
        print("\n📉 Стандартное отклонение корреляций (стабильность):")
        stability = quarterly_df.std().sort_values()
        
        for instrument, std_val in stability.items():
            name = instrument.replace('_Return', '')
            if std_val < 0.1:
                status = "СТАБИЛЬНАЯ"
            elif std_val < 0.2:
                status = "УМЕРЕННАЯ"
            else:
                status = "НЕСТАБИЛЬНАЯ"
            print(f"  {name:12}: {std_val:.4f}  [{status}]")
        
        self.results['correlation_stability'] = {
            'quarterly': quarterly_df,
            'stability': stability
        }
        
        return quarterly_df, stability
    
    def save_results(self, output_dir='/home/ubuntu/eurusd_analysis/results'):
        """Сохранение результатов в CSV"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Статическая корреляция
        if 'static_correlation' in self.results:
            self.results['static_correlation'].to_csv(
                f'{output_dir}/correlation_static.csv'
            )
            print(f"\n💾 Сохранено: correlation_static.csv")
        
        # Лаговая корреляция
        if 'lagged_correlation' in self.results:
            self.results['lagged_correlation'].to_csv(
                f'{output_dir}/correlation_lagged.csv'
            )
            print(f"💾 Сохранено: correlation_lagged.csv")
        
        # Стабильность
        if 'correlation_stability' in self.results:
            self.results['correlation_stability']['quarterly'].to_csv(
                f'{output_dir}/correlation_quarterly.csv'
            )
            print(f"💾 Сохранено: correlation_quarterly.csv")
        
        print(f"\n✅ Результаты корреляционного анализа сохранены в {output_dir}/")


def main():
    """Главная функция для запуска анализа"""
    print("\n" + "="*70)
    print("🎯 КОРРЕЛЯЦИОННЫЙ АНАЛИЗ EUR/USD")
    print("="*70)
    
    # Загрузка данных
    returns_df = pd.read_csv('/home/ubuntu/eurusd_analysis/data/returns_data.csv', 
                            index_col=0, parse_dates=True)
    prices_df = pd.read_csv('/home/ubuntu/eurusd_analysis/data/master_data.csv',
                           index_col=0, parse_dates=True)
    
    print(f"\n📊 Загружено данных: {len(returns_df)} дней")
    
    # Создаем анализатор
    analyzer = CorrelationAnalyzer(returns_df, prices_df)
    
    # 1. Статическая корреляция
    corr_matrix, eurusd_corr = analyzer.static_correlation()
    
    # 2. Скользящая корреляция
    rolling_corrs = analyzer.rolling_correlation(windows=[30, 60, 90])
    
    # 3. Лаговая корреляция
    lag_df = analyzer.lagged_correlation(max_lag=5)
    
    # 4. Стабильность корреляций
    quarterly_df, stability = analyzer.correlation_stability()
    
    # Сохранение результатов
    analyzer.save_results()
    
    print("\n" + "="*70)
    print("✅ КОРРЕЛЯЦИОННЫЙ АНАЛИЗ ЗАВЕРШЕН")
    print("="*70)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
