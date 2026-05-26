"""
МОДУЛЬ РАСПОЗНАВАНИЯ ПАТТЕРНОВ
Алгоритмическая детекция технических фигур и свечных паттернов
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

class PatternDetector:
    """Класс для распознавания технических паттернов"""
    
    def __init__(self, prices_df):
        """
        Args:
            prices_df: DataFrame с ценами OHLC
        """
        self.prices_df = prices_df.copy()
        self.patterns = {}
        
    def find_pivots(self, window=5):
        """Поиск локальных максимумов и минимумов (пивотов)"""
        highs = self.prices_df['EURUSD_High'].values
        lows = self.prices_df['EURUSD_Low'].values
        
        # Локальные максимумы
        high_pivots = argrelextrema(highs, np.greater, order=window)[0]
        
        # Локальные минимумы
        low_pivots = argrelextrema(lows, np.less, order=window)[0]
        
        return high_pivots, low_pivots
    
    def detect_double_top(self, window=10, tolerance=0.002):
        """
        Двойная вершина
        
        Args:
            window: окно для поиска пивотов
            tolerance: допустимое отклонение цены (0.2%)
        """
        high_pivots, _ = self.find_pivots(window)
        
        double_tops = []
        
        for i in range(len(high_pivots) - 1):
            for j in range(i + 1, len(high_pivots)):
                idx1, idx2 = high_pivots[i], high_pivots[j]
                
                # Проверяем расстояние между пивотами
                if idx2 - idx1 < 10 or idx2 - idx1 > 100:
                    continue
                
                high1 = self.prices_df.iloc[idx1]['EURUSD_High']
                high2 = self.prices_df.iloc[idx2]['EURUSD_High']
                
                # Проверяем похожесть вершин
                if abs(high1 - high2) / high1 <= tolerance:
                    # Проверяем наличие промежуточного минимума
                    between_lows = self.prices_df.iloc[idx1:idx2]['EURUSD_Low']
                    if between_lows.min() < min(high1, high2) * 0.99:
                        double_tops.append({
                            'start_idx': idx1,
                            'end_idx': idx2,
                            'start_date': self.prices_df.index[idx1],
                            'end_date': self.prices_df.index[idx2],
                            'high1': high1,
                            'high2': high2,
                            'neckline': between_lows.min(),
                            'pattern': 'Double Top'
                        })
        
        return double_tops
    
    def detect_double_bottom(self, window=10, tolerance=0.002):
        """Двойное дно"""
        _, low_pivots = self.find_pivots(window)
        
        double_bottoms = []
        
        for i in range(len(low_pivots) - 1):
            for j in range(i + 1, len(low_pivots)):
                idx1, idx2 = low_pivots[i], low_pivots[j]
                
                # Проверяем расстояние
                if idx2 - idx1 < 10 or idx2 - idx1 > 100:
                    continue
                
                low1 = self.prices_df.iloc[idx1]['EURUSD_Low']
                low2 = self.prices_df.iloc[idx2]['EURUSD_Low']
                
                # Проверяем похожесть днищ
                if abs(low1 - low2) / low1 <= tolerance:
                    # Проверяем наличие промежуточного максимума
                    between_highs = self.prices_df.iloc[idx1:idx2]['EURUSD_High']
                    if between_highs.max() > max(low1, low2) * 1.01:
                        double_bottoms.append({
                            'start_idx': idx1,
                            'end_idx': idx2,
                            'start_date': self.prices_df.index[idx1],
                            'end_date': self.prices_df.index[idx2],
                            'low1': low1,
                            'low2': low2,
                            'neckline': between_highs.max(),
                            'pattern': 'Double Bottom'
                        })
        
        return double_bottoms
    
    def detect_head_and_shoulders(self, window=10):
        """Голова и плечи"""
        high_pivots, _ = self.find_pivots(window)
        
        h_s_patterns = []
        
        for i in range(len(high_pivots) - 2):
            idx1, idx2, idx3 = high_pivots[i], high_pivots[i+1], high_pivots[i+2]
            
            # Проверяем расстояние
            if idx3 - idx1 < 30 or idx3 - idx1 > 150:
                continue
            
            h1 = self.prices_df.iloc[idx1]['EURUSD_High']
            h2 = self.prices_df.iloc[idx2]['EURUSD_High']  # Голова
            h3 = self.prices_df.iloc[idx3]['EURUSD_High']
            
            # Голова должна быть выше плеч
            if h2 > h1 * 1.005 and h2 > h3 * 1.005:
                # Плечи примерно на одном уровне
                if abs(h1 - h3) / h1 <= 0.01:
                    # Находим минимумы между пивотами (neckline)
                    neckline_left = self.prices_df.iloc[idx1:idx2]['EURUSD_Low'].min()
                    neckline_right = self.prices_df.iloc[idx2:idx3]['EURUSD_Low'].min()
                    neckline = (neckline_left + neckline_right) / 2
                    
                    h_s_patterns.append({
                        'start_idx': idx1,
                        'head_idx': idx2,
                        'end_idx': idx3,
                        'start_date': self.prices_df.index[idx1],
                        'head_date': self.prices_df.index[idx2],
                        'end_date': self.prices_df.index[idx3],
                        'left_shoulder': h1,
                        'head': h2,
                        'right_shoulder': h3,
                        'neckline': neckline,
                        'pattern': 'Head and Shoulders'
                    })
        
        return h_s_patterns
    
    def detect_candle_patterns(self):
        """Распознавание свечных паттернов"""
        df = self.prices_df.copy()
        
        # Расчет параметров свечей
        df['Body'] = abs(df['EURUSD_Close'] - df['EURUSD_Open'])
        df['Upper_Shadow'] = df['EURUSD_High'] - df[['EURUSD_Close', 'EURUSD_Open']].max(axis=1)
        df['Lower_Shadow'] = df[['EURUSD_Close', 'EURUSD_Open']].min(axis=1) - df['EURUSD_Low']
        df['Range'] = df['EURUSD_High'] - df['EURUSD_Low']
        df['Is_Bullish'] = (df['EURUSD_Close'] > df['EURUSD_Open']).astype(int)
        
        patterns = []
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # МОЛОТ (Hammer)
            if (current['Lower_Shadow'] > current['Body'] * 2 and
                current['Upper_Shadow'] < current['Body'] * 0.3 and
                current['Body'] > 0):
                patterns.append({
                    'idx': i,
                    'date': df.index[i],
                    'pattern': 'Hammer',
                    'signal': 'Bullish',
                    'close': current['EURUSD_Close']
                })
            
            # ПОВЕШЕННЫЙ (Hanging Man)
            if (current['Lower_Shadow'] > current['Body'] * 2 and
                current['Upper_Shadow'] < current['Body'] * 0.3 and
                current['Body'] > 0 and
                i > 0 and df.iloc[i-1]['EURUSD_Close'] > current['EURUSD_Close']):
                patterns.append({
                    'idx': i,
                    'date': df.index[i],
                    'pattern': 'Hanging Man',
                    'signal': 'Bearish',
                    'close': current['EURUSD_Close']
                })
            
            # ПОГЛОЩЕНИЕ БЫЧЬЕ (Bullish Engulfing)
            if (i > 0 and
                prev['Is_Bullish'] == 0 and
                current['Is_Bullish'] == 1 and
                current['EURUSD_Open'] < prev['EURUSD_Close'] and
                current['EURUSD_Close'] > prev['EURUSD_Open']):
                patterns.append({
                    'idx': i,
                    'date': df.index[i],
                    'pattern': 'Bullish Engulfing',
                    'signal': 'Bullish',
                    'close': current['EURUSD_Close']
                })
            
            # ПОГЛОЩЕНИЕ МЕДВЕЖЬЕ (Bearish Engulfing)
            if (i > 0 and
                prev['Is_Bullish'] == 1 and
                current['Is_Bullish'] == 0 and
                current['EURUSD_Open'] > prev['EURUSD_Close'] and
                current['EURUSD_Close'] < prev['EURUSD_Open']):
                patterns.append({
                    'idx': i,
                    'date': df.index[i],
                    'pattern': 'Bearish Engulfing',
                    'signal': 'Bearish',
                    'close': current['EURUSD_Close']
                })
            
            # ДОДЖИ (Doji)
            if current['Body'] < current['Range'] * 0.1:
                patterns.append({
                    'idx': i,
                    'date': df.index[i],
                    'pattern': 'Doji',
                    'signal': 'Neutral',
                    'close': current['EURUSD_Close']
                })
        
        return patterns
    
    def detect_all_patterns(self):
        """Детекция всех паттернов"""
        print("\n" + "="*70)
        print("🔍 РАСПОЗНАВАНИЕ ТЕХНИЧЕСКИХ ПАТТЕРНОВ")
        print("="*70)
        
        # Классические фигуры
        print("\n📊 Классические фигуры:")
        
        double_tops = self.detect_double_top()
        print(f"  Двойная вершина: {len(double_tops)}")
        
        double_bottoms = self.detect_double_bottom()
        print(f"  Двойное дно: {len(double_bottoms)}")
        
        h_s = self.detect_head_and_shoulders()
        print(f"  Голова и плечи: {len(h_s)}")
        
        # Свечные паттерны
        print("\n🕯️  Свечные паттерны:")
        candle_patterns = self.detect_candle_patterns()
        
        # Подсчет по типам
        pattern_counts = {}
        for p in candle_patterns:
            ptype = p['pattern']
            pattern_counts[ptype] = pattern_counts.get(ptype, 0) + 1
        
        for ptype, count in sorted(pattern_counts.items()):
            print(f"  {ptype}: {count}")
        
        # Сохранение в словарь
        self.patterns = {
            'double_top': double_tops,
            'double_bottom': double_bottoms,
            'head_shoulders': h_s,
            'candle_patterns': candle_patterns
        }
        
        print(f"\n✅ Всего найдено паттернов: {len(double_tops) + len(double_bottoms) + len(h_s) + len(candle_patterns)}")
        
        return self.patterns
    
    def get_pattern_statistics(self):
        """Статистика эффективности паттернов"""
        print("\n" + "="*70)
        print("📊 СТАТИСТИКА ЭФФЕКТИВНОСТИ ПАТТЕРНОВ")
        print("="*70)
        
        if not self.patterns:
            print("⚠️ Сначала запустите detect_all_patterns()")
            return
        
        # Статистика по свечным паттернам
        candle_patterns = self.patterns['candle_patterns']
        
        pattern_stats = {}
        
        for pattern in candle_patterns:
            if pattern['idx'] + 5 < len(self.prices_df):
                ptype = pattern['pattern']
                
                # Цена паттерна
                entry_price = self.prices_df.iloc[pattern['idx']]['EURUSD_Close']
                
                # Цена через 5 дней
                future_price = self.prices_df.iloc[pattern['idx'] + 5]['EURUSD_Close']
                
                # Доходность
                return_pct = (future_price - entry_price) / entry_price * 100
                
                if ptype not in pattern_stats:
                    pattern_stats[ptype] = {'returns': [], 'count': 0}
                
                pattern_stats[ptype]['returns'].append(return_pct)
                pattern_stats[ptype]['count'] += 1
        
        # Вывод статистики
        print("\n📈 Средняя доходность через 5 дней:")
        print("-" * 70)
        for ptype, stats in sorted(pattern_stats.items()):
            returns = stats['returns']
            avg_return = np.mean(returns)
            win_rate = (np.array(returns) > 0).mean() * 100
            print(f"  {ptype:25}: Средн. доход={avg_return:+.2f}%, Винрейт={win_rate:.1f}%, Кол-во={stats['count']}")
        
        return pattern_stats
    
    def save_patterns(self, output_dir='/home/ubuntu/eurusd_analysis/results'):
        """Сохранение паттернов в CSV"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Свечные паттерны
        if 'candle_patterns' in self.patterns:
            df = pd.DataFrame(self.patterns['candle_patterns'])
            if not df.empty:
                df.to_csv(f'{output_dir}/candle_patterns.csv', index=False)
                print(f"\n💾 Сохранено: candle_patterns.csv")
        
        # Двойные вершины
        if 'double_top' in self.patterns and self.patterns['double_top']:
            df = pd.DataFrame(self.patterns['double_top'])
            df.to_csv(f'{output_dir}/double_tops.csv', index=False)
            print(f"💾 Сохранено: double_tops.csv")
        
        # Двойные днища
        if 'double_bottom' in self.patterns and self.patterns['double_bottom']:
            df = pd.DataFrame(self.patterns['double_bottom'])
            df.to_csv(f'{output_dir}/double_bottoms.csv', index=False)
            print(f"💾 Сохранено: double_bottoms.csv")
        
        # Голова и плечи
        if 'head_shoulders' in self.patterns and self.patterns['head_shoulders']:
            df = pd.DataFrame(self.patterns['head_shoulders'])
            df.to_csv(f'{output_dir}/head_shoulders.csv', index=False)
            print(f"💾 Сохранено: head_shoulders.csv")


def main():
    """Главная функция"""
    # Загрузка данных
    prices_df = pd.read_csv('/home/ubuntu/eurusd_analysis/data/master_data.csv',
                           index_col=0, parse_dates=True)
    
    print(f"\n📊 Загружено {len(prices_df)} дней данных")
    
    # Создание детектора
    detector = PatternDetector(prices_df)
    
    # Детекция всех паттернов
    patterns = detector.detect_all_patterns()
    
    # Статистика эффективности
    stats = detector.get_pattern_statistics()
    
    # Сохранение
    detector.save_patterns()
    
    print("\n" + "="*70)
    print("✅ РАСПОЗНАВАНИЕ ПАТТЕРНОВ ЗАВЕРШЕНО")
    print("="*70)
    
    return detector


if __name__ == "__main__":
    detector = main()
