"""
Модуль детекции технических паттернов
Автор: Trading System
Дата: 2026-05-23

Реализованные паттерны:
- Голова и плечи (Head and Shoulders) - обычная и перевернутая
- Двойное дно и двойная вершина (Double Top/Bottom)
- Треугольники (Triangles) - восходящий, нисходящий, симметричный
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Pattern:
    """Класс для хранения информации о найденном паттерне"""
    pattern_type: str  # Тип паттерна
    direction: str  # 'bullish' или 'bearish'
    start_idx: int  # Индекс начала паттерна
    end_idx: int  # Индекс завершения паттерна
    breakout_idx: Optional[int]  # Индекс пробоя (если есть)
    key_levels: dict  # Ключевые уровни (neckline, support, resistance и т.д.)
    confidence: float  # Уверенность в паттерне (0-1)
    timestamp: pd.Timestamp  # Время формирования
    price_at_formation: float  # Цена на момент формирования
    
    
class PatternDetector:
    """Класс для детекции технических паттернов"""
    
    def __init__(self, df: pd.DataFrame, timeframe: str):
        """
        Инициализация детектора
        
        Args:
            df: DataFrame с OHLCV данными
            timeframe: строка с таймфреймом ('1h', '4h', '1d')
        """
        self.df = df.copy()
        self.timeframe = timeframe
        self.patterns = []
        
        # Параметры детекции в зависимости от таймфрейма
        if timeframe == '1h':
            self.window = 20  # окно для поиска экстремумов
            self.min_bars_between_peaks = 5
        elif timeframe == '4h':
            self.window = 15
            self.min_bars_between_peaks = 3
        else:  # 1d
            self.window = 10
            self.min_bars_between_peaks = 2
    
    def find_pivots(self):
        """
        Найти все пивоты (локальные максимумы и минимумы)
        
        Returns:
            Tuple с индексами максимумов и минимумов
        """
        # Поиск локальных максимумов
        highs = argrelextrema(
            self.df['high'].values, 
            np.greater, 
            order=self.window
        )[0]
        
        # Поиск локальных минимумов
        lows = argrelextrema(
            self.df['low'].values, 
            np.less, 
            order=self.window
        )[0]
        
        return highs, lows
    
    def detect_head_and_shoulders(self) -> List[Pattern]:
        """
        Детекция паттерна "Голова и плечи" (обычный и перевернутый)
        
        Критерии:
        - 3 пика (для обычного) или 3 впадины (для перевернутого)
        - Средний экстремум выше/ниже боковых
        - Симметрия боковых экстремумов (±10%)
        - Линия шеи соединяет два минимума между пиками
        
        Returns:
            Список найденных паттернов
        """
        patterns = []
        highs, lows = self.find_pivots()
        
        # ===== ОБЫЧНАЯ ГОЛОВА И ПЛЕЧИ (медвежья) =====
        # Ищем 3 последовательных максимума
        for i in range(len(highs) - 2):
            left_shoulder_idx = highs[i]
            head_idx = highs[i + 1]
            right_shoulder_idx = highs[i + 2]
            
            # Проверка минимального расстояния
            if (head_idx - left_shoulder_idx < self.min_bars_between_peaks or 
                right_shoulder_idx - head_idx < self.min_bars_between_peaks):
                continue
            
            left_shoulder = self.df['high'].iloc[left_shoulder_idx]
            head = self.df['high'].iloc[head_idx]
            right_shoulder = self.df['high'].iloc[right_shoulder_idx]
            
            # Проверка: голова выше плеч
            if head <= left_shoulder or head <= right_shoulder:
                continue
            
            # Проверка симметрии плеч (±15% допуск)
            shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
            if shoulder_diff > 0.15:
                continue
            
            # Поиск минимумов между пиками для линии шеи
            lows_between_left_head = [idx for idx in lows if left_shoulder_idx < idx < head_idx]
            lows_between_head_right = [idx for idx in lows if head_idx < idx < right_shoulder_idx]
            
            if not lows_between_left_head or not lows_between_head_right:
                continue
            
            # Линия шеи
            neckline_left_idx = lows_between_left_head[-1]
            neckline_right_idx = lows_between_head_right[0]
            neckline_left = self.df['low'].iloc[neckline_left_idx]
            neckline_right = self.df['low'].iloc[neckline_right_idx]
            
            # Средняя линия шеи
            neckline = (neckline_left + neckline_right) / 2
            
            # Расчет уверенности
            symmetry_score = 1 - shoulder_diff
            head_prominence = (head - max(left_shoulder, right_shoulder)) / head
            confidence = min(0.5 + symmetry_score * 0.3 + head_prominence * 0.2, 1.0)
            
            pattern = Pattern(
                pattern_type='Head_and_Shoulders',
                direction='bearish',
                start_idx=left_shoulder_idx,
                end_idx=right_shoulder_idx,
                breakout_idx=None,
                key_levels={
                    'left_shoulder': left_shoulder,
                    'head': head,
                    'right_shoulder': right_shoulder,
                    'neckline': neckline,
                    'target': neckline - (head - neckline)  # Проекция высоты
                },
                confidence=confidence,
                timestamp=self.df.index[right_shoulder_idx],
                price_at_formation=right_shoulder
            )
            patterns.append(pattern)
        
        # ===== ПЕРЕВЕРНУТАЯ ГОЛОВА И ПЛЕЧИ (бычья) =====
        # Ищем 3 последовательных минимума
        for i in range(len(lows) - 2):
            left_shoulder_idx = lows[i]
            head_idx = lows[i + 1]
            right_shoulder_idx = lows[i + 2]
            
            # Проверка минимального расстояния
            if (head_idx - left_shoulder_idx < self.min_bars_between_peaks or 
                right_shoulder_idx - head_idx < self.min_bars_between_peaks):
                continue
            
            left_shoulder = self.df['low'].iloc[left_shoulder_idx]
            head = self.df['low'].iloc[head_idx]
            right_shoulder = self.df['low'].iloc[right_shoulder_idx]
            
            # Проверка: голова ниже плеч
            if head >= left_shoulder or head >= right_shoulder:
                continue
            
            # Проверка симметрии плеч
            shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
            if shoulder_diff > 0.15:
                continue
            
            # Поиск максимумов между впадинами для линии шеи
            highs_between_left_head = [idx for idx in highs if left_shoulder_idx < idx < head_idx]
            highs_between_head_right = [idx for idx in highs if head_idx < idx < right_shoulder_idx]
            
            if not highs_between_left_head or not highs_between_head_right:
                continue
            
            # Линия шеи
            neckline_left_idx = highs_between_left_head[-1]
            neckline_right_idx = highs_between_head_right[0]
            neckline_left = self.df['high'].iloc[neckline_left_idx]
            neckline_right = self.df['high'].iloc[neckline_right_idx]
            
            neckline = (neckline_left + neckline_right) / 2
            
            # Расчет уверенности
            symmetry_score = 1 - shoulder_diff
            head_prominence = (min(left_shoulder, right_shoulder) - head) / min(left_shoulder, right_shoulder)
            confidence = min(0.5 + symmetry_score * 0.3 + head_prominence * 0.2, 1.0)
            
            pattern = Pattern(
                pattern_type='Inverse_Head_and_Shoulders',
                direction='bullish',
                start_idx=left_shoulder_idx,
                end_idx=right_shoulder_idx,
                breakout_idx=None,
                key_levels={
                    'left_shoulder': left_shoulder,
                    'head': head,
                    'right_shoulder': right_shoulder,
                    'neckline': neckline,
                    'target': neckline + (neckline - head)
                },
                confidence=confidence,
                timestamp=self.df.index[right_shoulder_idx],
                price_at_formation=right_shoulder
            )
            patterns.append(pattern)
        
        return patterns
    
    def detect_double_top_bottom(self) -> List[Pattern]:
        """
        Детекция двойной вершины и двойного дна
        
        Критерии:
        - 2 примерно равных экстремума (±3% от уровня)
        - Минимальное расстояние между пиками
        - Подтверждение пробоем промежуточного уровня
        
        Returns:
            Список найденных паттернов
        """
        patterns = []
        highs, lows = self.find_pivots()
        
        # ===== ДВОЙНАЯ ВЕРШИНА (медвежья) =====
        for i in range(len(highs) - 1):
            first_top_idx = highs[i]
            
            for j in range(i + 1, len(highs)):
                second_top_idx = highs[j]
                
                # Проверка минимального расстояния
                if second_top_idx - first_top_idx < self.min_bars_between_peaks * 2:
                    continue
                
                first_top = self.df['high'].iloc[first_top_idx]
                second_top = self.df['high'].iloc[second_top_idx]
                
                # Проверка схожести уровней (±3%)
                diff_pct = abs(first_top - second_top) / first_top
                if diff_pct > 0.03:
                    continue
                
                # Найти минимум между вершинами (уровень поддержки)
                lows_between = [idx for idx in lows if first_top_idx < idx < second_top_idx]
                if not lows_between:
                    continue
                
                support_idx = lows_between[np.argmin([self.df['low'].iloc[idx] for idx in lows_between])]
                support = self.df['low'].iloc[support_idx]
                
                # Проверка, что минимум значительно ниже вершин (минимум 1%)
                if (min(first_top, second_top) - support) / min(first_top, second_top) < 0.01:
                    continue
                
                # Уверенность
                similarity = 1 - diff_pct / 0.03
                depth = (min(first_top, second_top) - support) / min(first_top, second_top)
                confidence = min(0.6 + similarity * 0.2 + depth * 0.2, 1.0)
                
                pattern = Pattern(
                    pattern_type='Double_Top',
                    direction='bearish',
                    start_idx=first_top_idx,
                    end_idx=second_top_idx,
                    breakout_idx=None,
                    key_levels={
                        'first_top': first_top,
                        'second_top': second_top,
                        'support': support,
                        'target': support - (min(first_top, second_top) - support)
                    },
                    confidence=confidence,
                    timestamp=self.df.index[second_top_idx],
                    price_at_formation=second_top
                )
                patterns.append(pattern)
                break  # Берем только первую подходящую пару
        
        # ===== ДВОЙНОЕ ДНО (бычье) =====
        for i in range(len(lows) - 1):
            first_bottom_idx = lows[i]
            
            for j in range(i + 1, len(lows)):
                second_bottom_idx = lows[j]
                
                # Проверка минимального расстояния
                if second_bottom_idx - first_bottom_idx < self.min_bars_between_peaks * 2:
                    continue
                
                first_bottom = self.df['low'].iloc[first_bottom_idx]
                second_bottom = self.df['low'].iloc[second_bottom_idx]
                
                # Проверка схожести уровней
                diff_pct = abs(first_bottom - second_bottom) / first_bottom
                if diff_pct > 0.03:
                    continue
                
                # Найти максимум между днами (уровень сопротивления)
                highs_between = [idx for idx in highs if first_bottom_idx < idx < second_bottom_idx]
                if not highs_between:
                    continue
                
                resistance_idx = highs_between[np.argmax([self.df['high'].iloc[idx] for idx in highs_between])]
                resistance = self.df['high'].iloc[resistance_idx]
                
                # Проверка глубины
                if (resistance - max(first_bottom, second_bottom)) / resistance < 0.01:
                    continue
                
                # Уверенность
                similarity = 1 - diff_pct / 0.03
                depth = (resistance - max(first_bottom, second_bottom)) / resistance
                confidence = min(0.6 + similarity * 0.2 + depth * 0.2, 1.0)
                
                pattern = Pattern(
                    pattern_type='Double_Bottom',
                    direction='bullish',
                    start_idx=first_bottom_idx,
                    end_idx=second_bottom_idx,
                    breakout_idx=None,
                    key_levels={
                        'first_bottom': first_bottom,
                        'second_bottom': second_bottom,
                        'resistance': resistance,
                        'target': resistance + (resistance - max(first_bottom, second_bottom))
                    },
                    confidence=confidence,
                    timestamp=self.df.index[second_bottom_idx],
                    price_at_formation=second_bottom
                )
                patterns.append(pattern)
                break
        
        return patterns
    
    def detect_triangles(self) -> List[Pattern]:
        """
        Детекция треугольников (восходящий, нисходящий, симметричный)
        
        Критерии:
        - Сужающийся диапазон (сходящиеся трендовые линии)
        - Минимум 4-5 касаний линий
        - Для восходящего: горизонтальное сопротивление + восходящая поддержка
        - Для нисходящего: нисходящее сопротивление + горизонтальная поддержка
        - Для симметричного: обе линии сходятся
        
        Returns:
            Список найденных паттернов
        """
        patterns = []
        highs, lows = self.find_pivots()
        
        # Минимальное количество точек для формирования треугольника
        min_points = 4
        
        # Окно поиска (последние N баров)
        window_size = min(100, len(self.df) // 2)
        
        if len(highs) < min_points or len(lows) < min_points:
            return patterns
        
        # Берем последние максимумы и минимумы в окне
        recent_highs = [idx for idx in highs if idx >= len(self.df) - window_size]
        recent_lows = [idx for idx in lows if idx >= len(self.df) - window_size]
        
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return patterns
        
        # ===== ВОСХОДЯЩИЙ ТРЕУГОЛЬНИК (бычий) =====
        # Проверяем горизонтальное сопротивление
        high_values = [self.df['high'].iloc[idx] for idx in recent_highs]
        high_std = np.std(high_values)
        high_mean = np.mean(high_values)
        
        # Горизонтальное сопротивление: малая вариация максимумов
        if high_std / high_mean < 0.01:  # Вариация < 1%
            # Проверяем восходящую поддержку
            low_values = [(idx, self.df['low'].iloc[idx]) for idx in recent_lows]
            if len(low_values) >= 2:
                # Линейная регрессия для линии поддержки
                x = np.array([v[0] for v in low_values])
                y = np.array([v[1] for v in low_values])
                slope, intercept = np.polyfit(x, y, 1)
                
                # Восходящая линия: положительный наклон
                if slope > 0:
                    # Проверка качества касаний
                    predicted = slope * x + intercept
                    r_squared = 1 - (np.sum((y - predicted) ** 2) / np.sum((y - np.mean(y)) ** 2))
                    
                    if r_squared > 0.7:  # Хорошее соответствие
                        start_idx = recent_lows[0]
                        end_idx = recent_highs[-1]
                        
                        # Проверка сужения диапазона
                        range_start = self.df['high'].iloc[start_idx] - self.df['low'].iloc[start_idx]
                        range_end = high_mean - self.df['low'].iloc[recent_lows[-1]]
                        
                        if range_end < range_start * 0.7:  # Диапазон сузился минимум на 30%
                            confidence = min(0.6 + r_squared * 0.3, 0.95)
                            
                            pattern = Pattern(
                                pattern_type='Ascending_Triangle',
                                direction='bullish',
                                start_idx=start_idx,
                                end_idx=end_idx,
                                breakout_idx=None,
                                key_levels={
                                    'resistance': high_mean,
                                    'support_slope': slope,
                                    'support_intercept': intercept,
                                    'target': high_mean + (high_mean - self.df['low'].iloc[start_idx])
                                },
                                confidence=confidence,
                                timestamp=self.df.index[end_idx],
                                price_at_formation=self.df['close'].iloc[end_idx]
                            )
                            patterns.append(pattern)
        
        # ===== НИСХОДЯЩИЙ ТРЕУГОЛЬНИК (медвежий) =====
        # Проверяем горизонтальную поддержку
        low_values_check = [self.df['low'].iloc[idx] for idx in recent_lows]
        low_std = np.std(low_values_check)
        low_mean = np.mean(low_values_check)
        
        if low_std / low_mean < 0.01:
            # Проверяем нисходящее сопротивление
            high_values_trend = [(idx, self.df['high'].iloc[idx]) for idx in recent_highs]
            if len(high_values_trend) >= 2:
                x = np.array([v[0] for v in high_values_trend])
                y = np.array([v[1] for v in high_values_trend])
                slope, intercept = np.polyfit(x, y, 1)
                
                # Нисходящая линия: отрицательный наклон
                if slope < 0:
                    predicted = slope * x + intercept
                    r_squared = 1 - (np.sum((y - predicted) ** 2) / np.sum((y - np.mean(y)) ** 2))
                    
                    if r_squared > 0.7:
                        start_idx = recent_highs[0]
                        end_idx = recent_lows[-1]
                        
                        range_start = self.df['high'].iloc[start_idx] - self.df['low'].iloc[start_idx]
                        range_end = self.df['high'].iloc[recent_highs[-1]] - low_mean
                        
                        if range_end < range_start * 0.7:
                            confidence = min(0.6 + r_squared * 0.3, 0.95)
                            
                            pattern = Pattern(
                                pattern_type='Descending_Triangle',
                                direction='bearish',
                                start_idx=start_idx,
                                end_idx=end_idx,
                                breakout_idx=None,
                                key_levels={
                                    'support': low_mean,
                                    'resistance_slope': slope,
                                    'resistance_intercept': intercept,
                                    'target': low_mean - (self.df['high'].iloc[start_idx] - low_mean)
                                },
                                confidence=confidence,
                                timestamp=self.df.index[end_idx],
                                price_at_formation=self.df['close'].iloc[end_idx]
                            )
                            patterns.append(pattern)
        
        # ===== СИММЕТРИЧНЫЙ ТРЕУГОЛЬНИК (может быть бычьим или медвежьим) =====
        # Обе линии должны сходиться
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            # Линия сопротивления
            high_data = [(idx, self.df['high'].iloc[idx]) for idx in recent_highs]
            x_high = np.array([v[0] for v in high_data])
            y_high = np.array([v[1] for v in high_data])
            slope_high, intercept_high = np.polyfit(x_high, y_high, 1)
            
            # Линия поддержки
            low_data = [(idx, self.df['low'].iloc[idx]) for idx in recent_lows]
            x_low = np.array([v[0] for v in low_data])
            y_low = np.array([v[1] for v in low_data])
            slope_low, intercept_low = np.polyfit(x_low, y_low, 1)
            
            # Симметричный: сопротивление нисходящее, поддержка восходящая
            if slope_high < 0 and slope_low > 0:
                # Проверка качества линий
                pred_high = slope_high * x_high + intercept_high
                r2_high = 1 - (np.sum((y_high - pred_high) ** 2) / np.sum((y_high - np.mean(y_high)) ** 2))
                
                pred_low = slope_low * x_low + intercept_low
                r2_low = 1 - (np.sum((y_low - pred_low) ** 2) / np.sum((y_low - np.mean(y_low)) ** 2))
                
                if r2_high > 0.65 and r2_low > 0.65:
                    start_idx = min(recent_highs[0], recent_lows[0])
                    end_idx = max(recent_highs[-1], recent_lows[-1])
                    
                    # Проверка сужения
                    range_start = self.df['high'].iloc[recent_highs[0]] - self.df['low'].iloc[recent_lows[0]]
                    range_end = self.df['high'].iloc[recent_highs[-1]] - self.df['low'].iloc[recent_lows[-1]]
                    
                    if range_end < range_start * 0.6:
                        avg_r2 = (r2_high + r2_low) / 2
                        confidence = min(0.55 + avg_r2 * 0.35, 0.90)
                        
                        # Направление зависит от пробоя (пока неизвестно)
                        pattern = Pattern(
                            pattern_type='Symmetrical_Triangle',
                            direction='neutral',  # Будет определено при пробое
                            start_idx=start_idx,
                            end_idx=end_idx,
                            breakout_idx=None,
                            key_levels={
                                'resistance_slope': slope_high,
                                'resistance_intercept': intercept_high,
                                'support_slope': slope_low,
                                'support_intercept': intercept_low,
                                'apex_distance': abs(slope_high - slope_low)
                            },
                            confidence=confidence,
                            timestamp=self.df.index[end_idx],
                            price_at_formation=self.df['close'].iloc[end_idx]
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def detect_all_patterns(self) -> List[Pattern]:
        """
        Детектировать все типы паттернов
        
        Returns:
            Список всех найденных паттернов
        """
        all_patterns = []
        
        # Детекция каждого типа паттернов
        hs_patterns = self.detect_head_and_shoulders()
        dt_patterns = self.detect_double_top_bottom()
        tri_patterns = self.detect_triangles()
        
        all_patterns.extend(hs_patterns)
        all_patterns.extend(dt_patterns)
        all_patterns.extend(tri_patterns)
        
        # Сортировка по времени
        all_patterns.sort(key=lambda p: p.timestamp)
        
        return all_patterns


if __name__ == "__main__":
    # Тестовый запуск
    from data_loader import DataLoader
    
    loader = DataLoader(
        hourly_file='/home/ubuntu/.external_service_outputs/fetch_market_data_output_1779529162.json',
        daily_file='/home/ubuntu/.external_service_outputs/fetch_market_data_output_1779529148.json'
    )
    df_1h, df_4h, df_1d = loader.load_data()
    
    print("\n=== Детекция паттернов на дневном таймфрейме ===")
    detector_1d = PatternDetector(df_1d, '1d')
    patterns_1d = detector_1d.detect_all_patterns()
    
    print(f"\nНайдено паттернов: {len(patterns_1d)}")
    for p in patterns_1d[:5]:  # Показываем первые 5
        print(f"\n{p.pattern_type} ({p.direction})")
        print(f"  Время: {p.timestamp}")
        print(f"  Уверенность: {p.confidence:.2f}")
        print(f"  Ключевые уровни: {p.key_levels}")
