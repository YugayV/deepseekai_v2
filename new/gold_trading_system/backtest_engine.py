"""
Модуль бэктестинга торговой системы
Автор: Trading System
Дата: 2026-05-23

Функционал:
- Мультитаймфреймовый анализ паттернов
- Торговая логика с управлением рисками
- Расчет всех метрик производительности
- Учет издержек (спреды, проскальзывание)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from pattern_detector import PatternDetector, Pattern


@dataclass
class Trade:
    """Класс для хранения информации о сделке"""
    trade_id: int
    pattern_type: str
    timeframe: str
    direction: str  # 'long' или 'short'
    detection_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    exit_time: Optional[pd.Timestamp]
    exit_price: Optional[float]
    exit_reason: Optional[str]  # 'target', 'stop', 'timeout'
    shares: int
    gross_pnl: float
    costs: float
    net_pnl: float
    r_multiple: float  # Прибыль в R (R = размер риска)
    mtf_score: int  # Количество совпадающих таймфреймов
    confidence: float


class BacktestEngine:
    """Движок бэктестинга"""
    
    def __init__(self, 
                 df_1h: pd.DataFrame, 
                 df_4h: pd.DataFrame, 
                 df_1d: pd.DataFrame,
                 initial_capital: float = 100000,
                 risk_per_trade: float = 0.02,  # 2% риска на сделку
                 spread_pct: float = 0.0005,  # 0.05% спред
                 slippage_pct: float = 0.0003):  # 0.03% проскальзывание
        """
        Инициализация движка
        
        Args:
            df_1h, df_4h, df_1d: DataFrames с данными
            initial_capital: начальный капитал
            risk_per_trade: риск на сделку (% от капитала)
            spread_pct: спред (% от цены)
            slippage_pct: проскальзывание (% от цены)
        """
        self.df_1h = df_1h
        self.df_4h = df_4h
        self.df_1d = df_1d
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.spread_pct = spread_pct
        self.slippage_pct = slippage_pct
        
        self.capital = initial_capital
        self.trades: List[Trade] = []
        self.open_positions: List[Dict] = []
        self.equity_curve = []
        
    def detect_patterns_all_timeframes(self):
        """Детектировать паттерны на всех таймфреймах"""
        print("\n=== Детекция паттернов ===")
        
        detector_1h = PatternDetector(self.df_1h, '1h')
        detector_4h = PatternDetector(self.df_4h, '4h')
        detector_1d = PatternDetector(self.df_1d, '1d')
        
        patterns_1h = detector_1h.detect_all_patterns()
        patterns_4h = detector_4h.detect_all_patterns()
        patterns_1d = detector_1d.detect_all_patterns()
        
        print(f"Паттернов на 1h: {len(patterns_1h)}")
        print(f"Паттернов на 4h: {len(patterns_4h)}")
        print(f"Паттернов на 1d: {len(patterns_1d)}")
        
        return {
            '1h': patterns_1h,
            '4h': patterns_4h,
            '1d': patterns_1d
        }
    
    def check_mtf_confirmation(self, pattern: Pattern, all_patterns: Dict, 
                                window_days: int = 5) -> int:
        """
        Проверить мультитаймфреймовое подтверждение
        
        Args:
            pattern: паттерн для проверки
            all_patterns: словарь со всеми паттернами
            window_days: окно валидности (дни)
            
        Returns:
            Количество подтверждающих таймфреймов (1-3)
        """
        pattern_time = pattern.timestamp
        pattern_dir = pattern.direction
        window = timedelta(days=window_days)
        
        mtf_score = 1  # Сам паттерн уже считается
        
        # Проверяем другие таймфреймы
        for tf, patterns in all_patterns.items():
            for other_pattern in patterns:
                # Пропускаем тот же паттерн
                if (other_pattern.timestamp == pattern_time and 
                    other_pattern.pattern_type == pattern.pattern_type):
                    continue
                
                # Проверяем совпадение времени (в пределах окна)
                time_diff = abs((other_pattern.timestamp - pattern_time).total_seconds() / 86400)
                if time_diff <= window_days:
                    # Проверяем совпадение направления
                    if other_pattern.direction == pattern_dir:
                        mtf_score += 1
                        break  # Один паттерн с этого ТФ достаточно
        
        return min(mtf_score, 3)  # Максимум 3
    
    def calculate_position_size(self, entry_price: float, stop_price: float) -> int:
        """
        Рассчитать размер позиции на основе риска
        
        Args:
            entry_price: цена входа
            stop_price: цена стопа
            
        Returns:
            Количество акций
        """
        risk_amount = self.capital * self.risk_per_trade
        risk_per_share = abs(entry_price - stop_price)
        
        if risk_per_share == 0:
            return 0
        
        shares = int(risk_amount / risk_per_share)
        
        # Проверка достаточности капитала
        position_cost = shares * entry_price
        if position_cost > self.capital * 0.95:  # Не более 95% капитала
            shares = int((self.capital * 0.95) / entry_price)
        
        return max(shares, 0)
    
    def check_breakout(self, pattern: Pattern, df: pd.DataFrame, 
                       current_idx: int) -> Optional[Dict]:
        """
        Проверить пробой ключевого уровня паттерна
        
        Args:
            pattern: паттерн
            df: DataFrame с данными
            current_idx: текущий индекс бара
            
        Returns:
            Словарь с информацией о входе или None
        """
        if current_idx >= len(df):
            return None
        
        current_bar = df.iloc[current_idx]
        current_price = current_bar['close']
        current_high = current_bar['high']
        current_low = current_bar['low']
        
        # Определяем ключевой уровень для пробоя
        if pattern.pattern_type in ['Head_and_Shoulders', 'Double_Top']:
            # Медвежьи паттерны: пробой поддержки вниз
            if 'neckline' in pattern.key_levels:
                breakout_level = pattern.key_levels['neckline']
            elif 'support' in pattern.key_levels:
                breakout_level = pattern.key_levels['support']
            else:
                return None
            
            # Пробой вниз: цена закрылась ниже уровня
            if current_price < breakout_level * 0.998:  # С небольшим буфером
                return {
                    'direction': 'short',
                    'entry_price': current_price,
                    'stop_price': pattern.key_levels.get('head', 
                                  pattern.key_levels.get('first_top', breakout_level * 1.02)),
                    'target_price': pattern.key_levels.get('target', breakout_level * 0.95)
                }
        
        elif pattern.pattern_type in ['Inverse_Head_and_Shoulders', 'Double_Bottom']:
            # Бычьи паттерны: пробой сопротивления вверх
            if 'neckline' in pattern.key_levels:
                breakout_level = pattern.key_levels['neckline']
            elif 'resistance' in pattern.key_levels:
                breakout_level = pattern.key_levels['resistance']
            else:
                return None
            
            # Пробой вверх
            if current_price > breakout_level * 1.002:
                return {
                    'direction': 'long',
                    'entry_price': current_price,
                    'stop_price': pattern.key_levels.get('head', 
                                  pattern.key_levels.get('first_bottom', breakout_level * 0.98)),
                    'target_price': pattern.key_levels.get('target', breakout_level * 1.05)
                }
        
        elif pattern.pattern_type == 'Ascending_Triangle':
            # Бычий: пробой сопротивления
            breakout_level = pattern.key_levels['resistance']
            if current_price > breakout_level * 1.002:
                return {
                    'direction': 'long',
                    'entry_price': current_price,
                    'stop_price': breakout_level * 0.97,
                    'target_price': pattern.key_levels.get('target', breakout_level * 1.05)
                }
        
        elif pattern.pattern_type == 'Descending_Triangle':
            # Медвежий: пробой поддержки
            breakout_level = pattern.key_levels['support']
            if current_price < breakout_level * 0.998:
                return {
                    'direction': 'short',
                    'entry_price': current_price,
                    'stop_price': breakout_level * 1.03,
                    'target_price': pattern.key_levels.get('target', breakout_level * 0.95)
                }
        
        elif pattern.pattern_type == 'Symmetrical_Triangle':
            # Может пробиться в любую сторону
            resistance = pattern.key_levels['resistance_intercept']
            support = pattern.key_levels['support_intercept']
            
            if current_price > resistance * 1.002:
                return {
                    'direction': 'long',
                    'entry_price': current_price,
                    'stop_price': support,
                    'target_price': current_price * 1.05
                }
            elif current_price < support * 0.998:
                return {
                    'direction': 'short',
                    'entry_price': current_price,
                    'stop_price': resistance,
                    'target_price': current_price * 0.95
                }
        
        return None
    
    def run_backtest(self):
        """Запустить полный бэктест"""
        print("\n=== Запуск бэктестинга ===")
        
        # Детекция паттернов
        all_patterns = self.detect_patterns_all_timeframes()
        
        # Объединяем все паттерны с их таймфреймами
        all_patterns_flat = []
        for tf, patterns in all_patterns.items():
            for p in patterns:
                all_patterns_flat.append((tf, p))
        
        # Сортируем по времени
        all_patterns_flat.sort(key=lambda x: x[1].timestamp)
        
        print(f"\nВсего паттернов для анализа: {len(all_patterns_flat)}")
        
        # Используем дневной DataFrame как основной для симуляции
        df_main = self.df_1d.copy()
        
        trade_id = 0
        last_pattern_time = None  # Для дедупликации
        
        # Проходим по всем барам
        for i in range(len(df_main)):
            current_time = df_main.index[i]
            current_bar = df_main.iloc[i]
            
            # Обновляем эквити
            self.equity_curve.append({
                'time': current_time,
                'equity': self.capital
            })
            
            # Проверяем открытые позиции
            positions_to_close = []
            for pos_idx, pos in enumerate(self.open_positions):
                exit_reason = None
                exit_price = None
                
                # Проверка стопа и цели
                if pos['direction'] == 'long':
                    if current_bar['low'] <= pos['stop_price']:
                        exit_reason = 'stop'
                        exit_price = pos['stop_price']
                    elif current_bar['high'] >= pos['target_price']:
                        exit_reason = 'target'
                        exit_price = pos['target_price']
                elif pos['direction'] == 'short':
                    if current_bar['high'] >= pos['stop_price']:
                        exit_reason = 'stop'
                        exit_price = pos['stop_price']
                    elif current_bar['low'] <= pos['target_price']:
                        exit_reason = 'target'
                        exit_price = pos['target_price']
                
                # Закрытие по таймауту (30 дней)
                days_open = (current_time - pos['entry_time']).days
                if days_open > 30 and exit_reason is None:
                    exit_reason = 'timeout'
                    exit_price = current_bar['close']
                
                if exit_reason:
                    positions_to_close.append((pos_idx, exit_reason, exit_price))
            
            # Закрываем позиции
            for pos_idx, exit_reason, exit_price in reversed(positions_to_close):
                pos = self.open_positions.pop(pos_idx)
                self._close_position(pos, current_time, exit_price, exit_reason)
            
            # Ищем новые паттерны для входа
            for tf, pattern in all_patterns_flat:
                # Паттерн должен быть сформирован до текущего времени
                if pattern.timestamp > current_time:
                    continue
                
                # Дедупликация: не входим в похожие паттерны слишком часто
                if last_pattern_time is not None:
                    if (current_time - last_pattern_time).days < 3:
                        continue
                
                # Проверяем пробой на дневном таймфрейме
                breakout_info = self.check_breakout(pattern, df_main, i)
                
                if breakout_info:
                    # Мультитаймфреймовое подтверждение
                    mtf_score = self.check_mtf_confirmation(pattern, all_patterns)
                    
                    # Фильтр: требуем минимум 2 таймфрейма для входа
                    if mtf_score < 2:
                        continue
                    
                    # Рассчитываем размер позиции
                    shares = self.calculate_position_size(
                        breakout_info['entry_price'],
                        breakout_info['stop_price']
                    )
                    
                    if shares > 0:
                        # Открываем позицию
                        trade_id += 1
                        self._open_position(
                            trade_id=trade_id,
                            pattern=pattern,
                            timeframe=tf,
                            entry_time=current_time,
                            breakout_info=breakout_info,
                            shares=shares,
                            mtf_score=mtf_score
                        )
                        last_pattern_time = current_time
                        break  # Один вход за раз
        
        # Закрываем все открытые позиции в конце
        for pos in self.open_positions:
            final_price = df_main.iloc[-1]['close']
            self._close_position(pos, df_main.index[-1], final_price, 'end_of_data')
        
        self.open_positions = []
        
        print(f"\nБэктест завершен. Всего сделок: {len(self.trades)}")
        
        return self.trades
    
    def _open_position(self, trade_id: int, pattern: Pattern, timeframe: str,
                       entry_time: pd.Timestamp, breakout_info: Dict, 
                       shares: int, mtf_score: int):
        """Открыть позицию"""
        entry_price = breakout_info['entry_price']
        
        # Применяем спред и проскальзывание
        if breakout_info['direction'] == 'long':
            entry_price_adj = entry_price * (1 + self.spread_pct + self.slippage_pct)
        else:
            entry_price_adj = entry_price * (1 - self.spread_pct - self.slippage_pct)
        
        # Корректируем стоп и цель с учетом нового входа
        stop_price = breakout_info['stop_price']
        target_price = breakout_info['target_price']
        
        # Пересчитываем target с соотношением 3:1
        risk = abs(entry_price_adj - stop_price)
        if breakout_info['direction'] == 'long':
            target_price = entry_price_adj + risk * 3
        else:
            target_price = entry_price_adj - risk * 3
        
        position = {
            'trade_id': trade_id,
            'pattern_type': pattern.pattern_type,
            'timeframe': timeframe,
            'direction': breakout_info['direction'],
            'detection_time': pattern.timestamp,
            'entry_time': entry_time,
            'entry_price': entry_price_adj,
            'stop_price': stop_price,
            'target_price': target_price,
            'shares': shares,
            'mtf_score': mtf_score,
            'confidence': pattern.confidence
        }
        
        self.open_positions.append(position)
        
        # Уменьшаем капитал
        position_cost = shares * entry_price_adj
        self.capital -= position_cost
    
    def _close_position(self, pos: Dict, exit_time: pd.Timestamp, 
                        exit_price: float, exit_reason: str):
        """Закрыть позицию"""
        # Применяем спред и проскальзывание при выходе
        if pos['direction'] == 'long':
            exit_price_adj = exit_price * (1 - self.spread_pct - self.slippage_pct)
        else:
            exit_price_adj = exit_price * (1 + self.spread_pct + self.slippage_pct)
        
        # Расчет P&L
        if pos['direction'] == 'long':
            gross_pnl = (exit_price_adj - pos['entry_price']) * pos['shares']
        else:
            gross_pnl = (pos['entry_price'] - exit_price_adj) * pos['shares']
        
        # Издержки
        entry_cost = pos['shares'] * pos['entry_price'] * (self.spread_pct + self.slippage_pct)
        exit_cost = pos['shares'] * exit_price * (self.spread_pct + self.slippage_pct)
        total_costs = entry_cost + exit_cost
        
        net_pnl = gross_pnl - total_costs
        
        # R-multiple
        risk_per_share = abs(pos['entry_price'] - pos['stop_price'])
        if risk_per_share > 0:
            r_multiple = net_pnl / (risk_per_share * pos['shares'])
        else:
            r_multiple = 0
        
        # Обновляем капитал
        position_value = pos['shares'] * exit_price_adj
        self.capital += position_value + net_pnl
        
        # Создаем запись о сделке
        trade = Trade(
            trade_id=pos['trade_id'],
            pattern_type=pos['pattern_type'],
            timeframe=pos['timeframe'],
            direction=pos['direction'],
            detection_time=pos['detection_time'],
            entry_time=pos['entry_time'],
            entry_price=pos['entry_price'],
            stop_price=pos['stop_price'],
            target_price=pos['target_price'],
            exit_time=exit_time,
            exit_price=exit_price_adj,
            exit_reason=exit_reason,
            shares=pos['shares'],
            gross_pnl=gross_pnl,
            costs=total_costs,
            net_pnl=net_pnl,
            r_multiple=r_multiple,
            mtf_score=pos['mtf_score'],
            confidence=pos['confidence']
        )
        
        self.trades.append(trade)
    
    def calculate_metrics(self) -> Dict:
        """Рассчитать метрики производительности"""
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame([asdict(t) for t in self.trades])
        
        # Основные метрики
        total_trades = len(self.trades)
        winning_trades = trades_df[trades_df['net_pnl'] > 0]
        losing_trades = trades_df[trades_df['net_pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_pnl = trades_df['net_pnl'].sum()
        total_gross = trades_df['gross_pnl'].sum()
        total_costs = trades_df['costs'].sum()
        
        gross_profit = winning_trades['gross_pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['gross_pnl'].sum()) if len(losing_trades) > 0 else 0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
        avg_trade = trades_df['net_pnl'].mean()
        
        # Максимальная просадка
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = equity_df['equity'] - equity_df['cummax']
        equity_df['drawdown_pct'] = equity_df['drawdown'] / equity_df['cummax'] * 100
        
        max_drawdown = equity_df['drawdown'].min()
        max_drawdown_pct = equity_df['drawdown_pct'].min()
        
        # Sharpe Ratio (упрощенный)
        returns = trades_df['net_pnl'] / self.initial_capital
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Метрики по паттернам
        pattern_stats = trades_df.groupby('pattern_type').agg({
            'net_pnl': ['count', 'sum', 'mean'],
            'trade_id': 'count'
        }).round(2)
        
        # Метрики по таймфреймам
        timeframe_stats = trades_df.groupby('timeframe').agg({
            'net_pnl': ['count', 'sum', 'mean']
        }).round(2)
        
        final_capital = self.capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'final_capital': final_capital,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'total_costs': total_costs,
            'pattern_stats': pattern_stats,
            'timeframe_stats': timeframe_stats
        }
        
        return metrics


if __name__ == "__main__":
    # Тестовый запуск
    loader = DataLoader(
        hourly_file='/home/ubuntu/.external_service_outputs/fetch_market_data_output_1779529162.json',
        daily_file='/home/ubuntu/.external_service_outputs/fetch_market_data_output_1779529148.json'
    )
    
    df_1h, df_4h, df_1d = loader.load_data()
    
    engine = BacktestEngine(df_1h, df_4h, df_1d, 
                           initial_capital=100000, 
                           risk_per_trade=0.02)
    
    trades = engine.run_backtest()
    metrics = engine.calculate_metrics()
    
    print("\n=== РЕЗУЛЬТАТЫ БЭКТЕСТИНГА ===")
    print(f"Всего сделок: {metrics['total_trades']}")
    print(f"Прибыльных: {metrics['winning_trades']} ({metrics['win_rate']*100:.1f}%)")
    print(f"Убыточных: {metrics['losing_trades']}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Общий P&L: ${metrics['total_pnl']:.2f}")
    print(f"Доходность: {metrics['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Макс. просадка: ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2f}%)")
