"""
МОДУЛЬ ТОРГОВОЙ СТРАТЕГИИ И БЭКТЕСТИНГА
Комбинированная система с ML, корреляциями и паттернами
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

class TradingStrategy:
    """Комбинированная торговая стратегия"""
    
    def __init__(self, features_df, prices_df):
        self.features_df = features_df.copy()
        self.prices_df = prices_df.copy()
        self.trades = []
        self.equity_curve = []
        
        # Загрузка ML-модели
        self.ml_model = joblib.load('/home/ubuntu/eurusd_analysis/models/classification_XGBoost.pkl')
        self.scaler = joblib.load('/home/ubuntu/eurusd_analysis/models/scaler.pkl')
        
    def generate_signals(self):
        """Генерация торговых сигналов"""
        print("\n" + "="*70)
        print("🎯 ГЕНЕРАЦИЯ ТОРГОВЫХ СИГНАЛОВ")
        print("="*70)
        
        signals_df = self.features_df.copy()
        
        # 1. ML-сигнал (XGBoost предсказание)
        feature_cols = [col for col in self.features_df.columns 
                       if not col.startswith('Target')]
        X = self.features_df[feature_cols]
        X_scaled = self.scaler.transform(X)
        
        ml_predictions = self.ml_model.predict(X_scaled)
        signals_df['ML_Signal'] = ml_predictions  # 0=Down, 1=Up
        
        print(f"\n📊 ML-сигналы:")
        print(f"  Buy (1):  {(ml_predictions==1).sum()}")
        print(f"  Sell (0): {(ml_predictions==0).sum()}")
        
        # 2. Корреляционный сигнал (DXY как ведущий индикатор с лагом 1)
        # Если DXY вчера упал, EUR/USD сегодня вырастет (обратная корреляция -0.80)
        signals_df['DXY_Signal'] = (signals_df['DXY_Return_Lag1'] < 0).astype(int)
        
        # 3. Технический сигнал (RSI + MACD)
        signals_df['Tech_Signal'] = 0
        signals_df.loc[(signals_df['RSI_14'] < 40) & (signals_df['MACD_Diff'] > 0), 'Tech_Signal'] = 1  # Oversold + MACD bullish
        signals_df.loc[(signals_df['RSI_14'] > 60) & (signals_df['MACD_Diff'] < 0), 'Tech_Signal'] = -1  # Overbought + MACD bearish
        
        # 4. Комбинированный сигнал (нужны 2 из 3 подтверждений для входа)
        signals_df['Combined_Signal'] = 0
        
        # Бычий сигнал
        bullish_votes = (signals_df['ML_Signal'] == 1).astype(int) + \
                       (signals_df['DXY_Signal'] == 1).astype(int) + \
                       (signals_df['Tech_Signal'] == 1).astype(int)
        
        signals_df.loc[bullish_votes >= 2, 'Combined_Signal'] = 1
        
        # Медвежий сигнал
        bearish_votes = (signals_df['ML_Signal'] == 0).astype(int) + \
                       (signals_df['DXY_Signal'] == 0).astype(int) + \
                       (signals_df['Tech_Signal'] == -1).astype(int)
        
        signals_df.loc[bearish_votes >= 2, 'Combined_Signal'] = -1
        
        print(f"\n🎯 Комбинированные сигналы:")
        print(f"  Long (1):  {(signals_df['Combined_Signal']==1).sum()}")
        print(f"  Short (-1): {(signals_df['Combined_Signal']==-1).sum()}")
        print(f"  Neutral (0): {(signals_df['Combined_Signal']==0).sum()}")
        
        self.signals_df = signals_df
        return signals_df
    
    def backtest(self, initial_capital=10000, risk_per_trade=0.01, spread_pips=2):
        """
        Бэктестинг стратегии
        
        Args:
            initial_capital: Начальный капитал
            risk_per_trade: Риск на сделку (1% = 0.01)
            spread_pips: Спред в пипсах (10 pips = 0.0010)
        """
        print("\n" + "="*70)
        print("💰 БЭКТЕСТИНГ СТРАТЕГИИ")
        print("="*70)
        
        capital = initial_capital
        position = 0  # 0 = нет позиции, 1 = long, -1 = short
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        
        trades = []
        equity = [initial_capital]
        
        spread = spread_pips * 0.0001  # Конвертация пипсов в цену
        
        for i in range(len(self.signals_df)):
            row = self.signals_df.iloc[i]
            current_price = row['Close']
            signal = row['Combined_Signal']
            atr = row['ATR']
            
            # Проверка стоп-лосса и тейк-профита
            if position != 0:
                if position == 1:  # Long позиция
                    if current_price <= stop_loss or current_price >= take_profit:
                        # Закрываем позицию
                        exit_price = current_price - spread  # Spread на выход
                        pnl_pips = (exit_price - entry_price) / 0.0001
                        pnl_usd = (exit_price - entry_price) * position_size
                        capital += pnl_usd
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': row.name,
                            'direction': 'Long',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl_pips': pnl_pips,
                            'pnl_usd': pnl_usd,
                            'pnl_pct': (pnl_usd / initial_capital) * 100,
                            'reason': 'Stop Loss' if current_price <= stop_loss else 'Take Profit'
                        })
                        
                        position = 0
                        
                elif position == -1:  # Short позиция
                    if current_price >= stop_loss or current_price <= take_profit:
                        # Закрываем позицию
                        exit_price = current_price + spread
                        pnl_pips = (entry_price - exit_price) / 0.0001
                        pnl_usd = (entry_price - exit_price) * position_size
                        capital += pnl_usd
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': row.name,
                            'direction': 'Short',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl_pips': pnl_pips,
                            'pnl_usd': pnl_usd,
                            'pnl_pct': (pnl_usd / initial_capital) * 100,
                            'reason': 'Stop Loss' if current_price >= stop_loss else 'Take Profit'
                        })
                        
                        position = 0
            
            # Открытие новой позиции
            if position == 0 and signal != 0:
                # Рассчитываем стоп-лосс на основе ATR
                stop_distance = atr * 2  # 2 ATR
                
                if signal == 1:  # Long
                    entry_price = current_price + spread  # Spread на вход
                    stop_loss = entry_price - stop_distance
                    take_profit = entry_price + (stop_distance * 2)  # R:R = 1:2
                    
                    # Размер позиции (1% риска)
                    risk_usd = capital * risk_per_trade
                    pip_value = 10  # Примерное значение для стандартного лота
                    position_size = risk_usd / (stop_distance / 0.0001 * pip_value)
                    
                    position = 1
                    entry_date = row.name
                    
                elif signal == -1:  # Short
                    entry_price = current_price - spread
                    stop_loss = entry_price + stop_distance
                    take_profit = entry_price - (stop_distance * 2)
                    
                    risk_usd = capital * risk_per_trade
                    pip_value = 10
                    position_size = risk_usd / (stop_distance / 0.0001 * pip_value)
                    
                    position = -1
                    entry_date = row.name
            
            equity.append(capital)
        
        # Сохранение результатов
        self.trades = pd.DataFrame(trades)
        self.equity_curve = pd.DataFrame({
            'date': list(self.signals_df.index) + [self.signals_df.index[-1]],
            'equity': equity
        })
        
        # Расчет метрик
        self.calculate_metrics(initial_capital)
        
        return self.trades, self.equity_curve
    
    def calculate_metrics(self, initial_capital):
        """Расчет метрик производительности"""
        print("\n" + "="*70)
        print("📊 РЕЗУЛЬТАТЫ БЭКТЕСТИНГА")
        print("="*70)
        
        if len(self.trades) == 0:
            print("⚠️ Нет закрытых сделок")
            return
        
        # Основные метрики
        total_trades = len(self.trades)
        winning_trades = (self.trades['pnl_usd'] > 0).sum()
        losing_trades = (self.trades['pnl_usd'] < 0).sum()
        win_rate = winning_trades / total_trades * 100
        
        total_pnl = self.trades['pnl_usd'].sum()
        total_return_pct = (total_pnl / initial_capital) * 100
        
        avg_win = self.trades[self.trades['pnl_usd'] > 0]['pnl_usd'].mean() if winning_trades > 0 else 0
        avg_loss = self.trades[self.trades['pnl_usd'] < 0]['pnl_usd'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(self.trades[self.trades['pnl_usd'] > 0]['pnl_usd'].sum() / 
                           self.trades[self.trades['pnl_usd'] < 0]['pnl_usd'].sum()) if losing_trades > 0 else 0
        
        # Максимальная просадка
        equity_values = self.equity_curve['equity'].values
        peak = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (упрощенный)
        returns = self.equity_curve['equity'].pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Вывод метрик
        print(f"\n💼 Капитал:")
        print(f"  Начальный: ${initial_capital:,.2f}")
        print(f"  Конечный:  ${self.equity_curve['equity'].iloc[-1]:,.2f}")
        print(f"  P&L:       ${total_pnl:+,.2f} ({total_return_pct:+.2f}%)")
        
        print(f"\n📈 Сделки:")
        print(f"  Всего:     {total_trades}")
        print(f"  Прибыльн.: {winning_trades} ({win_rate:.1f}%)")
        print(f"  Убыточн.:  {losing_trades}")
        
        print(f"\n💰 Средние значения:")
        print(f"  Средняя прибыль: ${avg_win:+,.2f}")
        print(f"  Средний убыток:  ${avg_loss:+,.2f}")
        print(f"  Профит-фактор:   {profit_factor:.2f}")
        
        print(f"\n📉 Риск:")
        print(f"  Макс. просадка:  {max_drawdown:.2f}%")
        print(f"  Sharpe Ratio:    {sharpe_ratio:.2f}")
        
        # Сохранение метрик
        self.metrics = {
            'initial_capital': initial_capital,
            'final_capital': self.equity_curve['equity'].iloc[-1],
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    
    def save_results(self, output_dir='/home/ubuntu/eurusd_analysis/results'):
        """Сохранение результатов бэктеста"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохранение сделок
        if len(self.trades) > 0:
            self.trades.to_csv(f'{output_dir}/backtest_trades.csv', index=False)
            print(f"\n💾 Сохранено: backtest_trades.csv")
        
        # Сохранение equity curve
        self.equity_curve.to_csv(f'{output_dir}/backtest_equity.csv', index=False)
        print(f"💾 Сохранено: backtest_equity.csv")
        
        # Сохранение метрик
        metrics_df = pd.DataFrame([self.metrics])
        metrics_df.to_csv(f'{output_dir}/backtest_metrics.csv', index=False)
        print(f"💾 Сохранено: backtest_metrics.csv")


def main():
    """Главная функция"""
    print("\n" + "="*70)
    print("🚀 ТОРГОВАЯ СТРАТЕГИЯ И БЭКТЕСТИНГ")
    print("="*70)
    
    # Загрузка данных
    features_df = pd.read_csv('/home/ubuntu/eurusd_analysis/data/features.csv',
                             index_col=0, parse_dates=True)
    prices_df = pd.read_csv('/home/ubuntu/eurusd_analysis/data/master_data.csv',
                           index_col=0, parse_dates=True)
    
    print(f"\n📊 Загружено: {len(features_df)} примеров")
    
    # Создание стратегии
    strategy = TradingStrategy(features_df, prices_df)
    
    # Генерация сигналов
    signals = strategy.generate_signals()
    
    # Бэктестинг
    trades, equity = strategy.backtest(
        initial_capital=10000,
        risk_per_trade=0.01,  # 1% риска на сделку
        spread_pips=2
    )
    
    # Сохранение результатов
    strategy.save_results()
    
    print("\n" + "="*70)
    print("✅ БЭКТЕСТИНГ ЗАВЕРШЕН")
    print("="*70)
    
    return strategy


if __name__ == "__main__":
    strategy = main()
