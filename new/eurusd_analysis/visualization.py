"""
МОДУЛЬ ВИЗУАЛИЗАЦИИ
Создание всех графиков для анализа
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Настройки
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 9
plt.rcParams['axes.unicode_minus'] = False

# Поддержка русских шрифтов
plt.rcParams['font.family'] = 'DejaVu Sans'

class Visualizer:
    """Класс для создания визуализаций"""
    
    def __init__(self, output_dir='/home/ubuntu/eurusd_analysis/charts'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_to_base64(self, fig):
        """Конвертация графика в base64 для встраивания в отчет"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return img_str
    
    def plot_correlation_heatmap(self, returns_df):
        """Тепловая карта корреляций"""
        print("\n📊 Создание тепловой карты корреляций...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Корреляционная матрица
        corr = returns_df.corr()
        
        # Создаем маску для верхнего треугольника
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Тепловая карта
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
                   cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        # Упрощаем названия
        labels = [col.replace('_Return', '') for col in corr.columns]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)
        
        ax.set_title('KORRELYATSIONNAYA MATRITSA INSTRUMENTOV\n(Dnevnye dohodnosti, 2 goda)', 
                    fontsize=14, weight='bold', pad=20)
        
        plt.tight_layout()
        
        # Сохранение
        filepath = f'{self.output_dir}/01_correlation_heatmap.png'
        fig.savefig(filepath, dpi=100, bbox_inches='tight')
        print(f"  ✅ Сохранено: {filepath}")
        
        img_base64 = self.plot_to_base64(fig)
        return img_base64
    
    def plot_eurusd_correlations_bar(self, eurusd_corr):
        """Столбчатая диаграмма корреляций с EUR/USD"""
        print("\n📊 Создание диаграммы корреляций EUR/USD...")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Удаляем сам EUR/USD из списка
        corr_data = eurusd_corr.drop('EURUSD_Return').sort_values()
        
        # Цвета: положительные - зеленые, отрицательные - красные
        colors = ['#d62728' if x < 0 else '#2ca02c' for x in corr_data]
        
        # График
        bars = ax.barh(range(len(corr_data)), corr_data, color=colors, alpha=0.7)
        
        # Подписи значений
        for i, (val, bar) in enumerate(zip(corr_data, bars)):
            ax.text(val + 0.02 if val > 0 else val - 0.02, i, f'{val:.3f}',
                   va='center', ha='left' if val > 0 else 'right',
                   fontsize=9, weight='bold')
        
        # Упрощаем названия
        labels = [idx.replace('_Return', '') for idx in corr_data.index]
        ax.set_yticks(range(len(corr_data)))
        ax.set_yticklabels(labels)
        
        ax.set_xlabel('Korrelyatsiya', fontsize=11, weight='bold')
        ax.set_title('KORRELYATSII S EUR/USD\n(Staticheskie znacheniya za 2 goda)', 
                    fontsize=13, weight='bold', pad=15)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(-1, 1)
        
        plt.tight_layout()
        
        filepath = f'{self.output_dir}/02_eurusd_correlations.png'
        fig.savefig(filepath, dpi=100, bbox_inches='tight')
        print(f"  ✅ Сохранено: {filepath}")
        
        img_base64 = self.plot_to_base64(fig)
        return img_base64
    
    def plot_rolling_correlations(self, returns_df, window=60):
        """Скользящие корреляции ключевых инструментов"""
        print(f"\n📈 Создание графика скользящих корреляций (окно {window} дней)...")
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Ключевые инструменты для отображения
        instruments = ['DXY', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'XAUUSD']
        
        for idx, instrument in enumerate(instruments):
            col_name = f'{instrument}_Return'
            if col_name in returns_df.columns:
                # Скользящая корреляция
                rolling_corr = returns_df['EURUSD_Return'].rolling(
                    window=window
                ).corr(returns_df[col_name])
                
                ax = axes[idx]
                ax.plot(rolling_corr.index, rolling_corr.values, 
                       linewidth=1.5, label=f'{instrument}', color='#1f77b4')
                ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
                ax.fill_between(rolling_corr.index, rolling_corr.values, 0, 
                               alpha=0.2, color='#1f77b4')
                
                # Статистика
                mean_corr = rolling_corr.mean()
                ax.axhline(mean_corr, color='red', linewidth=1, 
                          linestyle='--', alpha=0.7, label=f'Mean: {mean_corr:.3f}')
                
                ax.set_title(f'{instrument} (Rolling {window}d)', 
                           fontsize=11, weight='bold')
                ax.set_ylabel('Korrelyatsiya', fontsize=9)
                ax.legend(loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-1, 1)
        
        fig.suptitle(f'SKOLZYASCHIE KORRELYATSII S EUR/USD (okno {window} dney)', 
                    fontsize=14, weight='bold', y=0.995)
        plt.tight_layout()
        
        filepath = f'{self.output_dir}/03_rolling_correlations.png'
        fig.savefig(filepath, dpi=100, bbox_inches='tight')
        print(f"  ✅ Сохранено: {filepath}")
        
        img_base64 = self.plot_to_base64(fig)
        return img_base64
    
    def plot_lagged_correlations(self, lag_df):
        """Лаговые корреляции (ведущие индикаторы)"""
        print("\n⏱️  Создание графика лаговых корреляций...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Транспонируем для удобства
        lag_data = lag_df.T
        
        # Выбираем ключевые инструменты
        key_instruments = ['DXY', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD', 'SPX']
        
        for instrument in key_instruments:
            if instrument in lag_data.columns:
                lags = range(len(lag_data))
                values = lag_data[instrument].values
                ax.plot(lags, values, marker='o', linewidth=2, 
                       markersize=6, label=instrument, alpha=0.8)
        
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Lag (dni)', fontsize=11, weight='bold')
        ax.set_ylabel('Korrelyatsiya', fontsize=11, weight='bold')
        ax.set_title('LAGOVYE KORRELYATSII: POISK VEDUSCHIH INDIKATOROV\n' +
                    '(Otritsatelnyy lag = instrument opazdyvaet, Polozhitelnyy = operejaet)',
                    fontsize=13, weight='bold', pad=15)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(6))
        ax.set_ylim(-1, 1)
        
        plt.tight_layout()
        
        filepath = f'{self.output_dir}/04_lagged_correlations.png'
        fig.savefig(filepath, dpi=100, bbox_inches='tight')
        print(f"  ✅ Сохранено: {filepath}")
        
        img_base64 = self.plot_to_base64(fig)
        return img_base64
    
    def plot_scatter_key_correlations(self, returns_df):
        """Scatter plots сильнейших корреляций"""
        print("\n📊 Создание scatter plots ключевых корреляций...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Ключевые пары для scatter
        pairs = [
            ('GBPUSD', +0.80),
            ('USDCHF', -0.79),
            ('AUDUSD', +0.62),
            ('USDJPY', -0.60),
            ('DXY', -0.15),
            ('XAUUSD', +0.07)
        ]
        
        for idx, (instrument, expected_corr) in enumerate(pairs):
            col_name = f'{instrument}_Return'
            if col_name in returns_df.columns:
                ax = axes[idx]
                
                x = returns_df['EURUSD_Return'].dropna()
                y = returns_df[col_name].dropna()
                
                # Выравниваем индексы
                common_idx = x.index.intersection(y.index)
                x = x.loc[common_idx]
                y = y.loc[common_idx]
                
                # Scatter plot
                ax.scatter(x, y, alpha=0.4, s=20, edgecolors='none')
                
                # Линия тренда
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x.sort_values(), p(x.sort_values()), 
                       color='red', linewidth=2, linestyle='--', alpha=0.8)
                
                # Корреляция
                corr = x.corr(y)
                
                ax.set_xlabel('EUR/USD Return (%)', fontsize=9)
                ax.set_ylabel(f'{instrument} Return (%)', fontsize=9)
                ax.set_title(f'{instrument} vs EUR/USD\nCorr = {corr:.3f}', 
                           fontsize=10, weight='bold')
                ax.grid(True, alpha=0.3)
                ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
                ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)
        
        fig.suptitle('SCATTER PLOTS: KLYUCHEVYE KORRELYATSII S EUR/USD', 
                    fontsize=14, weight='bold', y=0.995)
        plt.tight_layout()
        
        filepath = f'{self.output_dir}/05_scatter_correlations.png'
        fig.savefig(filepath, dpi=100, bbox_inches='tight')
        print(f"  ✅ Сохранено: {filepath}")
        
        img_base64 = self.plot_to_base64(fig)
        return img_base64
    
    def plot_price_comparison(self, prices_df):
        """Сравнение движения EUR/USD с коррелирующими инструментами"""
        print("\n📈 Создание графика сравнения цен...")
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Нормализация к 100
        def normalize(series):
            return (series / series.iloc[0]) * 100
        
        # График 1: EUR/USD vs GBP/USD (сильная прямая корреляция)
        ax1 = axes[0]
        ax1.plot(prices_df.index, normalize(prices_df['EURUSD_Close']), 
                label='EUR/USD', linewidth=2, color='#1f77b4')
        ax1.plot(prices_df.index, normalize(prices_df['GBPUSD_Close']), 
                label='GBP/USD', linewidth=2, color='#ff7f0e', alpha=0.8)
        ax1.set_ylabel('Normalizirovannaya tsena (100)', fontsize=10)
        ax1.set_title('EUR/USD vs GBP/USD (Pryamaya korrelyatsiya +0.80)', 
                     fontsize=11, weight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # График 2: EUR/USD vs DXY (обратная корреляция)
        ax2 = axes[1]
        ax2.plot(prices_df.index, normalize(prices_df['EURUSD_Close']), 
                label='EUR/USD', linewidth=2, color='#1f77b4')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(prices_df.index, normalize(prices_df['DXY_Close']), 
                     label='DXY (inverse)', linewidth=2, color='#d62728', alpha=0.8)
        ax2_twin.invert_yaxis()  # Инвертируем для наглядности
        ax2.set_ylabel('EUR/USD (100)', fontsize=10, color='#1f77b4')
        ax2_twin.set_ylabel('DXY inverted (100)', fontsize=10, color='#d62728')
        ax2.set_title('EUR/USD vs DXY (Obratnaya korrelyatsiya -0.15, Lag 1 den: -0.80!)', 
                     fontsize=11, weight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        ax2_twin.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # График 3: EUR/USD vs Золото
        ax3 = axes[2]
        ax3.plot(prices_df.index, normalize(prices_df['EURUSD_Close']), 
                label='EUR/USD', linewidth=2, color='#1f77b4')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(prices_df.index, normalize(prices_df['XAUUSD_Close']), 
                     label='Gold', linewidth=2, color='#ff9800', alpha=0.8)
        ax3.set_xlabel('Data', fontsize=10)
        ax3.set_ylabel('EUR/USD (100)', fontsize=10, color='#1f77b4')
        ax3_twin.set_ylabel('Gold (100)', fontsize=10, color='#ff9800')
        ax3.set_title('EUR/USD vs Zoloto (Slabaya, no veduschiy indikator +1 den)', 
                     fontsize=11, weight='bold')
        ax3.legend(loc='upper left', fontsize=9)
        ax3_twin.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = f'{self.output_dir}/06_price_comparison.png'
        fig.savefig(filepath, dpi=100, bbox_inches='tight')
        print(f"  ✅ Сохранено: {filepath}")
        
        img_base64 = self.plot_to_base64(fig)
        return img_base64


def main():
    """Главная функция для создания всех визуализаций корреляций"""
    print("\n" + "="*70)
    print("🎨 СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ - КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
    print("="*70)
    
    # Загрузка данных
    returns_df = pd.read_csv('/home/ubuntu/eurusd_analysis/data/returns_data.csv',
                            index_col=0, parse_dates=True)
    prices_df = pd.read_csv('/home/ubuntu/eurusd_analysis/data/master_data.csv',
                           index_col=0, parse_dates=True)
    
    # Загрузка результатов корреляционного анализа
    eurusd_corr = pd.read_csv('/home/ubuntu/eurusd_analysis/results/correlation_static.csv',
                             index_col=0).iloc[:, 0]
    lag_df = pd.read_csv('/home/ubuntu/eurusd_analysis/results/correlation_lagged.csv',
                        index_col=0)
    
    # Создаем визуализатор
    viz = Visualizer()
    
    # Создаем все графики
    viz.plot_correlation_heatmap(returns_df)
    viz.plot_eurusd_correlations_bar(eurusd_corr)
    viz.plot_rolling_correlations(returns_df, window=60)
    viz.plot_lagged_correlations(lag_df)
    viz.plot_scatter_key_correlations(returns_df)
    viz.plot_price_comparison(prices_df)
    
    print("\n" + "="*70)
    print("✅ ВСЕ ВИЗУАЛИЗАЦИИ КОРРЕЛЯЦИЙ СОЗДАНЫ")
    print("="*70)


if __name__ == "__main__":
    main()
