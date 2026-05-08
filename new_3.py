# c:\Users\User\Documents\deepseekAI1\deepseekai_v2\models\investigation_simple.py
#!/usr/bin/env python3
"""
Простое исследование модели для бота
- Линейный код для удобного понимания
- Много моделей (XGB, RF, GB, LightGBM, CatBoost, HistGradientBoosting)
- Детальные графики (confusion matrix, feature importance, PR AUC, equity curve)
- Использует ORB-признаки и горизонт 6 баров (как бот)
"""

import os
import json
import pickle
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Установим стиль для графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

warnings.filterwarnings('ignore')

# ============================================================
# 1. Настройки (все в одном месте)
# ============================================================
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X",
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"
]
LOOKBACK_DAYS = 700
INTERVAL = "1h"
LOOKAHEAD_BARS = 6  # совпадает с max_hold_bars в боте
TARGET_THRESHOLD = 0.0015  # 0.15% (немного выше, чтобы сигналы были чище)
TRAIN_SIZE = 0.75  # больше данных для обучения

print("="*70)
print("🔬 ПРОСТОЕ ИССЛЕДОВАНИЕ МОДЕЛИ")
print("="*70)
print(f"Символы: {SYMBOLS}")
print(f"Горизонт прогноза: {LOOKAHEAD_BARS} баров")
print(f"Порог цели: {TARGET_THRESHOLD*100:.2f}%")
print()

# ============================================================
# 2. Загружаем данные и считаем индикаторы
# ============================================================
print("2. Загружаем данные и считаем индикаторы...")
all_data = []

for symbol in SYMBOLS:
    print(f"  Загружаем {symbol}...")
    
    try:
        end = datetime.now()
        start = end - timedelta(days=LOOKBACK_DAYS)
        df = yf.download(symbol, start=start, end=end, interval=INTERVAL, progress=False)
        
        if df.empty:
            print(f"    ❌ Нет данных для {symbol}")
            continue
            
        # Убираем MultiIndex, если есть (yfinance иногда возвращает его)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        df.columns = [c.lower() for c in df.columns]
        df['symbol'] = symbol
        
        # Базовые доходности
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Волатильность (20 баров)
        df['volatility'] = df['returns'].rolling(20).std()
        
        # EMA
        df['ema_8'] = df['close'].ewm(span=8).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        hl = df['high'] - df['low']
        hc = np.abs(df['high'] - df['close'].shift())
        lc = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma20 + (std20 * 2)
        df['bb_lower'] = sma20 - (std20 * 2)
        
        # Лаги
        df['returns_lag_1'] = df['returns'].shift(1)
        df['close_lag_1'] = df['close'].shift(1)
        df['returns_lag_2'] = df['returns'].shift(2)
        df['close_lag_2'] = df['close'].shift(2)
        df['returns_lag_3'] = df['returns'].shift(3)
        df['close_lag_3'] = df['close'].shift(3)
        df['returns_lag_5'] = df['returns'].shift(5)
        df['close_lag_5'] = df['close'].shift(5)
        
        # ORB (24-часовой диапазон)
        df['or_high_24'] = df['high'].rolling(24).max()
        df['or_low_24'] = df['low'].rolling(24).min()
        df['or_range'] = df['or_high_24'] - df['or_low_24']
        df['or_range_pct'] = df['or_range'] / df['close'] * 100
        df['breakout_up'] = (df['close'] > df['or_high_24'].shift(1)).astype(int)
        df['breakout_down'] = (df['close'] < df['or_low_24'].shift(1)).astype(int)
        
        # Час и сессии
        df['hour'] = df.index.hour
        df['is_eu_session'] = ((df['hour'] >= 7) & (df['hour'] <= 15)).astype(int)
        df['is_us_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)
        
        # Целевая переменная: что будет через 6 баров
        df['future_return'] = df['close'].shift(-LOOKAHEAD_BARS) / df['close'] - 1
        df['target'] = np.where(
            df['future_return'] > TARGET_THRESHOLD, 2,   # bullish (растем)
            np.where(df['future_return'] < -TARGET_THRESHOLD, 0, 1)  # bearish (падаем) / flat (не двигаемся)
        )
        
        # Удаляем пропуски
        df = df.dropna()
        all_data.append(df)
        print(f"    ✅ {symbol}: {len(df)} строк")
        
    except Exception as e:
        print(f"    ❌ Ошибка для {symbol}: {e}")

if len(all_data) == 0:
    print("❌ Нет данных для обучения!")
    exit()

data = pd.concat(all_data, axis=0)
print(f"\n✅ Объединённые данные: {len(data)} строк")

# ============================================================
# 3. Подготовка признаков и разбиение на train/test
# ============================================================
print("\n3. Подготовка признаков...")

# Список всех признаков (то, что модель будет видеть)
feature_cols = [
    'returns', 'log_returns', 'volatility',
    'ema_8', 'ema_21', 'ema_50',
    'macd', 'macd_signal', 'macd_hist',
    'rsi', 'atr', 'atr_pct',
    'bb_upper', 'bb_lower',
    'returns_lag_1', 'close_lag_1',
    'returns_lag_2', 'close_lag_2',
    'returns_lag_3', 'close_lag_3',
    'returns_lag_5', 'close_lag_5',
    'or_range_pct', 'breakout_up', 'breakout_down',
    'hour', 'is_eu_session', 'is_us_session'
]

# Разделяем на X (признаки) и y (цель)
X = data[feature_cols]
y = data['target'].astype(int)

# Временное разбиение (не случайное, чтобы модель не смотрела в будущее)
split_idx = int(len(X) * TRAIN_SIZE)
X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

print(f"  Train: {len(X_train)} строк")
print(f"  Test:  {len(X_test)} строк")

# Масштабируем признаки (важно для многих моделей)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. Обучение моделей
# ============================================================
print("\n4. Обучение моделей...")

# Создаём словарь с моделями
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=10,
        min_samples_leaf=2, n_jobs=-1, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),
    "HistGradient Boosting": HistGradientBoostingClassifier(
        max_iter=200, max_depth=8, learning_rate=0.05,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, eval_metric='mlogloss'
    )
}

# Если установлены LightGBM/CatBoost, добавляем их
try:
    from lightgbm import LGBMClassifier
    models["LightGBM"] = LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1
    )
    print("  ✅ Добавлен LightGBM")
except ImportError:
    print("  ⚠️ LightGBM не установлен")

try:
    from catboost import CatBoostClassifier
    models["CatBoost"] = CatBoostClassifier(
        iterations=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bylevel=0.8,
        random_state=42, verbose=0
    )
    print("  ✅ Добавлен CatBoost")
except ImportError:
    print("  ⚠️ CatBoost не установлен")

# Обучение и оценка каждой модели
results = []
trained_models = {}

for name, model in models.items():
    print(f"\n  Обучаем {name}...")
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    # Метрики
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    try:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    except:
        roc_auc = 0.5
        
    # PR AUC для bullish класса
    le = LabelEncoder()
    y_test_bullish = le.fit_transform((y_test == 2).astype(int))
    precision, recall, _ = precision_recall_curve(y_test_bullish, y_proba[:, 2])
    pr_auc = auc(recall, precision)
    
    results.append({
        "model": name,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1,
        "roc_auc": roc_auc,
        "pr_auc_bullish": pr_auc
    })
    trained_models[name] = model
    
    print(f"    Balanced Accuracy: {bal_acc:.4f}")
    print(f"    F1-macro:          {f1:.4f}")
    print(f"    ROC-AUC:           {roc_auc:.4f}")
    print(f"    PR-AUC (bullish):  {pr_auc:.4f}")

# Создаём Voting Ensemble
print("\n  Создаём Voting Ensemble...")
voting = VotingClassifier(
    estimators=[(n, m) for n, m in trained_models.items()],
    voting='soft'
)
voting.fit(X_train_scaled, y_train)
y_pred_voting = voting.predict(X_test_scaled)
y_proba_voting = voting.predict_proba(X_test_scaled)

bal_acc_voting = balanced_accuracy_score(y_test, y_pred_voting)
f1_voting = f1_score(y_test, y_pred_voting, average='macro')
try:
    roc_auc_voting = roc_auc_score(y_test, y_proba_voting, multi_class='ovr')
except:
    roc_auc_voting = 0.5

y_test_bullish_voting = le.fit_transform((y_test == 2).astype(int))
precision_v, recall_v, _ = precision_recall_curve(y_test_bullish_voting, y_proba_voting[:, 2])
pr_auc_voting = auc(recall_v, precision_v)

results.append({
    "model": "Voting Ensemble",
    "balanced_accuracy": bal_acc_voting,
    "f1_macro": f1_voting,
    "roc_auc": roc_auc_voting,
    "pr_auc_bullish": pr_auc_voting
})
trained_models["Voting Ensemble"] = voting

print(f"    Balanced Accuracy: {bal_acc_voting:.4f}")
print(f"    F1-macro:          {f1_voting:.4f}")
print(f"    ROC-AUC:           {roc_auc_voting:.4f}")
print(f"    PR-AUC (bullish):  {pr_auc_voting:.4f}")

# ============================================================
# 5. Выбираем лучшую модель
# ============================================================
print("\n5. Выбираем лучшую модель...")
results_df = pd.DataFrame(results).sort_values("balanced_accuracy", ascending=False)
print("\nРезультаты всех моделей:")
print(results_df.round(4).to_string(index=False))

best_name = results_df.iloc[0]["model"]
best_model = trained_models[best_name]
print(f"\n🏆 Лучшая модель: {best_name}")

# ============================================================
# 6. Графики
# ============================================================
print("\n6. Создаём графики...")

# 1. Сравнение моделей по метрикам
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
metrics = ["balanced_accuracy", "f1_macro", "roc_auc", "pr_auc_bullish"]
metric_names = ["Balanced Accuracy", "F1-macro", "ROC-AUC", "PR-AUC (bullish)"]

for i, (metric, name) in enumerate(zip(metrics, metric_names)):
    ax = axes[i//2, i%2]
    sns.barplot(data=results_df, x="model", y=metric, ax=ax, palette="viridis")
    ax.set_title(f"Сравнение моделей: {name}", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel(name)
    ax.tick_params(axis='x', rotation=45)
    
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# 2. Confusion Matrix для лучшей модели
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)
labels = ["bearish", "flat", "bullish"]

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=labels, yticklabels=labels)
ax.set_title(f"Confusion Matrix: {best_name}", fontsize=14)
ax.set_xlabel("Предсказано")
ax.set_ylabel("Реально")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()

# 3. Feature Importance для лучшей модели
if hasattr(best_model, 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(12, 8))
    imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=True).tail(15)  # топ 15 признаков
    
    ax.barh(imp['feature'], imp['importance'], color=sns.color_palette('viridis', len(imp)))
    ax.set_title(f"Top 15 признаков: {best_name}", fontsize=14)
    ax.set_xlabel("Важность")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ График важности признаков сохранён")

# 4. PR Curve для bullish класса
fig, ax = plt.subplots(figsize=(10, 7))
for name, model in trained_models.items():
    y_proba = model.predict_proba(X_test_scaled)
    precision, recall, _ = precision_recall_curve(y_test_bullish, y_proba[:, 2])
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.4f})")

ax.set_title("Precision-Recall Curve (bullish класс)", fontsize=14)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_pr_curve.png"), dpi=150, bbox_inches="tight")
plt.close()

print("  ✅ Все графики сохранены в папку models/")

# ============================================================
# 7. Сохраняем модель
# ============================================================
print("\n7. Сохраняем модель...")

# Данные для бота
metadata = {
    'feature_columns': feature_cols,
    'model_type': f"{best_name} - Simple Investigation",
    'target_mapping': {0: 'bearish', 1: 'flat', 2: 'bullish'},
    'lookahead_bars': LOOKAHEAD_BARS,
    'target_threshold': TARGET_THRESHOLD,
    'performance': results_df.iloc[0].to_dict(),
    'all_models_performance': results_df.to_dict(orient='records'),
    'training_date': str(datetime.now())
}

# Сохраняем
with open(os.path.join(OUTPUT_DIR, "model_best.pkl"), 'wb') as f:
    pickle.dump(best_model, f)
with open(os.path.join(OUTPUT_DIR, "scaler_best.pkl"), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(OUTPUT_DIR, "metadata_best.json"), 'w') as f:
    json.dump(metadata, f, indent=2)

# Сохраняем результаты в CSV
results_df.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)

print("  ✅ Модель сохранена как model_best.pkl")
print("  ✅ Масштабировщик сохранён как scaler_best.pkl")
print("  ✅ Метаданные сохранены как metadata_best.json")

print("\n" + "="*70)
print("✅ ИССЛЕДОВАНИЕ ЗАВЕРШЕНО!")
print("="*70)
print("\nЧто дальше:")
print("1. Посмотри графики в папке models/")
print("2. Если модель хорошая, обнови bot.py, чтобы она использовала model_best.pkl")
print("3. Включи 'Use ML regime filter' в дашборде и протестируй")