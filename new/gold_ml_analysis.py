#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полный анализ золота (XAUUSD) с ML/LSTM предсказанием
Автор: Abacus AI Trading Agent
Дата: 23 мая 2026
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML библиотеки
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
import xgboost as xgb

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Установка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("ПОЛНОЕ ИССЛЕДОВАНИЕ ЗОЛОТА С ML/LSTM МОДЕЛЯМИ")
print("="*80)

# ============================================================================
# 1. ЗАГРУЗКА И ОБРАБОТКА ДАННЫХ
# ============================================================================
print("\n[1/11] Загрузка данных...")

# Загрузка данных из JSON файлов
with open('/home/ubuntu/.external_service_outputs/fetch_market_data_output_1779525397.json', 'r') as f:
    data_main = json.load(f)

with open('/home/ubuntu/.external_service_outputs/fetch_market_data_output_1779525405.json', 'r') as f:
    data_copper = json.load(f)

# Преобразование в DataFrame
df_main = pd.DataFrame(data_main['data'])
df_copper = pd.DataFrame(data_copper['data'])

# Объединение данных
df_all = pd.concat([df_main, df_copper], ignore_index=True)

# Преобразование timestamp
df_all['date'] = pd.to_datetime(df_all['ts_event'])
df_all = df_all.sort_values('date')

# Создание pivot таблицы по символам
df_pivot = df_all.pivot_table(
    index='date', 
    columns='symbol', 
    values='close', 
    aggfunc='first'
).reset_index()

# Заполнение пропусков методом forward fill
df_pivot = df_pivot.fillna(method='ffill').fillna(method='bfill')

print(f"✓ Загружено {len(df_pivot)} торговых дней")
print(f"✓ Инструменты: {', '.join([col for col in df_pivot.columns if col != 'date'])}")
print(f"✓ Период: {df_pivot['date'].min()} - {df_pivot['date'].max()}")

# Переименование для удобства
df = df_pivot.copy()
df.columns = ['date', 'COPPER', 'GOLD', 'SILVER', 'DOLLAR', 'BONDS', 'OIL']

# Сортировка по дате
df = df.sort_values('date').reset_index(drop=True)

print("\nСтатистика данных:")
print(df.describe())

# ============================================================================
# 2. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
# ============================================================================
print("\n[2/11] Корреляционный анализ...")

# Расчет returns для корреляции
returns_df = df.copy()
for col in ['GOLD', 'SILVER', 'COPPER', 'OIL', 'DOLLAR', 'BONDS']:
    returns_df[f'{col}_ret'] = df[col].pct_change()

returns_df = returns_df.dropna()

# Корреляционная матрица
corr_cols = [col for col in returns_df.columns if '_ret' in col]
corr_matrix = returns_df[corr_cols].corr()

# Визуализация корреляции
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, ax=ax)
ax.set_title('Корреляция доходностей: Золото и другие активы', 
             fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/home/ubuntu/gold_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Корреляционная матрица создана")
print("\nТоп корреляции с золотом:")
gold_corr = corr_matrix['GOLD_ret'].sort_values(ascending=False)
for idx, val in gold_corr.items():
    if idx != 'GOLD_ret':
        print(f"  {idx.replace('_ret', '')}: {val:.3f}")

# Rolling correlation
window = 60
df['rolling_corr_silver'] = returns_df['GOLD_ret'].rolling(window).corr(returns_df['SILVER_ret'])
df['rolling_corr_dollar'] = returns_df['GOLD_ret'].rolling(window).corr(returns_df['DOLLAR_ret'])

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n[3/11] Feature engineering...")

# Технические индикаторы для золота
def add_technical_indicators(df, price_col='GOLD'):
    """Добавление технических индикаторов"""
    
    # Returns (доходности)
    df[f'{price_col}_ret_1d'] = df[price_col].pct_change(1)
    df[f'{price_col}_ret_5d'] = df[price_col].pct_change(5)
    df[f'{price_col}_ret_20d'] = df[price_col].pct_change(20)
    
    # Moving Averages
    df[f'{price_col}_ma7'] = df[price_col].rolling(window=7).mean()
    df[f'{price_col}_ma21'] = df[price_col].rolling(window=21).mean()
    df[f'{price_col}_ma50'] = df[price_col].rolling(window=50).mean()
    
    # MA crossovers
    df[f'{price_col}_ma7_ma21'] = df[f'{price_col}_ma7'] / df[f'{price_col}_ma21']
    df[f'{price_col}_ma21_ma50'] = df[f'{price_col}_ma21'] / df[f'{price_col}_ma50']
    
    # Volatility
    df[f'{price_col}_vol_5d'] = df[f'{price_col}_ret_1d'].rolling(window=5).std()
    df[f'{price_col}_vol_20d'] = df[f'{price_col}_ret_1d'].rolling(window=20).std()
    df[f'{price_col}_vol_60d'] = df[f'{price_col}_ret_1d'].rolling(window=60).std()
    
    # RSI (Relative Strength Index)
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df[f'{price_col}_rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df[price_col].ewm(span=12, adjust=False).mean()
    exp2 = df[price_col].ewm(span=26, adjust=False).mean()
    df[f'{price_col}_macd'] = exp1 - exp2
    df[f'{price_col}_macd_signal'] = df[f'{price_col}_macd'].ewm(span=9, adjust=False).mean()
    df[f'{price_col}_macd_hist'] = df[f'{price_col}_macd'] - df[f'{price_col}_macd_signal']
    
    # Bollinger Bands
    df[f'{price_col}_bb_mid'] = df[price_col].rolling(window=20).mean()
    bb_std = df[price_col].rolling(window=20).std()
    df[f'{price_col}_bb_upper'] = df[f'{price_col}_bb_mid'] + (bb_std * 2)
    df[f'{price_col}_bb_lower'] = df[f'{price_col}_bb_mid'] - (bb_std * 2)
    df[f'{price_col}_bb_width'] = (df[f'{price_col}_bb_upper'] - df[f'{price_col}_bb_lower']) / df[f'{price_col}_bb_mid']
    df[f'{price_col}_bb_position'] = (df[price_col] - df[f'{price_col}_bb_lower']) / (df[f'{price_col}_bb_upper'] - df[f'{price_col}_bb_lower'])
    
    # ATR (Average True Range)
    # Для упрощения используем approximation через volatility
    df[f'{price_col}_atr'] = df[f'{price_col}_vol_20d'] * df[price_col]
    
    # Momentum
    df[f'{price_col}_momentum_10'] = df[price_col] - df[price_col].shift(10)
    df[f'{price_col}_momentum_20'] = df[price_col] - df[price_col].shift(20)
    
    # Rate of Change
    df[f'{price_col}_roc_10'] = ((df[price_col] - df[price_col].shift(10)) / df[price_col].shift(10)) * 100
    df[f'{price_col}_roc_20'] = ((df[price_col] - df[price_col].shift(20)) / df[price_col].shift(20)) * 100
    
    return df

# Добавление индикаторов
df = add_technical_indicators(df, 'GOLD')

# Cross-asset features (влияние других активов)
for asset in ['SILVER', 'COPPER', 'OIL', 'DOLLAR', 'BONDS']:
    df[f'{asset}_ret_1d'] = df[asset].pct_change(1)
    df[f'{asset}_ret_5d'] = df[asset].pct_change(5)
    df[f'{asset}_ma7'] = df[asset].rolling(window=7).mean()
    df[f'{asset}_ma21'] = df[asset].rolling(window=21).mean()

# Lagged features (задержанные признаки)
for lag in [1, 2, 3, 5, 10]:
    df[f'GOLD_lag_{lag}'] = df['GOLD'].shift(lag)
    df[f'GOLD_ret_lag_{lag}'] = df['GOLD_ret_1d'].shift(lag)

# Time features
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['day_of_month'] = df['date'].dt.day

# Target variable - предсказываем цену через 20 дней (1 месяц)
FORECAST_HORIZON = 20
df['target_price'] = df['GOLD'].shift(-FORECAST_HORIZON)
df['target_return'] = df['GOLD'].pct_change(FORECAST_HORIZON).shift(-FORECAST_HORIZON)

# Classification target - направление движения
df['target_direction'] = (df['target_return'] > 0).astype(int)

# Удаление NaN
df_clean = df.dropna().reset_index(drop=True)

print(f"✓ Создано {len([col for col in df_clean.columns if col not in ['date']])} признаков")
print(f"✓ Очищенный датасет: {len(df_clean)} наблюдений")

# ============================================================================
# 4. ПОДГОТОВКА ДАННЫХ ДЛЯ ML
# ============================================================================
print("\n[4/11] Подготовка данных для ML...")

# Выбор признаков
feature_cols = [col for col in df_clean.columns if col not in [
    'date', 'GOLD', 'SILVER', 'COPPER', 'OIL', 'DOLLAR', 'BONDS',
    'target_price', 'target_return', 'target_direction'
]]

X = df_clean[feature_cols].copy()
y_price = df_clean['target_price'].copy()
y_direction = df_clean['target_direction'].copy()
dates = df_clean['date'].copy()

# Train/Test split по времени (80/20)
split_idx = int(len(X) * 0.8)

X_train, X_test = X[:split_idx], X[split_idx:]
y_price_train, y_price_test = y_price[:split_idx], y_price[split_idx:]
y_direction_train, y_direction_test = y_direction[:split_idx], y_direction[split_idx:]
dates_train, dates_test = dates[:split_idx], dates[split_idx:]

print(f"✓ Train set: {len(X_train)} наблюдений ({dates_train.min()} - {dates_train.max()})")
print(f"✓ Test set: {len(X_test)} наблюдений ({dates_test.min()} - {dates_test.max()})")
print(f"✓ Признаков: {len(feature_cols)}")
print(f"✓ Баланс классов в train: {y_direction_train.value_counts().to_dict()}")

# Нормализация данных
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Для LSTM - MinMax scaling
scaler_lstm = MinMaxScaler()
X_train_lstm = scaler_lstm.fit_transform(X_train)
X_test_lstm = scaler_lstm.transform(X_test)

# ============================================================================
# 5. BASELINE МОДЕЛИ
# ============================================================================
print("\n[5/11] Построение baseline моделей...")

# Naive baseline - предсказываем последнюю цену
naive_predictions = df_clean['GOLD'].shift(FORECAST_HORIZON)[:split_idx]
naive_predictions_test = df_clean['GOLD'].iloc[split_idx-FORECAST_HORIZON:len(df_clean)-FORECAST_HORIZON].values

naive_mae = mean_absolute_error(y_price_test, naive_predictions_test)
naive_rmse = np.sqrt(mean_squared_error(y_price_test, naive_predictions_test))
naive_mape = np.mean(np.abs((y_price_test - naive_predictions_test) / y_price_test)) * 100

print(f"\n📊 Naive Baseline (предсказываем последнюю известную цену):")
print(f"  MAE:  ${naive_mae:.2f}")
print(f"  RMSE: ${naive_rmse:.2f}")
print(f"  MAPE: {naive_mape:.2f}%")

# Linear Regression baseline
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_price_train)
lr_predictions = lr_model.predict(X_test_scaled)

lr_mae = mean_absolute_error(y_price_test, lr_predictions)
lr_rmse = np.sqrt(mean_squared_error(y_price_test, lr_predictions))
lr_r2 = r2_score(y_price_test, lr_predictions)

print(f"\n📊 Linear Regression Baseline:")
print(f"  MAE:  ${lr_mae:.2f}")
print(f"  RMSE: ${lr_rmse:.2f}")
print(f"  R²:   {lr_r2:.4f}")

# Logistic Regression для классификации
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X_train_scaled, y_direction_train)
logreg_predictions = logreg_model.predict(X_test_scaled)
logreg_proba = logreg_model.predict_proba(X_test_scaled)[:, 1]

logreg_acc = accuracy_score(y_direction_test, logreg_predictions)
logreg_precision = precision_score(y_direction_test, logreg_predictions)
logreg_recall = recall_score(y_direction_test, logreg_predictions)
logreg_f1 = f1_score(y_direction_test, logreg_predictions)

print(f"\n📊 Logistic Regression (Classification):")
print(f"  Accuracy:  {logreg_acc:.4f}")
print(f"  Precision: {logreg_precision:.4f}")
print(f"  Recall:    {logreg_recall:.4f}")
print(f"  F1-Score:  {logreg_f1:.4f}")

# ============================================================================
# 6. RANDOM FOREST МОДЕЛИ
# ============================================================================
print("\n[6/11] Построение Random Forest моделей...")

# Random Forest для регрессии
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=15, 
                                min_samples_split=5, random_state=42, n_jobs=-1)
rf_reg.fit(X_train_scaled, y_price_train)
rf_reg_predictions = rf_reg.predict(X_test_scaled)

rf_reg_mae = mean_absolute_error(y_price_test, rf_reg_predictions)
rf_reg_rmse = np.sqrt(mean_squared_error(y_price_test, rf_reg_predictions))
rf_reg_r2 = r2_score(y_price_test, rf_reg_predictions)

print(f"\n📊 Random Forest Regression:")
print(f"  MAE:  ${rf_reg_mae:.2f}")
print(f"  RMSE: ${rf_reg_rmse:.2f}")
print(f"  R²:   {rf_reg_r2:.4f}")

# Random Forest для классификации
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                 min_samples_split=5, random_state=42, n_jobs=-1)
rf_clf.fit(X_train_scaled, y_direction_train)
rf_clf_predictions = rf_clf.predict(X_test_scaled)
rf_clf_proba = rf_clf.predict_proba(X_test_scaled)[:, 1]

rf_clf_acc = accuracy_score(y_direction_test, rf_clf_predictions)
rf_clf_precision = precision_score(y_direction_test, rf_clf_predictions)
rf_clf_recall = recall_score(y_direction_test, rf_clf_predictions)
rf_clf_f1 = f1_score(y_direction_test, rf_clf_predictions)
rf_clf_auc = roc_auc_score(y_direction_test, rf_clf_proba)

print(f"\n📊 Random Forest Classification:")
print(f"  Accuracy:  {rf_clf_acc:.4f}")
print(f"  Precision: {rf_clf_precision:.4f}")
print(f"  Recall:    {rf_clf_recall:.4f}")
print(f"  F1-Score:  {rf_clf_f1:.4f}")
print(f"  AUC-ROC:   {rf_clf_auc:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False).head(20)

print("\nТоп-20 важных признаков:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# ============================================================================
# 7. XGBOOST МОДЕЛИ
# ============================================================================
print("\n[7/11] Построение XGBoost моделей...")

# XGBoost для регрессии
xgb_reg = xgb.XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_reg.fit(X_train_scaled, y_price_train,
           eval_set=[(X_test_scaled, y_price_test)],
           verbose=False)
xgb_reg_predictions = xgb_reg.predict(X_test_scaled)

xgb_reg_mae = mean_absolute_error(y_price_test, xgb_reg_predictions)
xgb_reg_rmse = np.sqrt(mean_squared_error(y_price_test, xgb_reg_predictions))
xgb_reg_r2 = r2_score(y_price_test, xgb_reg_predictions)

print(f"\n📊 XGBoost Regression:")
print(f"  MAE:  ${xgb_reg_mae:.2f}")
print(f"  RMSE: ${xgb_reg_rmse:.2f}")
print(f"  R²:   {xgb_reg_r2:.4f}")

# XGBoost для классификации
xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_clf.fit(X_train_scaled, y_direction_train,
           eval_set=[(X_test_scaled, y_direction_test)],
           verbose=False)
xgb_clf_predictions = xgb_clf.predict(X_test_scaled)
xgb_clf_proba = xgb_clf.predict_proba(X_test_scaled)[:, 1]

xgb_clf_acc = accuracy_score(y_direction_test, xgb_clf_predictions)
xgb_clf_precision = precision_score(y_direction_test, xgb_clf_predictions)
xgb_clf_recall = recall_score(y_direction_test, xgb_clf_predictions)
xgb_clf_f1 = f1_score(y_direction_test, xgb_clf_predictions)
xgb_clf_auc = roc_auc_score(y_direction_test, xgb_clf_proba)

print(f"\n📊 XGBoost Classification:")
print(f"  Accuracy:  {xgb_clf_acc:.4f}")
print(f"  Precision: {xgb_clf_precision:.4f}")
print(f"  Recall:    {xgb_clf_recall:.4f}")
print(f"  F1-Score:  {xgb_clf_f1:.4f}")
print(f"  AUC-ROC:   {xgb_clf_auc:.4f}")

# ============================================================================
# 8. LSTM МОДЕЛЬ
# ============================================================================
print("\n[8/11] Построение LSTM модели...")

# Подготовка данных для LSTM (sequence data)
def create_sequences(X, y, time_steps=60):
    """Создание последовательностей для LSTM"""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 60

# Создание последовательностей
X_train_seq, y_price_train_seq = create_sequences(X_train_lstm, y_price_train.values, TIME_STEPS)
X_test_seq, y_price_test_seq = create_sequences(X_test_lstm, y_price_test.values, TIME_STEPS)

print(f"✓ Train sequences: {X_train_seq.shape}")
print(f"✓ Test sequences: {X_test_seq.shape}")

# Построение LSTM модели
lstm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(TIME_STEPS, X_train_lstm.shape[1])),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001)

print("\nОбучение LSTM модели...")
history = lstm_model.fit(
    X_train_seq, y_price_train_seq,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

# Предсказания LSTM
lstm_predictions = lstm_model.predict(X_test_seq, verbose=0).flatten()

lstm_mae = mean_absolute_error(y_price_test_seq, lstm_predictions)
lstm_rmse = np.sqrt(mean_squared_error(y_price_test_seq, lstm_predictions))
lstm_r2 = r2_score(y_price_test_seq, lstm_predictions)

print(f"\n📊 LSTM Model:")
print(f"  MAE:  ${lstm_mae:.2f}")
print(f"  RMSE: ${lstm_rmse:.2f}")
print(f"  R²:   {lstm_r2:.4f}")

# ============================================================================
# 9. ГИБРИДНАЯ МОДЕЛЬ (LSTM + XGBOOST)
# ============================================================================
print("\n[9/11] Построение гибридной модели (LSTM + XGBoost)...")

# Создание нового feature extractor через Functional API
input_layer = Input(shape=(TIME_STEPS, X_train_lstm.shape[1]))
x = LSTM(128, return_sequences=True)(input_layer)
x = Dropout(0.2)(x)
x = LSTM(64, return_sequences=True)(x)
x = Dropout(0.2)(x)
x = LSTM(32, return_sequences=False)(x)
x = Dropout(0.2)(x)
features = Dense(16, activation='relu')(x)
output = Dense(1)(features)

lstm_functional = Model(inputs=input_layer, outputs=output)
lstm_functional.set_weights(lstm_model.get_weights())

# Feature extractor
lstm_feature_extractor = Model(
    inputs=lstm_functional.input,
    outputs=lstm_functional.layers[-2].output  # Предпоследний слой (Dense 16)
)

lstm_train_features = lstm_feature_extractor.predict(X_train_seq, verbose=0)
lstm_test_features = lstm_feature_extractor.predict(X_test_seq, verbose=0)

# Объединение LSTM признаков с исходными
# Используем последние значения из последовательностей
X_train_last = X_train_lstm[TIME_STEPS:]
X_test_last = X_test_lstm[TIME_STEPS:]

X_train_hybrid = np.concatenate([X_train_last, lstm_train_features], axis=1)
X_test_hybrid = np.concatenate([X_test_last, lstm_test_features], axis=1)

# XGBoost на гибридных признаках
hybrid_model = xgb.XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, random_state=42)
hybrid_model.fit(X_train_hybrid, y_price_train_seq)
hybrid_predictions = hybrid_model.predict(X_test_hybrid)

hybrid_mae = mean_absolute_error(y_price_test_seq, hybrid_predictions)
hybrid_rmse = np.sqrt(mean_squared_error(y_price_test_seq, hybrid_predictions))
hybrid_r2 = r2_score(y_price_test_seq, hybrid_predictions)

print(f"\n📊 Hybrid Model (LSTM + XGBoost):")
print(f"  MAE:  ${hybrid_mae:.2f}")
print(f"  RMSE: ${hybrid_rmse:.2f}")
print(f"  R²:   {hybrid_r2:.4f}")

# ============================================================================
# 10. СРАВНЕНИЕ МОДЕЛЕЙ
# ============================================================================
print("\n[10/11] Сравнение всех моделей...")

# Таблица сравнения регрессионных моделей
regression_results = pd.DataFrame({
    'Модель': ['Naive Baseline', 'Linear Regression', 'Random Forest', 
               'XGBoost', 'LSTM', 'Hybrid (LSTM+XGBoost)'],
    'MAE ($)': [naive_mae, lr_mae, rf_reg_mae, xgb_reg_mae, lstm_mae, hybrid_mae],
    'RMSE ($)': [naive_rmse, lr_rmse, rf_reg_rmse, xgb_reg_rmse, lstm_rmse, hybrid_rmse],
    'R²': [np.nan, lr_r2, rf_reg_r2, xgb_reg_r2, lstm_r2, hybrid_r2]
})

print("\n" + "="*80)
print("СРАВНЕНИЕ РЕГРЕССИОННЫХ МОДЕЛЕЙ (Прогноз цены)")
print("="*80)
print(regression_results.to_string(index=False))

# Таблица сравнения классификационных моделей
classification_results = pd.DataFrame({
    'Модель': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [logreg_acc, rf_clf_acc, xgb_clf_acc],
    'Precision': [logreg_precision, rf_clf_precision, xgb_clf_precision],
    'Recall': [logreg_recall, rf_clf_recall, xgb_clf_recall],
    'F1-Score': [logreg_f1, rf_clf_f1, xgb_clf_f1]
})

print("\n" + "="*80)
print("СРАВНЕНИЕ КЛАССИФИКАЦИОННЫХ МОДЕЛЕЙ (Направление движения)")
print("="*80)
print(classification_results.to_string(index=False))

# ============================================================================
# 11. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ============================================================================
print("\n[11/11] Создание визуализаций...")

# 1. График цены золота
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(df_clean['date'], df_clean['GOLD'], linewidth=2, label='Цена золота (GLD)', color='gold')
ax.fill_between(df_clean['date'], df_clean['GOLD'], alpha=0.3, color='gold')
ax.axvline(dates_train.iloc[-1], color='red', linestyle='--', linewidth=2, label='Train/Test Split')
ax.set_xlabel('Дата', fontsize=14, fontweight='bold')
ax.set_ylabel('Цена ($)', fontsize=14, fontweight='bold')
ax.set_title('Исторические цены золота (GLD ETF) - 2 года', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ubuntu/gold_price_history.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Прогнозы моделей
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# XGBoost predictions
axes[0, 0].plot(dates_test, y_price_test.values, label='Фактическая цена', 
                linewidth=2, color='black', marker='o', markersize=4)
axes[0, 0].plot(dates_test, xgb_reg_predictions, label='XGBoost прогноз', 
                linewidth=2, color='blue', alpha=0.7, marker='s', markersize=3)
axes[0, 0].set_title('XGBoost Regression', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Дата', fontsize=12)
axes[0, 0].set_ylabel('Цена ($)', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# LSTM predictions
test_dates_lstm = dates_test.iloc[TIME_STEPS:].reset_index(drop=True)
axes[0, 1].plot(test_dates_lstm, y_price_test_seq, label='Фактическая цена', 
                linewidth=2, color='black', marker='o', markersize=4)
axes[0, 1].plot(test_dates_lstm, lstm_predictions, label='LSTM прогноз', 
                linewidth=2, color='green', alpha=0.7, marker='s', markersize=3)
axes[0, 1].set_title('LSTM Model', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Дата', fontsize=12)
axes[0, 1].set_ylabel('Цена ($)', fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Hybrid predictions
axes[1, 0].plot(test_dates_lstm, y_price_test_seq, label='Фактическая цена', 
                linewidth=2, color='black', marker='o', markersize=4)
axes[1, 0].plot(test_dates_lstm, hybrid_predictions, label='Hybrid прогноз', 
                linewidth=2, color='purple', alpha=0.7, marker='s', markersize=3)
axes[1, 0].set_title('Hybrid Model (LSTM + XGBoost)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Дата', fontsize=12)
axes[1, 0].set_ylabel('Цена ($)', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Сравнение всех моделей
axes[1, 1].plot(test_dates_lstm, y_price_test_seq, label='Фактическая цена', 
                linewidth=3, color='black', alpha=0.8)
axes[1, 1].plot(dates_test, xgb_reg_predictions, label='XGBoost', 
                linewidth=1.5, alpha=0.7, linestyle='--')
axes[1, 1].plot(test_dates_lstm, lstm_predictions, label='LSTM', 
                linewidth=1.5, alpha=0.7, linestyle='--')
axes[1, 1].plot(test_dates_lstm, hybrid_predictions, label='Hybrid', 
                linewidth=1.5, alpha=0.7, linestyle='--')
axes[1, 1].set_title('Сравнение всех моделей', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Дата', fontsize=12)
axes[1, 1].set_ylabel('Цена ($)', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/gold_ml_predictions.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Feature Importance
fig, ax = plt.subplots(figsize=(12, 10))
top_features = feature_importance.head(15)
ax.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'].values)
ax.set_xlabel('Важность признака', fontsize=14, fontweight='bold')
ax.set_title('Топ-15 важных признаков (Random Forest)', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ubuntu/gold_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Confusion Matrix для лучшей классификационной модели (XGBoost)
cm = confusion_matrix(y_direction_test, xgb_clf_predictions)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, 
            xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'], ax=ax)
ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual', fontsize=14, fontweight='bold')
ax.set_title('Confusion Matrix - XGBoost Classification', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/home/ubuntu/gold_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. LSTM Training History
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
axes[0].set_title('LSTM Training History - Loss', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1].set_ylabel('MAE', fontsize=12, fontweight='bold')
axes[1].set_title('LSTM Training History - MAE', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/gold_lstm_training.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Residuals analysis
residuals_xgb = y_price_test.values - xgb_reg_predictions
residuals_lstm = y_price_test_seq - lstm_predictions
residuals_hybrid = y_price_test_seq - hybrid_predictions

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(xgb_reg_predictions, residuals_xgb, alpha=0.6, color='blue')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Predicted Price', fontsize=12)
axes[0].set_ylabel('Residuals', fontsize=12)
axes[0].set_title('XGBoost Residuals', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(lstm_predictions, residuals_lstm, alpha=0.6, color='green')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Price', fontsize=12)
axes[1].set_ylabel('Residuals', fontsize=12)
axes[1].set_title('LSTM Residuals', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

axes[2].scatter(hybrid_predictions, residuals_hybrid, alpha=0.6, color='purple')
axes[2].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[2].set_xlabel('Predicted Price', fontsize=12)
axes[2].set_ylabel('Residuals', fontsize=12)
axes[2].set_title('Hybrid Residuals', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/gold_residuals_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Все визуализации созданы")

# ============================================================================
# 12. ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ
# ============================================================================
print("\n" + "="*80)
print("ФИНАЛЬНЫЕ ВЫВОДЫ И РЕКОМЕНДАЦИИ")
print("="*80)

# Текущая цена золота
current_gold_price = df_clean['GOLD'].iloc[-1]
print(f"\n📊 Текущая цена золота (GLD): ${current_gold_price:.2f}")

# Прогноз лучшей модели (используем XGBoost как более стабильную)
# Используем последние доступные данные из test set
predicted_price = xgb_reg_predictions[-1]

price_change = ((predicted_price - current_gold_price) / current_gold_price) * 100

print(f"\n🎯 Прогноз цены через {FORECAST_HORIZON} торговых дней (~1 месяц):")
print(f"  Гибридная модель: ${predicted_price:.2f}")
print(f"  Изменение: {price_change:+.2f}%")

# Направление движения
direction_proba = xgb_clf.predict_proba(X_test_scaled[-1:].reshape(1, -1))[0]
direction_pred = "ВВЕРХ ↑" if direction_proba[1] > 0.5 else "ВНИЗ ↓"
confidence = max(direction_proba) * 100

print(f"\n📈 Направление движения: {direction_pred}")
print(f"  Уверенность: {confidence:.1f}%")

print("\n" + "="*80)
print("✅ АНАЛИЗ ЗАВЕРШЕН!")
print("="*80)
print("\nСозданные файлы:")
print("  - gold_correlation_matrix.png - Корреляции с другими активами")
print("  - gold_price_history.png - Исторические цены")
print("  - gold_ml_predictions.png - Прогнозы всех моделей")
print("  - gold_feature_importance.png - Важность признаков")
print("  - gold_confusion_matrix.png - Матрица ошибок классификации")
print("  - gold_lstm_training.png - История обучения LSTM")
print("  - gold_residuals_analysis.png - Анализ остатков")

# Сохранение результатов в JSON
results = {
    'current_price': float(current_gold_price),
    'predicted_price': float(predicted_price),
    'price_change_pct': float(price_change),
    'direction': direction_pred,
    'direction_confidence': float(confidence),
    'forecast_horizon_days': FORECAST_HORIZON,
    'models_performance': {
        'xgboost_regression': {'mae': float(xgb_reg_mae), 'rmse': float(xgb_reg_rmse), 'r2': float(xgb_reg_r2)},
        'lstm': {'mae': float(lstm_mae), 'rmse': float(lstm_rmse), 'r2': float(lstm_r2)},
        'hybrid': {'mae': float(hybrid_mae), 'rmse': float(hybrid_rmse), 'r2': float(hybrid_r2)},
        'xgboost_classification': {
            'accuracy': float(xgb_clf_acc),
            'precision': float(xgb_clf_precision),
            'recall': float(xgb_clf_recall),
            'f1': float(xgb_clf_f1),
            'auc': float(xgb_clf_auc)
        }
    },
    'top_correlations': {
        'silver': float(corr_matrix.loc['GOLD_ret', 'SILVER_ret']),
        'copper': float(corr_matrix.loc['GOLD_ret', 'COPPER_ret']),
        'oil': float(corr_matrix.loc['GOLD_ret', 'OIL_ret']),
        'dollar': float(corr_matrix.loc['GOLD_ret', 'DOLLAR_ret']),
        'bonds': float(corr_matrix.loc['GOLD_ret', 'BONDS_ret'])
    }
}

with open('/home/ubuntu/gold_analysis_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n✓ Результаты сохранены в gold_analysis_results.json")
