# =====================================================
# 1. ИМПОРТ ВСЕХ БИБЛИОТЕК
# =====================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM,
        Dense,
        Dropout,
        Input,
        MultiHeadAttention,
        LayerNormalization,
        GlobalAveragePooling1D,
        Bidirectional,
        GRU,
        Conv1D,
        MaxPooling1D,
        Flatten,
        SimpleRNN,
        BatchNormalization,
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
except Exception:
    tf = None
    Sequential = None
    Model = None
    LSTM = Dense = Dropout = Input = MultiHeadAttention = LayerNormalization = GlobalAveragePooling1D = None
    Bidirectional = GRU = Conv1D = MaxPooling1D = Flatten = SimpleRNN = BatchNormalization = None
    Adam = None
    EarlyStopping = ReduceLROnPlateau = None

# Данные
import yfinance as yf

if os.getenv("EXPORT_SWITCH_MODEL", "false").strip().lower() in ("1", "true", "yes"):
    import json
    import joblib
    import bot as bot_module

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    symbol = os.getenv("SWITCH_SYMBOL", "EURUSD=X")
    interval = os.getenv("SWITCH_INTERVAL", "1h")
    period = os.getenv("SWITCH_PERIOD", "365d")
    label_horizon = int(os.getenv("SWITCH_LABEL_HORIZON", 12))
    label_thr_pct = float(os.getenv("SWITCH_LABEL_THR_PCT", 0.20)) / 100.0
    lookahead = int(os.getenv("SWITCH_LOOKAHEAD", 6))

    out_dir = os.getenv("SWITCH_OUT_DIR", "models")
    os.makedirs(out_dir, exist_ok=True)

    raw = yf.Ticker(symbol).history(period=period, interval=interval)
    if raw is None or raw.empty:
        raise RuntimeError("No data")

    raw = raw[["Open", "High", "Low", "Close", "Volume"]]
    raw.columns = ["open", "high", "low", "close", "volume"]

    df = bot_module.calculate_indicators(raw).dropna().copy()

    close = pd.to_numeric(df["close"], errors="coerce")
    fwd = (close.shift(-label_horizon) / close) - 1.0
    label = np.where(fwd > label_thr_pct, 2, np.where(fwd < -label_thr_pct, 0, 1))
    label = pd.Series(label, index=df.index)

    sw = (label.ne(label.shift(1))).astype(int)
    sw.iloc[0] = 0

    future_sw = pd.concat([sw.shift(-i) for i in range(1, lookahead + 1)], axis=1)
    y = future_sw.max(axis=1)

    feats = [
        "returns", "log_returns", "volatility", "ema_8", "ema_21", "ema_50",
        "macd", "macd_signal", "macd_hist", "rsi", "atr", "bb_upper", "bb_lower",
        "close_kalman", "close_kalman_std", "ema_cross",
        "jaw", "teeth", "lips",
        "alligator_asleep", "alligator_bullish", "alligator_bearish", "alligator_expanding",
        "fractal_bullish", "fractal_bearish",
        "bullish_fractal_alligator", "bearish_fractal_alligator",
        "strong_bullish", "strong_bearish",
        "returns_lag_1", "returns_lag_2", "returns_lag_3", "returns_lag_5",
        "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_5",
    ]
    feats = [c for c in feats if c in df.columns]

    x = df[feats].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y = y.loc[x.index].fillna(0).astype(int)

    n = len(x)
    split = int(n * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x_train, y_train)

    proba = clf.predict_proba(x_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else 0.0

    joblib.dump(clf, os.path.join(out_dir, "switch_filter.pkl"))
    with open(os.path.join(out_dir, "switch_filter_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "symbol": symbol,
                "interval": interval,
                "period": period,
                "feature_columns": feats,
                "label_horizon": label_horizon,
                "label_thr_pct": label_thr_pct,
                "lookahead": lookahead,
                "auc_test": auc,
            },
            f,
            indent=2,
        )

    print(f"✅ switch_filter.pkl exported | AUC={auc:.4f} | feats={len(feats)}")
    raise SystemExit(0)

# ML модели
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, VotingClassifier, BaggingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

# Метрики и preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, roc_curve, confusion_matrix, classification_report,
                             mean_squared_error, mean_absolute_error, r2_score)

# Для работы с датами
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

print(f"✅ TensorFlow: {tf.__version__}")
print(f"📅 Дата: {datetime.now().strftime('%Y-%m-%d')}")

# =====================================================
# 2. ЗАГРУЗКА ДАННЫХ (6 ЛЕТ, 8 ФОРЕКС ПАР)
# =====================================================
print("\n" + "="*70)
print("ЗАГРУЗКА ДАННЫХ ЗА 6 ЛЕТ")
print("="*70)

end_date = datetime.now()
start_date = end_date - timedelta(days=6*365)

print(f"Период: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")

forex_pairs = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X', 
    'USDJPY': 'JPY=X',
    'AUDUSD': 'AUDUSD=X',
    'USDCAD': 'CAD=X',
    'EURJPY': 'EURJPY=X',
    'EURGBP': 'EURGBP=X',
    'USDCHF': 'CHF=X'
}

market_data = {}
for name, ticker in forex_pairs.items():
    print(f"Загрузка {name}...", end=" ")
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if not df.empty:
            market_data[name] = df
            print(f"✅ {len(df)} свечей")
        else:
            print("❌ нет данных")
    except Exception as e:
        print(f"❌ ошибка")

print(f"\n✅ Загружено: {len(market_data)} пар")

# Выбираем основную пару
main_pair = 'EURUSD'
df_raw = market_data[main_pair].copy()
print(f"\nАнализ для {main_pair}")
print(df_raw.head())

# =====================================================
# 3. FEATURE ENGINEERING (БОЛЬШЕ 100 ПРИЗНАКОВ)
# =====================================================
print("\n" + "="*70)
print("СОЗДАНИЕ ПРИЗНАКОВ")
print("="*70)

df = df_raw.copy()

# Базовые признаки
df['returns'] = df['Close'].pct_change()
df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
df['high_low_ratio'] = df['High'] / df['Low']
df['close_open_ratio'] = df['Close'] / df['Open']
df['range'] = df['High'] - df['Low']
df['range_pct'] = df['range'] / df['Close']

# Скользящие средние (разные периоды)
sma_periods = [5, 10, 20, 30, 50, 100, 200]
for period in sma_periods:
    df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
    df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    df[f'SMA_High_{period}'] = df['High'].rolling(period).mean()
    df[f'SMA_Low_{period}'] = df['Low'].rolling(period).mean()

# RSI (несколько периодов)
rsi_periods = [7, 14, 21, 28]
for period in rsi_periods:
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

# MACD (разные конфигурации)
macd_configs = [(12, 26, 9), (8, 17, 9), (19, 39, 9), (10, 20, 5)]
for fast, slow, signal in macd_configs:
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df[f'MACD_{fast}_{slow}'] = ema_fast - ema_slow
    df[f'MACD_Signal_{fast}_{slow}'] = df[f'MACD_{fast}_{slow}'].ewm(span=signal, adjust=False).mean()
    df[f'MACD_Hist_{fast}_{slow}'] = df[f'MACD_{fast}_{slow}'] - df[f'MACD_Signal_{fast}_{slow}']

# Bollinger Bands (разные периоды и отклонения)
bb_configs = [(20, 2), (20, 3), (14, 2), (30, 2), (50, 2)]
for period, std_dev in bb_configs:
    middle = df['Close'].rolling(period).mean()
    std = df['Close'].rolling(period).std()
    df[f'BB_Upper_{period}_{std_dev}'] = middle + std_dev * std
    df[f'BB_Lower_{period}_{std_dev}'] = middle - std_dev * std
    df[f'BB_Width_{period}_{std_dev}'] = (df[f'BB_Upper_{period}_{std_dev}'] - df[f'BB_Lower_{period}_{std_dev}']) / middle
    df[f'BB_Position_{period}_{std_dev}'] = (df['Close'] - df[f'BB_Lower_{period}_{std_dev}']) / (df[f'BB_Upper_{period}_{std_dev}'] - df[f'BB_Lower_{period}_{std_dev}'])

# ATR (Average True Range)
atr_periods = [7, 14, 21, 30]
for period in atr_periods:
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f'ATR_{period}'] = tr.rolling(period).mean()
    df[f'ATR_Pct_{period}'] = df[f'ATR_{period}'] / df['Close']

# Стохастический осциллятор
stoch_periods = [14, 21, 28]
for period in stoch_periods:
    low_min = df['Low'].rolling(period).min()
    high_max = df['High'].rolling(period).max()
    df[f'Stoch_K_{period}'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df[f'Stoch_D_{period}'] = df[f'Stoch_K_{period}'].rolling(3).mean()

# Rate of Change (ROC)
roc_periods = [1, 5, 10, 20, 50]
for period in roc_periods:
    df[f'ROC_{period}'] = df['Close'].pct_change(period)

# Объёмные индикаторы
df['volume_sma'] = df['Volume'].rolling(20).mean()
df['volume_ratio'] = df['Volume'] / df['volume_sma']
df['volume_change'] = df['Volume'].pct_change()
df['obv'] = (np.sign(df['returns']) * df['Volume']).fillna(0).cumsum()

# Волатильность
df['volatility_20'] = df['returns'].rolling(20).std()
df['volatility_50'] = df['returns'].rolling(50).std()
df['volatility_ratio'] = df['volatility_20'] / df['volatility_50']

# Сезонность и календарные признаки
df['day_of_week'] = df.index.dayofweek
df['day_of_month'] = df.index.day
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['year'] = df.index.year
df['is_month_end'] = df.index.is_month_end.astype(int)
df['is_month_start'] = df.index.is_month_start.astype(int)
df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

# Лаговые переменные
lag_periods = [1, 2, 3, 5, 10, 20]
for lag in lag_periods:
    df[f'close_lag_{lag}'] = df['Close'].shift(lag)
    df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
    df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
    df[f'high_lag_{lag}'] = df['High'].shift(lag)
    df[f'low_lag_{lag}'] = df['Low'].shift(lag)

# Разности цен
diff_periods = [1, 2, 3, 5, 10]
for diff in diff_periods:
    df[f'close_diff_{diff}'] = df['Close'].diff(diff)
    df[f'high_diff_{diff}'] = df['High'].diff(diff)
    df[f'low_diff_{diff}'] = df['Low'].diff(diff)

# Свечные паттерны
df['body'] = abs(df['Close'] - df['Open'])
df['upper_shadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
df['lower_shadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']
df['doji'] = (df['body'] / df['range'] < 0.1).astype(int)
df['marubozu'] = ((df['upper_shadow'] / df['range'] < 0.05) & (df['lower_shadow'] / df['range'] < 0.05)).astype(int)

# Целевая переменная (предсказание через 24 часа)
df['target'] = (df['Close'].shift(-24) > df['Close']).astype(int)

# Очистка
df_clean = df.dropna()
print(f"✅ Всего признаков: {len(df_clean.columns)}")
print(f"✅ Строк после очистки: {len(df_clean)}")
print(f"✅ Баланс целевой: {df_clean['target'].mean():.2%} UP сигналов")

# =====================================================
# 4. ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ
# =====================================================
print("\n" + "="*70)
print("ПОДГОТОВКА ДАННЫХ")
print("="*70)

# Определяем признаки (исключаем ценовые и целевую)
exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'target', 'obv']
feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

print(f"Количество признаков для обучения: {len(feature_cols)}")

# Матрица признаков и целевая
X = df_clean[feature_cols].values
y = df_clean['target'].values

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Временное разделение (80/20, без перемешивания)
split_idx = int(0.8 * len(X_scaled))
X_train = X_scaled[:split_idx]
X_test = X_scaled[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"Обучающая выборка: {X_train.shape}")
print(f"Тестовая выборка: {X_test.shape}")
print(f"Баланс train: {y_train.mean():.2%} UP")
print(f"Баланс test: {y_test.mean():.2%} UP")

# =====================================================
# 5. МНОЖЕСТВО ML МОДЕЛЕЙ (15 МОДЕЛЕЙ)
# =====================================================
print("\n" + "="*70)
print("ОБУЧЕНИЕ 15 ML МОДЕЛЕЙ")
print("="*70)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=0.1),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=150, learning_rate=0.05, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
    'LightGBM': LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=6, random_state=42, verbose=-1, n_jobs=-1),
    'CatBoost': CatBoostClassifier(iterations=200, learning_rate=0.03, depth=6, random_state=42, verbose=False),
    'SVM RBF': SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=42),
    'SVM Linear': SVC(kernel='linear', probability=True, C=0.1, random_state=42),
    'Naive Bayes': GaussianNB(),
    'Bagging Tree': BaggingClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=100, random_state=42, n_jobs=-1),
    'Extra Trees': RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=10, random_state=42, n_jobs=-1),
    'MLP Classifier': tf.keras.wrappers.scikit_learn.KerasClassifier(model=lambda: Sequential([
        Dense(128, activation='relu', input_dim=X_train.shape[1]),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ]), epochs=50, batch_size=32, verbose=0)
}

results = {}
probabilities_dict = {}
predictions_dict = {}

for name, model in models.items():
    print(f"\n🔄 {name}...", end=" ")
    try:
        model.fit(X_train, y_train)
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "predict"):
            y_pred_class = model.predict(X_test)
            y_proba = y_pred_class
        else:
            y_proba = np.zeros(len(y_test))
        
        y_pred = (y_proba > 0.5).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_proba)) > 1 else 0.5
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results[name] = {'accuracy': acc, 'auc': auc, 'precision': prec, 'recall': rec, 'f1': f1, 'model': model}
        probabilities_dict[name] = y_proba
        predictions_dict[name] = y_pred
        
        print(f"✅ Acc={acc:.4f}, AUC={auc:.4f}")
    except Exception as e:
        print(f"❌ Ошибка: {str(e)[:50]}")
        results[name] = {'accuracy': 0, 'auc': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'model': None}

# Создаём DataFrame результатов
results_df = pd.DataFrame([{k: v[k] for k in ['accuracy', 'auc', 'precision', 'recall', 'f1']} 
                           for v in results.values()], index=results.keys())
results_df = results_df.sort_values('auc', ascending=False)

print("\n" + "="*70)
print("РЕЗУЛЬТАТЫ ML МОДЕЛЕЙ")
print("="*70)
print(results_df.round(4))

best_model_name = results_df.index[0]
best_model = results[best_model_name]['model']
print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model_name} (AUC={results_df.iloc[0]['auc']:.4f})")

# =====================================================
# 6. DEEP LEARNING МОДЕЛИ (LSTM, Bi-LSTM, GRU, RNN)
# =====================================================
print("\n" + "="*70)
print("ОБУЧЕНИЕ DEEP LEARNING МОДЕЛЕЙ")
print("="*70)

sequence_length = 60

# Подготовка последовательностей
X_sequences = []
y_sequences = []

for i in range(sequence_length, len(X_scaled) - 24):
    X_sequences.append(X_scaled[i-sequence_length:i])
    y_sequences.append(y[i+24])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

print(f"Данные для DL: X={X_sequences.shape}, y={y_sequences.shape}")

# Разделение
dl_split = int(0.8 * len(X_sequences))
X_train_dl = X_sequences[:dl_split]
y_train_dl = y_sequences[:dl_split]
X_test_dl = X_sequences[dl_split:]
y_test_dl = y_sequences[dl_split:]

print(f"Train DL: {X_train_dl.shape}, Test DL: {X_test_dl.shape}")

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

dl_results = {}

# 1. LSTM
print("\n🔄 LSTM...")
lstm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(sequence_length, X_sequences.shape[2])),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
lstm_history = lstm_model.fit(X_train_dl, y_train_dl, validation_split=0.2, epochs=50, batch_size=64, callbacks=[early_stop, reduce_lr], verbose=0)
lstm_proba = lstm_model.predict(X_test_dl).flatten()
lstm_pred = (lstm_proba > 0.5).astype(int)
lstm_auc = roc_auc_score(y_test_dl, lstm_proba)
lstm_acc = accuracy_score(y_test_dl, lstm_pred)
dl_results['LSTM'] = {'accuracy': lstm_acc, 'auc': lstm_auc, 'history': lstm_history}
print(f"   ✅ Acc={lstm_acc:.4f}, AUC={lstm_auc:.4f}")

# 2. Bi-LSTM
print("\n🔄 Bi-LSTM...")
bi_lstm_model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(sequence_length, X_sequences.shape[2])),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
bi_lstm_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
bi_lstm_history = bi_lstm_model.fit(X_train_dl, y_train_dl, validation_split=0.2, epochs=50, batch_size=64, callbacks=[early_stop, reduce_lr], verbose=0)
bi_lstm_proba = bi_lstm_model.predict(X_test_dl).flatten()
bi_lstm_auc = roc_auc_score(y_test_dl, bi_lstm_proba)
bi_lstm_acc = accuracy_score(y_test_dl, (bi_lstm_proba > 0.5).astype(int))
dl_results['Bi-LSTM'] = {'accuracy': bi_lstm_acc, 'auc': bi_lstm_auc, 'history': bi_lstm_history}
print(f"   ✅ Acc={bi_lstm_acc:.4f}, AUC={bi_lstm_auc:.4f}")

# 3. GRU
print("\n🔄 GRU...")
gru_model = Sequential([
    GRU(128, return_sequences=True, input_shape=(sequence_length, X_sequences.shape[2])),
    Dropout(0.3),
    GRU(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
gru_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
gru_history = gru_model.fit(X_train_dl, y_train_dl, validation_split=0.2, epochs=50, batch_size=64, callbacks=[early_stop, reduce_lr], verbose=0)
gru_proba = gru_model.predict(X_test_dl).flatten()
gru_auc = roc_auc_score(y_test_dl, gru_proba)
gru_acc = accuracy_score(y_test_dl, (gru_proba > 0.5).astype(int))
dl_results['GRU'] = {'accuracy': gru_acc, 'auc': gru_auc, 'history': gru_history}
print(f"   ✅ Acc={gru_acc:.4f}, AUC={gru_auc:.4f}")

# 4. Simple RNN
print("\n🔄 Simple RNN...")
rnn_model = Sequential([
    SimpleRNN(128, return_sequences=True, input_shape=(sequence_length, X_sequences.shape[2])),
    Dropout(0.3),
    SimpleRNN(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
rnn_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
rnn_history = rnn_model.fit(X_train_dl, y_train_dl, validation_split=0.2, epochs=50, batch_size=64, callbacks=[early_stop, reduce_lr], verbose=0)
rnn_proba = rnn_model.predict(X_test_dl).flatten()
rnn_auc = roc_auc_score(y_test_dl, rnn_proba)
rnn_acc = accuracy_score(y_test_dl, (rnn_proba > 0.5).astype(int))
dl_results['SimpleRNN'] = {'accuracy': rnn_acc, 'auc': rnn_auc, 'history': rnn_history}
print(f"   ✅ Acc={rnn_acc:.4f}, AUC={rnn_auc:.4f}")

# Добавляем DL результаты в общую таблицу
for name, res in dl_results.items():
    results_df.loc[name] = [res['accuracy'], res['auc'], 0, 0, 0]

results_df = results_df.sort_values('auc', ascending=False)
print("\n" + "="*70)
print("ИТОГОВЫЙ РЕЙТИНГ (ML + DL)")
print("="*70)
print(results_df.round(4))

# =====================================================
# 7. АНСАМБЛЬ VOTING CLASSIFIER
# =====================================================
print("\n" + "="*70)
print("СОЗДАНИЕ АНСАМБЛЯ")
print("="*70)

# Берём топ-5 моделей по AUC
top5_models = results_df.head(5).index.tolist()
print(f"Топ-5 моделей для ансамбля: {top5_models}")

voting_estimators = []
for name in top5_models:
    if name in results and results[name]['model'] is not None:
        voting_estimators.append((name.replace(' ', '_'), results[name]['model']))

if len(voting_estimators) >= 2:
    voting_clf = VotingClassifier(estimators=voting_estimators, voting='soft')
    voting_clf.fit(X_train, y_train)
    voting_proba = voting_clf.predict_proba(X_test)[:, 1]
    voting_pred = (voting_proba > 0.5).astype(int)
    voting_auc = roc_auc_score(y_test, voting_proba)
    voting_acc = accuracy_score(y_test, voting_pred)
    
    print(f"✅ Voting Ensemble - Acc={voting_acc:.4f}, AUC={voting_auc:.4f}")
    
    # Добавляем в результаты
    results_df.loc['Voting Ensemble'] = [voting_acc, voting_auc, 0, 0, 0]
    results_df = results_df.sort_values('auc', ascending=False)
    print("\n📊 Финальный рейтинг с ансамблем:")
    print(results_df.head(10).round(4))

# =====================================================
# 8. ГРАФИКИ: СРАВНЕНИЕ МОДЕЛЕЙ
# =====================================================
print("\n" + "="*70)
print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("="*70)

fig = plt.figure(figsize=(22, 18))

# 1. Сравнение AUC (топ-15)
ax1 = plt.subplot(3, 3, 1)
top15 = results_df.head(15)
colors = ['gold' if i==0 else 'silver' if i==1 else 'bronze' if i==2 else 'steelblue' for i in range(len(top15))]
ax1.barh(range(len(top15)), top15['auc'].values, color=colors, alpha=0.7)
ax1.set_yticks(range(len(top15)))
ax1.set_yticklabels(top15.index, fontsize=9)
ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
ax1.set_title('AUC Comparison - Top 15 Models', fontsize=12)
ax1.set_xlabel('AUC Score')
ax1.set_xlim(0.45, max(0.75, top15['auc'].max() + 0.05))
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. ROC Curves (топ-5)
ax2 = plt.subplot(3, 3, 2)
top5_names = results_df.head(5).index
colors_roc = ['blue', 'red', 'green', 'orange', 'purple']
for idx, name in enumerate(top5_names):
    if name in probabilities_dict:
        fpr, tpr, _ = roc_curve(y_test, probabilities_dict[name])
        ax2.plot(fpr, tpr, color=colors_roc[idx], label=f'{name[:15]} (AUC={results_df.loc[name, "auc"]:.3f})', linewidth=2)
    elif name in dl_results:
        if name == 'LSTM':
            fpr, tpr, _ = roc_curve(y_test_dl, lstm_proba)
        elif name == 'Bi-LSTM':
            fpr, tpr, _ = roc_curve(y_test_dl, bi_lstm_proba)
        elif name == 'GRU':
            fpr, tpr, _ = roc_curve(y_test_dl, gru_proba)
        else:
            continue
        ax2.plot(fpr, tpr, color=colors_roc[idx], label=f'{name} (AUC={results_df.loc[name, "auc"]:.3f})', linewidth=2)
ax2.plot([0,1], [0,1], 'k--', alpha=0.5)
ax2.set_title('ROC Curves - Top 5 Models', fontsize=12)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

# 3. Confusion Matrix (лучшая модель)
ax3 = plt.subplot(3, 3, 3)
best_name = results_df.index[0]
if best_name in predictions_dict:
    cm = confusion_matrix(y_test, predictions_dict[best_name])
elif best_name in dl_results:
    if best_name == 'LSTM':
        cm = confusion_matrix(y_test_dl, lstm_pred)
    elif best_name == 'Bi-LSTM':
        cm = confusion_matrix(y_test_dl, (bi_lstm_proba > 0.5).astype(int))
    elif best_name == 'GRU':
        cm = confusion_matrix(y_test_dl, (gru_proba > 0.5).astype(int))
    else:
        cm = confusion_matrix(y_test_dl, (rnn_proba > 0.5).astype(int))
else:
    cm = confusion_matrix(y_test, voting_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
ax3.set_title(f'Confusion Matrix - {best_name}', fontsize=12)
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')

# 4. Feature Importance (Random Forest)
ax4 = plt.subplot(3, 3, 4)
if 'Random Forest' in results and results['Random Forest']['model'] is not None:
    rf_model = results['Random Forest']['model']
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False).head(20)
    ax4.barh(range(len(importances)), importances.values, color='teal')
    ax4.set_yticks(range(len(importances)))
    ax4.set_yticklabels(importances.index, fontsize=8)
    ax4.set_title('Feature Importance (Random Forest) - Top 20', fontsize=12)
    ax4.set_xlabel('Importance')
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3)

# 5. Feature Importance (XGBoost)
ax5 = plt.subplot(3, 3, 5)
if 'XGBoost' in results and results['XGBoost']['model'] is not None:
    xgb_model = results['XGBoost']['model']
    importances_xgb = pd.Series(xgb_model.feature_importances_, index=feature_cols)
    importances_xgb = importances_xgb.sort_values(ascending=False).head(20)
    ax5.barh(range(len(importances_xgb)), importances_xgb.values, color='coral')
    ax5.set_yticks(range(len(importances_xgb)))
    ax5.set_yticklabels(importances_xgb.index, fontsize=8)
    ax5.set_title('Feature Importance (XGBoost) - Top 20', fontsize=12)
    ax5.set_xlabel('Importance')
    ax5.invert_yaxis()
    ax5.grid(True, alpha=0.3)

# 6. DL Training Curves (LSTM)
ax6 = plt.subplot(3, 3, 6)
lstm_hist = dl_results['LSTM']['history']
ax6.plot(lstm_hist.history['loss'], label='Train Loss')
ax6.plot(lstm_hist.history['val_loss'], label='Val Loss')
ax6.set_title('LSTM - Training Loss', fontsize=12)
ax6.set_xlabel('Epoch')
ax6.set_ylabel('Loss')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. DL Training Curves (Bi-LSTM)
ax7 = plt.subplot(3, 3, 7)
bi_hist = dl_results['Bi-LSTM']['history']
ax7.plot(bi_hist.history['accuracy'], label='Train Acc')
ax7.plot(bi_hist.history['val_accuracy'], label='Val Acc')
ax7.set_title('Bi-LSTM - Accuracy', fontsize=12)
ax7.set_xlabel('Epoch')
ax7.set_ylabel('Accuracy')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Радарная диаграмма (топ-5)
ax8 = plt.subplot(3, 3, 8, projection='polar')
metrics_radar = ['accuracy', 'auc', 'precision', 'recall', 'f1']
top5_metrics = results_df.head(5)[metrics_radar]
angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
angles += angles[:1]
for idx, name in enumerate(top5_metrics.index):
    values = top5_metrics.loc[name].values.tolist()
    values += values[:1]
    ax8.plot(angles, values, 'o-', linewidth=2, label=name[:15])
    ax8.fill(angles, values, alpha=0.1)
ax8.set_xticks(angles[:-1])
ax8.set_xticklabels(metrics_radar)
ax8.set_title('Radar Chart - Top 5 Models', fontsize=12)
ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# 9. Корреляция предсказаний
ax9 = plt.subplot(3, 3, 9)
top5_preds = {}
for name in top5_names[:5]:
    if name in predictions_dict:
        top5_preds[name] = predictions_dict[name]
    elif name in dl_results:
        if name == 'LSTM':
            top5_preds[name] = lstm_pred
        elif name == 'Bi-LSTM':
            top5_preds[name] = (bi_lstm_proba > 0.5).astype(int)
        elif name == 'GRU':
            top5_preds[name] = (gru_proba > 0.5).astype(int)
if len(top5_preds) >= 2:
    pred_corr = pd.DataFrame(top5_preds).corr()
    sns.heatmap(pred_corr, annot=True, fmt='.3f', cmap='coolwarm', ax=ax9, square=True)
    ax9.set_title('Predictions Correlation (Top 5)', fontsize=12)

plt.tight_layout()
plt.show()

# =====================================================
# 9. ТОРГОВАЯ СИМУЛЯЦИЯ (BACKTEST)
# =====================================================
print("\n" + "="*70)
print("ТОРГОВАЯ СИМУЛЯЦИЯ")
print("="*70)

# Выбираем лучшую модель для торговли
best_proba = None
if best_name in probabilities_dict:
    best_proba = probabilities_dict[best_name]
    test_prices = df_clean['Close'].iloc[split_idx:split_idx+len(y_test)].values
elif best_name in dl_results:
    if best_name == 'LSTM':
        best_proba = lstm_proba
        test_prices = df_clean['Close'].iloc[split_idx+sequence_length+24:split_idx+sequence_length+24+len(y_test_dl)].values
    elif best_name == 'Bi-LSTM':
        best_proba = bi_lstm_proba
        test_prices = df_clean['Close'].iloc[split_idx+sequence_length+24:split_idx+sequence_length+24+len(y_test_dl)].values
    else:
        best_proba = gru_proba
        test_prices = df_clean['Close'].iloc[split_idx+sequence_length+24:split_idx+sequence_length+24+len(y_test_dl)].values
else:
    best_proba = voting_proba
    test_prices = df_clean['Close'].iloc[split_idx:split_idx+len(y_test)].values

# Симуляция торговли
initial_capital = 10000
capital = initial_capital
equity_curve = [capital]
trades = []
position = None
confidence_threshold = 0.65

for i, (proba, price) in enumerate(zip(best_proba, test_prices)):
    if position is None:
        if proba > confidence_threshold:
            position = {
                'entry_price': price,
                'entry_idx': i,
                'risk': capital * 0.02
            }
    else:
        exit_price = price
        pnl = position['risk'] * (exit_price / position['entry_price'] - 1)
        capital += pnl
        trades.append({'pnl': pnl, 'win': pnl > 0})
        position = None
        equity_curve.append(capital)

if trades:
    wins = sum(1 for t in trades if t['win'])
    win_rate = wins / len(trades)
    total_return = (capital / initial_capital - 1) * 100
    avg_win = np.mean([t['pnl'] for t in trades if t['win']]) if wins > 0 else 0
    avg_loss = np.mean([t['pnl'] for t in trades if not t['win']]) if (len(trades)-wins) > 0 else 0
    sharpe = np.mean([t['pnl'] for t in trades]) / (np.std([t['pnl'] for t in trades]) + 1e-6) * np.sqrt(252)
    
    print(f"\n📊 Торговые результаты ({best_name}):")
    print(f"   Начальный капитал: ${initial_capital:,.2f}")
    print(f"   Финальный капитал: ${capital:,.2f}")
    print(f"   Доходность: {total_return:.2f}%")
    print(f"   Всего сделок: {len(trades)}")
    print(f"   Win Rate: {win_rate:.1%}")
    print(f"   Средний профит: ${avg_win:.2f}")
    print(f"   Средний убыток: ${avg_loss:.2f}")
    print(f"   Sharpe Ratio: {sharpe:.3f}")

# =====================================================
# 10. ФИНАЛЬНЫЕ ГРАФИКИ ТОРГОВЛИ
# =====================================================
print("\n" + "="*70)
print("ФИНАЛЬНАЯ ВИЗУАЛИЗАЦИЯ ТОРГОВЛИ")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Equity Curve
ax1 = axes[0,0]
ax1.plot(equity_curve, color='green', linewidth=2)
ax1.axhline(y=initial_capital, color='red', linestyle='--', alpha=0.7)
ax1.fill_between(range(len(equity_curve)), initial_capital, equity_curve, 
                  where=np.array(equity_curve) >= initial_capital, color='green', alpha=0.3)
ax1.fill_between(range(len(equity_curve)), initial_capital, equity_curve, 
                  where=np.array(equity_curve) < initial_capital, color='red', alpha=0.3)
ax1.set_title(f'Equity Curve - {best_name}', fontsize=12)
ax1.set_xlabel('Trade Number')
ax1.set_ylabel('Capital ($)')
ax1.grid(True, alpha=0.3)

# 2. PnL Distribution
ax2 = axes[0,1]
if trades:
    pnls = [t['pnl'] for t in trades]
    ax2.hist(pnls, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=np.mean(pnls), color='green', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(pnls):.2f}')
    ax2.set_title(f'PnL Distribution ({len(trades)} trades)', fontsize=12)
    ax2.set_xlabel('PnL ($)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# 3. Cumulative Return
ax3 = axes[0,2]
cumulative_returns = (np.array(equity_curve) / initial_capital - 1) * 100
ax3.plot(cumulative_returns, color='purple', linewidth=2)
ax3.fill_between(range(len(cumulative_returns)), 0, cumulative_returns, alpha=0.3, color='purple')
ax3.set_title('Cumulative Return (%)', fontsize=12)
ax3.set_xlabel('Trade Number')
ax3.set_ylabel('Return (%)')
ax3.grid(True, alpha=0.3)

# 4. Drawdown
ax4 = axes[1,0]
running_max = np.maximum.accumulate(equity_curve)
drawdown = (running_max - equity_curve) / running_max * 100
ax4.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.5)
ax4.set_title('Drawdown (%)', fontsize=12)
ax4.set_xlabel('Trade Number')
ax4.set_ylabel('Drawdown (%)')
ax4.grid(True, alpha=0.3)

# 5. Win/Loss by Trade
ax5 = axes[1,1]
if trades:
    colors_trades = ['green' if t['win'] else 'red' for t in trades]
    ax5.bar(range(len(trades)), [t['pnl'] for t in trades], color=colors_trades, alpha=0.7)
    ax5.axhline(y=0, color='black', linewidth=1)
    ax5.set_title('Individual Trade PnL', fontsize=12)
    ax5.set_xlabel('Trade Number')
    ax5.set_ylabel('PnL ($)')
    ax5.grid(True, alpha=0.3)

# 6. Summary Table
ax6 = axes[1,2]
ax6.axis('off')
summary_text = f"""
TRADING SUMMARY - {best_name}
{'='*35}

Initial Capital:    ${initial_capital:,.2f}
Final Capital:      ${capital:,.2f}
Total Return:       {total_return:.2f}%

Total Trades:       {len(trades)}
Win Rate:           {win_rate:.1%}
Profit Factor:      {abs(sum([t['pnl'] for t in trades if t['win']]) / sum([t['pnl'] for t in trades if not t['win']])) if len([t for t in trades if not t['win']]) > 0 else float('inf'):.2f}

Avg Win:            ${avg_win:.2f}
Avg Loss:           ${avg_loss:.2f}
Sharpe Ratio:       {sharpe:.3f}
Max Drawdown:       {max(drawdown):.1f}%
"""
ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=10,
         verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.show()

# =====================================================
# 11. ИТОГОВЫЙ ОТЧЁТ
# =====================================================
print("\n" + "="*70)
print("ИТОГОВЫЙ ОТЧЁТ")
print("="*70)
print(f"""
📅 ПЕРИОД: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}
📊 ВАЛЮТНАЯ ПАРА: {main_pair}
🔬 ПРИЗНАКОВ: {len(feature_cols)}
🤖 ML МОДЕЛЕЙ: {len([m for m in results.keys() if m not in dl_results.keys()])}
🧠 DL МОДЕЛЕЙ: {len(dl_results)}
🏆 ЛУЧШАЯ МОДЕЛЬ: {best_name}
🎯 AUC ЛУЧШЕЙ МОДЕЛИ: {results_df.loc[best_name, 'auc']:.4f}
💰 ДОХОДНОСТЬ: {total_return:.2f}%
🎯 WIN RATE: {win_rate:.1%}
📈 SHARPE RATIO: {sharpe:.3f}
""")

print("="*70)
print("✅ АНАЛИЗ ЗАВЕРШЁН")
print("="*70)


def export_switch_model():
    import os
    import json
    import joblib
    import bot as bot_module

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    symbol = os.getenv("SWITCH_SYMBOL", "EURUSD=X")
    interval = os.getenv("SWITCH_INTERVAL", "1h")
    period = os.getenv("SWITCH_PERIOD", "365d")
    label_horizon = int(os.getenv("SWITCH_LABEL_HORIZON", 12))
    label_thr_pct = float(os.getenv("SWITCH_LABEL_THR_PCT", 0.20)) / 100.0
    lookahead = int(os.getenv("SWITCH_LOOKAHEAD", 6))

    out_dir = os.getenv("SWITCH_OUT_DIR", "models")
    os.makedirs(out_dir, exist_ok=True)

    raw = yf.Ticker(symbol).history(period=period, interval=interval)
    if raw is None or raw.empty:
        raise RuntimeError("No data")

    raw = raw[["Open", "High", "Low", "Close", "Volume"]]
    raw.columns = ["open", "high", "low", "close", "volume"]

    df = bot_module.calculate_indicators(raw).dropna().copy()

    close = pd.to_numeric(df["close"], errors="coerce")
    fwd = (close.shift(-label_horizon) / close) - 1.0
    label = np.where(fwd > label_thr_pct, 2, np.where(fwd < -label_thr_pct, 0, 1))
    label = pd.Series(label, index=df.index)

    sw = (label.ne(label.shift(1))).astype(int)
    sw.iloc[0] = 0

    future_sw = pd.concat([sw.shift(-i) for i in range(1, lookahead + 1)], axis=1)
    y = future_sw.max(axis=1)

    feats = [
        "returns", "log_returns", "volatility", "ema_8", "ema_21", "ema_50",
        "macd", "macd_signal", "macd_hist", "rsi", "atr", "bb_upper", "bb_lower",
        "close_kalman", "close_kalman_std", "ema_cross",
        "jaw", "teeth", "lips",
        "alligator_asleep", "alligator_bullish", "alligator_bearish", "alligator_expanding",
        "fractal_bullish", "fractal_bearish",
        "bullish_fractal_alligator", "bearish_fractal_alligator",
        "strong_bullish", "strong_bearish",
        "returns_lag_1", "returns_lag_2", "returns_lag_3", "returns_lag_5",
        "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_5",
    ]
    feats = [c for c in feats if c in df.columns]

    x = df[feats].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y = y.loc[x.index].fillna(0).astype(int)

    n = len(x)
    split = int(n * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x_train, y_train)

    proba = clf.predict_proba(x_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else 0.0

    joblib.dump(clf, os.path.join(out_dir, "switch_filter.pkl"))
    with open(os.path.join(out_dir, "switch_filter_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "symbol": symbol,
                "interval": interval,
                "period": period,
                "feature_columns": feats,
                "label_horizon": label_horizon,
                "label_thr_pct": label_thr_pct,
                "lookahead": lookahead,
                "auc_test": auc,
            },
            f,
            indent=2,
        )

    print(f"✅ switch_filter.pkl exported | AUC={auc:.4f} | feats={len(feats)}")


if os.getenv("EXPORT_SWITCH_MODEL", "false").strip().lower() in ("1", "true", "yes"):
    export_switch_model()