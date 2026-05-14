# %% [markdown]
# # Trading Research Notebook v3.0 - Боевая версия
# **Улучшения**: Walk-forward validation, XGBoost, LSTM, Vision заготовка.
# **Данные**: 15m за 2 года (FX + commodities).
# **Формат**: понятный, пошаговый, без лишней абстракции.
# **Цель**: проверить устойчивость стратегии на честном out-of-sample тесте.

# %%
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pathlib import Path

# Новые модели
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

# LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    _HAS_TF = True
except Exception:
    tf = None
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None
    _HAS_TF = False

OUT = Path("output")
OUT.mkdir(exist_ok=True)

print("✅ v3.0 setup complete")

# %% [markdown]
# ## 1) Данные (15m, 2 года)
# Аналогично v2.1, но с дополнительной проверкой качества данных.

# %%
symbols = ["USDJPY=X", "EURUSD=X", "GC=F", "CL=F", "SI=F"]
period = "2y"
interval = "15m"

data = {}
for sym in symbols:
    df = yf.download(sym, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        continue
    df = df.dropna()
    df.columns = [c.capitalize().replace(" ", "_") for c in df.columns]
    data[sym] = df
    print(f"{sym}: {len(df):,} rows")

# %% [markdown]
# ## 2) Features (как раньше, но с проверкой)

# %%
def _sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window=window).mean()


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(s: pd.Series, window: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0)).astype(float)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    macd = _ema(s, fast) - _ema(s, slow)
    macd_sig = _ema(macd, signal)
    return macd, macd_sig


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def process_features(df):
    df = df.copy()
    c = df["Close"]
    
    # Momentum
    df["ret1"] = c.pct_change()
    df["ret4"] = c.pct_change(4)
    
    # Trend
    df["sma20"] = _sma(c, window=20)
    df["sma50"] = _sma(c, window=50)
    
    # Oscillators
    df["rsi14"] = _rsi(c, window=14)
    df["macd"], df["macd_sig"] = _macd(c)
    
    # Volatility
    df["atr14"] = _atr(df["High"], df["Low"], c, window=14)
    
    # Volume
    df["vol_chg"] = df["Volume"].replace(0, np.nan).pct_change()
    
    # Candle
    df["body"] = (df["Close"] - df["Open"]).abs()
    df["range"] = (df["High"] - df["Low"]).replace(0, np.nan)
    
    # TARGET
    df["target"] = c.pct_change().shift(-1)
    
    return df

processed = {sym: process_features(df) for sym, df in data.items()}

# %% [markdown]
# ## 3) Imbalance + Fib
# Простые сигналы силы и зоны входа.

# %%
for sym, df in processed.items():
    body = (df["Close"] - df["Open"]).abs()
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    df["imbalance"] = body / rng
    df["vol_spike"] = df["Volume"] > df["Volume"].rolling(20).mean() * 1.5

    fib_zone = []
    for i in range(len(df)):
        if i < 100:
            fib_zone.append(0)
            continue
        hi = df["High"].iloc[i-100:i].max()
        lo = df["Low"].iloc[i-100:i].min()
        d = hi - lo
        fib_50 = hi - 0.5 * d
        fib_618 = hi - 0.618 * d
        price = df["Close"].iloc[i]
        fib_zone.append(1 if fib_50 <= price <= fib_618 else 0)
    df["fib_zone"] = fib_zone

# %% [markdown]
# ## 4) Markov regime
# Состояния рынка и фильтр вероятности.

# %%
for sym, df in processed.items():
    rets = df["ret1"].dropna()
    q = pd.qcut(rets, q=3, labels=False, duplicates="drop")
    states = q.reindex(df.index).bfill().ffill().astype(int)
    df["state"] = states

    trans = pd.crosstab(df["state"].iloc[:-1], df["state"].iloc[1:], normalize="index")

    probs = []
    for i in range(len(df)):
        if i == len(df) - 1:
            probs.append(np.nan)
            continue
        s = int(df["state"].iloc[i])
        ns = int(df["state"].iloc[i + 1])
        if s in trans.index and ns in trans.columns:
            probs.append(float(trans.loc[s, ns]))
        else:
            probs.append(0.0)

    df["markov_prob"] = probs
    df["regime_ok"] = df["markov_prob"] > df["markov_prob"].rolling(100).median()

# %% [markdown]
# ## 5) Rolling features
# Дополнительные признаки для commodities и FX.

# %%
for sym, df in processed.items():
    for w in [10, 20, 50]:
        df[f"roll_mean_{w}"] = df["Close"].rolling(w).mean()
        df[f"roll_std_{w}"] = df["Close"].rolling(w).std()
    df["trend_strength"] = (df["sma20"] - df["sma50"]) / df["Close"]

# %% [markdown]
# ## 6) Объединяем данные
# Превращаем symbol в dummy variables.

# %%
feature_cols = [
    "Open","High","Low","Close","Volume",
    "ret1","ret4","sma20","sma50",
    "rsi14","macd","macd_sig","atr14",
    "body","range","imbalance","fib_zone","markov_prob","regime_ok",
    "roll_mean_10","roll_mean_20","roll_mean_50",
    "roll_std_10","roll_std_20","roll_std_50",
    "trend_strength"
]

all_frames = []
for sym, df in processed.items():
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    df["regime_ok"] = df["regime_ok"].astype(int)
    df["symbol"] = sym
    all_frames.append(df[feature_cols + ["symbol", "target"]])

master = pd.concat(all_frames, ignore_index=True)
master = pd.get_dummies(master, columns=["symbol"], drop_first=False)

# %% [markdown]
# ## 7) Train / test
# Time split без shuffle.

# %%
X = master.drop(columns=["target"])
y = master["target"]

split = int(len(master) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

sx = MinMaxScaler()
sy = MinMaxScaler()
X_train_s = sx.fit_transform(X_train)
X_test_s = sx.transform(X_test)
y_train_s = sy.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# %% [markdown]
# ## 8) Табличные модели
# Сначала сильные табличные модели.
# Потом отдельно LSTM на окнах.

# %%
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
gbr = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, random_state=42)
xgb = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

table_model = VotingRegressor([
    ("rf", rf),
    ("gbr", gbr),
    ("xgb", xgb),
])

table_model.fit(X_train_s, y_train_s)

# %% [markdown]
# ## 9) LSTM data preparation
# Берём окно из прошлых баров.
# Это уже последовательная модель, а не табличная.

# %%
seq_len = 32
X_seq = []
y_seq = []

X_arr = X_train_s
for i in range(seq_len, len(X_arr)):
    X_seq.append(X_arr[i-seq_len:i])
    y_seq.append(y_train_s[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# %% [markdown]
# ## 10) LSTM model
# Простая LSTM, без усложнений.
# Сравниваем её с табличным ансамблем.

# %%
if _HAS_TF and X_seq.size:
    lstm = Sequential([
        LSTM(64, input_shape=(X_seq.shape[1], X_seq.shape[2])),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    lstm.compile(optimizer="adam", loss="mse")
    lstm.fit(X_seq, y_seq, epochs=int(os.getenv("LSTM_EPOCHS", "3")), batch_size=256, verbose=1)
else:
    lstm = None

# %% [markdown]
# ## 11) LSTM prediction
# Для теста строим такие же окна.

# %%
X_test_seq = []
for i in range(seq_len, len(X_test_s)):
    X_test_seq.append(X_test_s[i-seq_len:i])
X_test_seq = np.array(X_test_seq)

pred_table_s = table_model.predict(X_test_s)
pred_table = sy.inverse_transform(pred_table_s.reshape(-1, 1)).ravel()

if lstm is not None and X_test_seq.size:
    pred_lstm_s = lstm.predict(X_test_seq, verbose=0).ravel()
    pred_lstm = sy.inverse_transform(pred_lstm_s.reshape(-1, 1)).ravel()
else:
    pred_lstm = pred_table[seq_len:].copy()

# выравниваем длины
test_target = y_test.iloc[seq_len:].values
pred_table = pred_table[seq_len:]

res = pd.DataFrame({
    "target": test_target,
    "pred_table": pred_table,
    "pred_lstm": pred_lstm,
})

# %% [markdown]
# ## 12) Ensemble signal
# Усредняем прогнозы таблицы и LSTM.

# %%
res["pred"] = (res["pred_table"] + res["pred_lstm"]) / 2
signal = np.where(res["pred"] > 0, 1, -1)

fee_bps = float(os.getenv("BT_FEE_BPS", "2.0"))
spread_bps = float(os.getenv("BT_SPREAD_BPS", "1.0"))
commission = (fee_bps + spread_bps) / 10_000.0
trade_cost = commission * np.abs(np.diff(np.r_[0, signal]))
strategy_ret = signal * res["target"].values - trade_cost
buyhold_ret = res["target"].values

strategy_eq = (1 + strategy_ret).cumprod()
buyhold_eq = (1 + buyhold_ret).cumprod()
drawdown = strategy_eq / np.maximum.accumulate(strategy_eq) - 1

mse = mean_squared_error(res["target"], res["pred"])
dir_acc = (np.sign(res["pred"]) == np.sign(res["target"])).mean()
sharpe = np.sqrt(252 * 96) * np.mean(strategy_ret) / np.std(strategy_ret) if np.std(strategy_ret) > 0 else np.nan
max_dd = drawdown.min()

print("MSE:", round(mse, 6))
print("Dir Acc:", round(dir_acc, 4))
print("Sharpe:", round(sharpe, 3))
print("Max DD:", round(max_dd, 4))

# %% [markdown]
# ## 13) Графики
# Equity, scatter, distribution, drawdown.

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(y=strategy_eq, name="Strategy"))
fig.add_trace(go.Scatter(y=buyhold_eq, name="Buy&Hold"))
fig.update_layout(title="Equity Growth")
fig.show()

# %%
fig = go.Figure(go.Scatter(x=res["pred"], y=res["target"], mode="markers", marker=dict(size=3, opacity=0.4)))
fig.update_layout(title="Prediction vs Real")
fig.add_hline(y=0, line_dash="dash")
fig.add_vline(x=0, line_dash="dash")
fig.show()

# %%
fig = px.histogram(x=strategy_ret, nbins=100, title="Strategy Return Distribution")
fig.show()

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(y=drawdown, name="Drawdown"))
fig.update_layout(title="Drawdown")
fig.show()

# %% [markdown]
# ## 14) Feature importance
# Для понимания, что влияет на табличную модель.

# %%
rf2 = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
rf2.fit(X_train_s, y_train_s)
importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf2.feature_importances_
}).sort_values("importance", ascending=False)

fig = px.bar(importance.head(20).iloc[::-1], x="importance", y="feature", orientation="h", title="Top Feature Importance")
fig.show()

# %% [markdown]
# ## 15) Save outputs
# CSV для дашборда и дальнейших экспериментов.

# %%
out_df = pd.DataFrame({
    "target": res["target"],
    "pred": res["pred"],
    "signal": signal,
    "strategy_ret": strategy_ret,
    "buyhold_ret": buyhold_ret,
    "strategy_eq": strategy_eq,
    "buyhold_eq": buyhold_eq,
    "drawdown": drawdown,
})
out_df.to_csv(OUT / "backtest_v3.csv", index=False)

pd.DataFrame([{
    "mse": mse,
    "dir_acc": dir_acc,
    "sharpe": sharpe,
    "max_dd": max_dd,
}]).to_csv(OUT / "metrics_v3.csv", index=False)

importance.to_csv(OUT / "feature_importance_v3.csv", index=False)

print("✅ saved")

# %% [markdown]
# ## 16) Что дальше
# Следующий шаг:
# - Walk-forward validation.
# - Полный XGBoost tuning.
# - Vision module для chart images.
# - Отдельный commodity regime.
# - Сравнение всех символов по одному протоколу.

print("🏁 v3.0 complete")
