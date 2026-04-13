"""Backtest (practical)"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf

from bot import calculate_indicators, MLModelLoader

SYMBOL = os.getenv("BT_SYMBOL", "EURUSD=X")
INTERVAL = os.getenv("BT_INTERVAL", "1h")
PERIOD = os.getenv("BT_PERIOD", "60d")
RISK_PCT = float(os.getenv("BT_RISK_PCT", 1.0)) / 100.0
FEE_BPS = float(os.getenv("BT_FEE_BPS", 2.0))
SPREAD_BPS = float(os.getenv("BT_SPREAD_BPS", 1.0))
START_CAPITAL = float(os.getenv("PAPER_START_CAPITAL", 10000))

ATR_SL_GRID = [1.2, 1.5, 1.8]
ATR_TP_GRID = [2.0, 2.5, 3.0]

ADAPTIVE_LOOKAHEAD_GRID = [3, 6, 12]
ADAPTIVE_THR_GRID = [0.52, 0.55, 0.58, 0.60]
MIN_TRADES = int(os.getenv("BT_MIN_TRADES", 20))


def _cost(notional: float) -> float:
    return float(notional) * (FEE_BPS + SPREAD_BPS) / 10_000.0


def _adaptive_iter(df: pd.DataFrame, lookahead: int):
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    lookahead = int(max(1, lookahead))

    x = pd.DataFrame(index=df.index)
    x["ret1"] = pd.to_numeric(df.get("returns"), errors="coerce").fillna(0.0)
    x["rsi"] = pd.to_numeric(df.get("rsi"), errors="coerce").fillna(0.0)
    x["macd_h"] = pd.to_numeric(df.get("macd_hist"), errors="coerce").fillna(0.0)
    x["atr"] = pd.to_numeric(df.get("atr"), errors="coerce").fillna(0.0)
    x["bb_pos"] = ((pd.to_numeric(df.get("close"), errors="coerce") - pd.to_numeric(df.get("bb_lower"), errors="coerce")) / (pd.to_numeric(df.get("bb_upper"), errors="coerce") - pd.to_numeric(df.get("bb_lower"), errors="coerce"))).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    x["a_bull"] = pd.to_numeric(df.get("alligator_bullish"), errors="coerce").fillna(0.0)
    x["a_bear"] = pd.to_numeric(df.get("alligator_bearish"), errors="coerce").fillna(0.0)
    x["f_bull"] = pd.to_numeric(df.get("fractal_bullish"), errors="coerce").fillna(0.0)
    x["f_bear"] = pd.to_numeric(df.get("fractal_bearish"), errors="coerce").fillna(0.0)
    x = x.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    close = pd.to_numeric(df["close"], errors="coerce")
    y = (close.shift(-lookahead) > close).astype(int)

    scaler = StandardScaler(with_mean=True, with_std=True)
    model = SGDClassifier(loss="log_loss", alpha=0.0005, max_iter=1, tol=None)
    fitted = False

    for i in range(100, len(df) - lookahead - 1):
        xi = x.iloc[: i + 1]
        yi = y.iloc[: i + 1]
        xi = xi.loc[yi.dropna().index]
        yi = yi.dropna()

        if len(yi) < 200:
            continue

        if not fitted:
            scaler.fit(xi)
            xs = scaler.transform(xi)
            model.partial_fit(xs, yi.values, classes=np.array([0, 1]))
            fitted = True
        else:
            xs = scaler.transform(xi.tail(250))
            model.partial_fit(xs, yi.tail(250).values)

        x_last = scaler.transform(x.iloc[i : i + 1])
        p_up = float(model.predict_proba(x_last)[0][1])
        yield i, p_up


def run_once(df: pd.DataFrame, loader: MLModelLoader, atr_sl: float, atr_tp: float, adaptive: dict | None = None):
    bal = float(START_CAPITAL)
    pos = None
    trades = []

    adaptive = adaptive or {}
    use_adaptive = bool(adaptive.get("enabled"))
    lookahead = int(adaptive.get("lookahead") or 6)
    thr = float(adaptive.get("thr") or 0.55)

    adaptive_map = {}
    if use_adaptive:
        try:
            for i, p_up in _adaptive_iter(df, lookahead):
                adaptive_map[i] = float(p_up)
        except Exception:
            adaptive_map = {}
            use_adaptive = False

    for i in range(60, len(df)):
        w = df.iloc[: i + 1]
        px = float(w["close"].iloc[-1])
        atr = float(w["atr"].iloc[-1]) if "atr" in w.columns else 0.0

        if pos is not None:
            side = pos["side"]
            if (side == "long" and (px <= pos["sl"] or px >= pos["tp"])) or (side == "short" and (px >= pos["sl"] or px <= pos["tp"])):
                pnl = (px - pos["entry"]) * pos["size"] if side == "long" else (pos["entry"] - px) * pos["size"]
                bal += pnl
                bal -= _cost(abs(pos["entry"] * pos["size"]))
                bal -= _cost(abs(px * pos["size"]))
                trades.append({"side": side, "entry": pos["entry"], "exit": px, "pnl": pnl, "atr_sl": atr_sl, "atr_tp": atr_tp})
                pos = None
            continue

        ml = loader.predict(w)
        if not ml:
            continue

        reg = str(ml.get("regime_name") or "")
        bull = int(w.get("strong_bullish", pd.Series([0])).iloc[-1]) == 1 or int(w.get("bullish_fractal_alligator", pd.Series([0])).iloc[-1]) == 1
        bear = int(w.get("strong_bearish", pd.Series([0])).iloc[-1]) == 1 or int(w.get("bearish_fractal_alligator", pd.Series([0])).iloc[-1]) == 1

        side = None
        if bull and reg == "bullish":
            side = "long"
        elif bear and reg == "bearish":
            side = "short"
        if side is None or atr <= 0:
            continue

        if use_adaptive and i in adaptive_map:
            p_up = float(adaptive_map[i])
            if side == "long" and p_up < thr:
                continue
            if side == "short" and (1.0 - p_up) < thr:
                continue

        sl = px - atr * atr_sl if side == "long" else px + atr * atr_sl
        tp = px + atr * atr_tp if side == "long" else px - atr * atr_tp
        risk_amt = bal * RISK_PCT
        unit_risk = abs(px - sl)
        if unit_risk <= 0:
            continue
        size = risk_amt / unit_risk
        pos = {"side": side, "entry": px, "sl": sl, "tp": tp, "size": size}

    t = pd.DataFrame(trades)
    if t.empty:
        return {"trades": 0, "win_rate": 0.0, "expectancy": 0.0, "pf": 0.0, "total": 0.0}

    pnl = pd.to_numeric(t["pnl"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else (float("inf") if wins.sum() > 0 else 0.0)
    return {
        "trades": int(len(pnl)),
        "win_rate": float(len(wins) / len(pnl) * 100.0),
        "expectancy": float(pnl.mean()),
        "pf": pf,
        "total": float(pnl.sum()),
    }


def main():
    raw = yf.Ticker(SYMBOL).history(period=PERIOD, interval=INTERVAL)
    raw = raw[["Open", "High", "Low", "Close", "Volume"]]
    raw.columns = ["open", "high", "low", "close", "volume"]
    df = calculate_indicators(raw).dropna()

    loader = MLModelLoader()

    rows = []
    for sl in ATR_SL_GRID:
        for tp in ATR_TP_GRID:
            r = run_once(df, loader, sl, tp)
            r.update({"atr_sl": sl, "atr_tp": tp, "adaptive": False})
            rows.append(r)

    base = pd.DataFrame(rows).sort_values(by=["pf", "expectancy"], ascending=False)
    print("\n=== BASELINE (no adaptive) ===")
    print(base.head(10).to_string(index=False))

    rows2 = []
    for lookahead in ADAPTIVE_LOOKAHEAD_GRID:
        for thr in ADAPTIVE_THR_GRID:
            for sl in ATR_SL_GRID:
                for tp in ATR_TP_GRID:
                    r = run_once(df, loader, sl, tp, adaptive={"enabled": True, "lookahead": lookahead, "thr": thr})
                    r.update({"atr_sl": sl, "atr_tp": tp, "adaptive": True, "lookahead": lookahead, "thr": thr})
                    rows2.append(r)

    out2 = pd.DataFrame(rows2)
    if not out2.empty:
        out2 = out2[out2["trades"] >= MIN_TRADES]
    out2 = out2.sort_values(by=["pf", "expectancy"], ascending=False)

    print("\n=== ADAPTIVE FILTER (best) ===")
    if out2.empty:
        print("No adaptive results (or not enough trades).")
    else:
        print(out2.head(10).to_string(index=False))
        best = out2.iloc[0].to_dict()
        print("\nBEST_ADAPTIVE:")
        print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()