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


def _cost(notional: float) -> float:
    return float(notional) * (FEE_BPS + SPREAD_BPS) / 10_000.0


def run_once(df: pd.DataFrame, loader: MLModelLoader, atr_sl: float, atr_tp: float):
    bal = float(START_CAPITAL)
    pos = None
    trades = []

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
            r.update({"atr_sl": sl, "atr_tp": tp})
            rows.append(r)

    out = pd.DataFrame(rows).sort_values(by=["pf", "expectancy"], ascending=False)
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()