
"""
Trading Assistant Module
- Multi-timeframe computer vision analysis (Weekly, 4h, 1h, 15m, 5m)
- Smart Money Concepts (SMC) + SNR
- DeepSeek (text) + Vision model via OpenRouter
"""

import os
import json
import base64
import io
from datetime import datetime
from typing import Any, Dict, Optional
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image, ImageDraw, ImageFont

try:
    import openai
except ImportError:
    openai = None

load_dotenv()

class TradingAssistant:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "").strip()
        self.text_model = os.getenv("OPENROUTER_TEXT_MODEL", "deepseek/deepseek-chat")
        self.vision_model = os.getenv("OPENROUTER_VISION_MODEL", "openai/gpt-4o-mini")
        
        self.timeframes = {
            "1wk": "Weekly",
            "4h": "4-Hour",
            "1h": "1-Hour",
            "15m": "15-Minute",
            "5m": "5-Minute"
        }

        self.client = None
        if self.api_key and openai:
            self._init_client(self.api_key)

    def _init_client(self, api_key: str) -> None:
        headers = {}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=headers if headers else None,
        )

    def ensure_client(self, api_key: Optional[str] = None) -> None:
        if api_key is not None:
            self.api_key = str(api_key).strip()
        if not openai:
            raise RuntimeError("openai library is not installed")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing")
        if self.client is None:
            self._init_client(self.api_key)

    def fetch_ohlcv(self, symbol: str, *, interval: str, period: str) -> Optional[pd.DataFrame]:
        try:
            df = yf.Ticker(symbol).history(period=period, interval=interval)
        except Exception:
            return None
        if df is None or df.empty:
            return None
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        if len(cols) < 4:
            return None
        out = df[cols].copy()
        out.columns = [c.lower() for c in cols]
        try:
            idx = pd.to_datetime(out.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            out.index = idx
        except Exception:
            pass
        return out.dropna()

    def resample_ohlc(self, df: pd.DataFrame, rule: str) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        if not isinstance(df.index, pd.DatetimeIndex):
            return None
        try:
            o = df["open"].resample(rule).first()
            h = df["high"].resample(rule).max()
            l = df["low"].resample(rule).min()
            c = df["close"].resample(rule).last()
            v = df["volume"].resample(rule).sum() if "volume" in df.columns else None
            out = pd.concat([o, h, l, c], axis=1)
            out.columns = ["open", "high", "low", "close"]
            if v is not None:
                out["volume"] = v
            return out.dropna()
        except Exception:
            return None

    def _ema(self, s: pd.Series, span: int) -> pd.Series:
        try:
            return s.ewm(span=int(span), adjust=False).mean()
        except Exception:
            return pd.Series(index=s.index, dtype=float)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
            if not np.isfinite(out):
                return float(default)
            return out
        except Exception:
            return float(default)

    def _normalize_confidence(self, value: Any) -> float:
        conf = self._safe_float(value, 0.0)
        if conf > 1.0:
            conf = conf / 100.0
        return float(max(0.0, min(1.0, conf)))

    def _swing_levels(self, df: pd.DataFrame, lookback: int = 80) -> Dict[str, list[float]]:
        if df is None or df.empty:
            return {"highs": [], "lows": []}
        d = df.tail(int(max(10, lookback))).copy()
        hi = d["high"].astype(float)
        lo = d["low"].astype(float)
        ph = hi[(hi.shift(1) < hi) & (hi.shift(-1) < hi)].dropna()
        pl = lo[(lo.shift(1) > lo) & (lo.shift(-1) > lo)].dropna()
        highs = [float(x) for x in ph.tail(3).tolist()]
        lows = [float(x) for x in pl.tail(3).tolist()]
        return {"highs": highs, "lows": lows}

    def render_candles_png(
        self,
        df: pd.DataFrame,
        *,
        title: str,
        width: int = 1280,
        height: int = 720,
        max_bars: int = 180,
    ) -> bytes:
        if df is None or df.empty:
            raise ValueError("empty_df")

        d = df.copy()
        d = d[["open", "high", "low", "close"] + (["volume"] if "volume" in d.columns else [])]
        d = d.dropna()
        if len(d) > int(max_bars):
            d = d.iloc[-int(max_bars) :]

        close = d["close"].astype(float)
        ema21 = self._ema(close, 21)
        ema50 = self._ema(close, 50)
        swings = self._swing_levels(d, lookback=min(80, len(d)))
        current_price = self._safe_float(close.iloc[-1], 0.0)

        pmin = float(np.nanmin(d["low"].astype(float).values))
        pmax = float(np.nanmax(d["high"].astype(float).values))
        if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
            raise ValueError("bad_price_range")

        pad = (pmax - pmin) * 0.08
        pmin -= pad
        pmax += pad

        bg = (14, 17, 23)
        fg = (220, 225, 235)
        grid = (40, 45, 58)
        bull = (46, 204, 113)
        bear = (231, 76, 60)
        ema21_c = (52, 152, 219)
        ema50_c = (231, 76, 60)

        img = Image.new("RGB", (int(width), int(height)), bg)
        draw = ImageDraw.Draw(img)

        margin_l = 70
        margin_r = 30
        margin_t = 55
        margin_b = 45

        x0 = margin_l
        y0 = margin_t
        x1 = width - margin_r
        y1 = height - margin_b

        draw.rectangle([x0, y0, x1, y1], outline=grid, width=1)

        for k in range(1, 6):
            yy = y0 + (y1 - y0) * k / 6.0
            draw.line([x0, yy, x1, yy], fill=grid, width=1)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        draw.text((x0, 14), title, fill=fg, font=font)
        draw.text((x0, y1 + 8), f"min={pmin:.5f}  max={pmax:.5f}", fill=(160, 170, 190), font=font)

        n = len(d)
        if n <= 1:
            raise ValueError("not_enough_bars")

        step = (x1 - x0) / float(n)
        body_w = max(2, int(step * 0.62))

        def y_of(p: float) -> float:
            return y1 - (float(p) - pmin) / (pmax - pmin) * (y1 - y0)

        def x_of(i: int) -> float:
            return x0 + (i + 0.5) * step

        for i, (_, row) in enumerate(d.iterrows()):
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])

            col = bull if c >= o else bear
            xi = x_of(i)
            y_open = y_of(o)
            y_close = y_of(c)
            y_high = y_of(h)
            y_low = y_of(l)

            draw.line([xi, y_high, xi, y_low], fill=col, width=2)

            top = min(y_open, y_close)
            bot = max(y_open, y_close)
            if abs(bot - top) < 1.5:
                bot = top + 2.0
            draw.rectangle([xi - body_w / 2, top, xi + body_w / 2, bot], fill=col, outline=col)

        def draw_series(series: pd.Series, color: tuple[int, int, int]):
            pts = []
            for i, v in enumerate(series.iloc[-n:].tolist()):
                try:
                    fv = float(v)
                except Exception:
                    continue
                if not np.isfinite(fv):
                    continue
                pts.append((x_of(i), y_of(fv)))
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=2, joint="curve")

        draw_series(ema21, ema21_c)
        draw_series(ema50, ema50_c)

        # Current price reference
        if current_price > 0:
            y_cur = y_of(current_price)
            draw.line([x0, y_cur, x1, y_cur], fill=(255, 193, 7), width=1)
            draw.text((x1 - 120, max(y0, y_cur - 12)), f"PX {current_price:.5f}", fill=(255, 193, 7), font=font)

        # Recent swing levels to give vision model stronger S/R context.
        for lvl in swings.get("highs", []):
            yy = y_of(lvl)
            draw.line([x0, yy, x1, yy], fill=(180, 80, 80), width=1)
            draw.text((x0 + 4, max(y0, yy - 10)), f"SH {lvl:.5f}", fill=(180, 80, 80), font=font)
        for lvl in swings.get("lows", []):
            yy = y_of(lvl)
            draw.line([x0, yy, x1, yy], fill=(80, 180, 120), width=1)
            draw.text((x0 + 4, max(y0, yy - 10)), f"SL {lvl:.5f}", fill=(80, 180, 120), font=font)

        # Right-top legend.
        legend_x = x1 - 220
        draw.text((legend_x, y0 + 6), "EMA21", fill=ema21_c, font=font)
        draw.text((legend_x, y0 + 22), "EMA50", fill=ema50_c, font=font)
        draw.text((legend_x, y0 + 38), "SH/SL levels", fill=fg, font=font)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def build_images_from_market_data(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        weekly = self.fetch_ohlcv(symbol, interval="1wk", period="2y")
        h1 = self.fetch_ohlcv(symbol, interval="1h", period="60d")
        m15 = self.fetch_ohlcv(symbol, interval="15m", period="60d")
        m5 = self.fetch_ohlcv(symbol, interval="5m", period="7d")

        h4 = self.resample_ohlc(h1, "4H") if h1 is not None else None

        out: Dict[str, Dict[str, Any]] = {}
        if weekly is not None:
            out["1wk"] = {"bytes": self.render_candles_png(weekly, title=f"{symbol} • 1W (auto)"), "mime": "image/png"}
        if h4 is not None:
            out["4h"] = {"bytes": self.render_candles_png(h4, title=f"{symbol} • 4H (auto)"), "mime": "image/png"}
        if h1 is not None:
            out["1h"] = {"bytes": self.render_candles_png(h1, title=f"{symbol} • 1H (auto)"), "mime": "image/png"}
        if m15 is not None:
            out["15m"] = {"bytes": self.render_candles_png(m15, title=f"{symbol} • 15m (auto)"), "mime": "image/png"}
        if m5 is not None:
            out["5m"] = {"bytes": self.render_candles_png(m5, title=f"{symbol} • 5m (auto)"), "mime": "image/png"}

        return out

    def analyze_timeframe_image(self, *, image_bytes: bytes, mime: str, symbol: str, timeframe_key: str, user_prompt_ru: str = "") -> Dict[str, Any]:
        try:
            self.ensure_client()
        except Exception as e:
            return {"error": str(e)}

        tf_name = self.timeframes.get(timeframe_key, str(timeframe_key))
        img_b64 = base64.b64encode(image_bytes or b"").decode("utf-8")
        if not img_b64:
            return {"error": "empty_image"}

        extra = (user_prompt_ru or "").strip()
        if extra:
            extra = "\n\nДоп. контекст от пользователя:\n" + extra

        prompt = f"""
Ты профессиональный трейдер по Smart Money Concepts (SMC) и Support/Resistance.
Таймфрейм: {tf_name}
Символ: {symbol}

Сделай анализ строго по скриншоту графика. Обязательно:
1) Тренд (bullish/bearish/neutral) на этом ТФ
2) Ключевые уровни поддержки и сопротивления (конкретные цены)
3) Ликвидность (sweeps), имбалансы/FVG, order blocks (если видны)
4) Сценарий входа: long/short/none + entry/SL/TP + уверенность
5) Коротко укажи, что должно произойти, чтобы сценарий отменился

Ответ строго JSON:
{{
  "timeframe": "{tf_name}",
  "trend": "bullish|bearish|neutral",
  "support_levels": [0.0],
  "resistance_levels": [0.0],
  "liquidity_sweeps": [{{"price": 0.0, "type": "buy_side|sell_side", "note": ""}}],
  "fair_value_gaps": [{{"price_low": 0.0, "price_high": 0.0, "type": "bullish|bearish", "note": ""}}],
  "order_blocks": [{{"price_low": 0.0, "price_high": 0.0, "type": "bullish|bearish", "note": ""}}],
  "potential_entry": {{"direction": "long|short|none", "entry_price": 0.0, "stop_loss": 0.0, "take_profit_1": 0.0, "take_profit_2": 0.0, "confidence": 0}},
  "invalidation": "string",
  "analysis_notes": "string"
}}
{extra}
""".strip()

        try:
            resp = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                        ],
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=1700,
            )
            return json.loads(resp.choices[0].message.content or "{}")
        except Exception as e:
            return {"error": str(e)}

    def full_vision_assessment(self, *, symbol: str, images: Dict[str, Dict[str, Any]], user_prompt_ru: str = "") -> Dict[str, Any]:
        try:
            self.ensure_client()
        except Exception as e:
            return {"error": str(e)}

        required = ["1wk", "4h", "1h", "15m", "5m"]
        missing = [tf for tf in required if tf not in images]
        if missing:
            return {"error": f"missing_timeframes: {missing}"}

        vision_analyses: Dict[str, Any] = {}
        for tf in required:
            item = images.get(tf) or {}
            vision_analyses[tf] = self.analyze_timeframe_image(
                image_bytes=item.get("bytes") or b"",
                mime=item.get("mime") or "image/png",
                symbol=symbol,
                timeframe_key=tf,
                user_prompt_ru=user_prompt_ru,
            )

        final_recommendation = self.combine_vision_analyses(vision_analyses, symbol)

        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "vision_analyses": vision_analyses,
            "final_recommendation": final_recommendation,
        }

    def _score_setup(self, vision_analyses: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        order = ["1wk", "4h", "1h", "15m", "5m"]
        weights = {"1wk": 30, "4h": 25, "1h": 20, "15m": 15, "5m": 10}
        trend_to_dir = {"bullish": "long", "bearish": "short"}

        final_entry = analysis.get("entry_recommendation") if isinstance(analysis.get("entry_recommendation"), dict) else {}
        final_direction = str(final_entry.get("direction") or "wait").lower()
        if final_direction not in ("long", "short"):
            return {
                "setup_score": 0,
                "trade_allowed": False,
                "alignment_score": 0,
                "guardrails": ["Нет подтвержденного направления long/short в финальной рекомендации."],
                "consensus": {},
            }

        score = 0
        max_score = 100
        guardrails: list[str] = []
        aligned = 0
        total_considered = 0
        per_tf: dict[str, Any] = {}

        for tf in order:
            va = vision_analyses.get(tf) if isinstance(vision_analyses, dict) else None
            if not isinstance(va, dict) or ("error" in va):
                guardrails.append(f"{self.timeframes.get(tf, tf)}: нет валидного vision-анализа.")
                continue

            total_considered += 1
            tf_trend = str(va.get("trend") or "neutral").lower()
            pe = va.get("potential_entry") if isinstance(va.get("potential_entry"), dict) else {}
            tf_dir = str(pe.get("direction") or "none").lower()
            tf_conf = self._normalize_confidence(pe.get("confidence") or va.get("confidence") or 0.0)
            weight = int(weights.get(tf, 10))

            tf_score = 0.0
            if trend_to_dir.get(tf_trend) == final_direction:
                tf_score += weight * 0.6
            elif tf_trend in ("bullish", "bearish"):
                guardrails.append(f"{self.timeframes.get(tf, tf)}: тренд против финального направления.")

            if tf_dir == final_direction:
                tf_score += weight * 0.4
                aligned += 1
            elif tf_dir in ("long", "short"):
                guardrails.append(f"{self.timeframes.get(tf, tf)}: локальный entry-конфликт.")

            tf_score = tf_score * (0.35 + 0.65 * tf_conf)
            score += tf_score
            per_tf[tf] = {
                "trend": tf_trend,
                "direction": tf_dir,
                "confidence": round(tf_conf, 3),
                "score": round(tf_score, 1),
            }

        alignment_score = int(round((aligned / max(1, total_considered)) * 100))

        rr_ok = True
        try:
            rr_raw = str(analysis.get("risk_reward_ratio") or "").replace(",", ".")
            if ":" in rr_raw:
                left, right = rr_raw.split(":", 1)
                rr_val = self._safe_float(right, 0.0) / max(self._safe_float(left, 1.0), 1e-9)
            else:
                rr_val = self._safe_float(rr_raw, 0.0)
            rr_ok = rr_val >= 1.5
            if not rr_ok:
                guardrails.append("Risk/Reward ниже 1.5.")
        except Exception:
            guardrails.append("Не удалось надежно определить Risk/Reward.")
            rr_ok = False

        final_conf = self._normalize_confidence(analysis.get("confidence") or 0.0)
        if final_conf < 0.60:
            guardrails.append("Итоговая уверенность модели ниже 60%.")

        entry_price = self._safe_float(final_entry.get("entry_price"), 0.0)
        sl_price = self._safe_float(final_entry.get("stop_loss"), 0.0)
        tp1_price = self._safe_float(final_entry.get("take_profit_1"), 0.0)
        if entry_price <= 0 or sl_price <= 0 or tp1_price <= 0:
            guardrails.append("Некорректные уровни entry/SL/TP1.")
        else:
            risk = abs(entry_price - sl_price)
            reward = abs(tp1_price - entry_price)
            if reward <= risk:
                guardrails.append("TP1 не перекрывает риск до SL.")

        if alignment_score < 60:
            guardrails.append("Недостаточная согласованность таймфреймов (< 60%).")

        setup_score = int(round(min(max_score, max(0.0, score)) * (0.4 + 0.6 * final_conf)))
        trade_allowed = (setup_score >= 65) and rr_ok and (alignment_score >= 60) and (final_conf >= 0.60)

        return {
            "setup_score": int(setup_score),
            "trade_allowed": bool(trade_allowed),
            "alignment_score": int(alignment_score),
            "guardrails": list(dict.fromkeys(guardrails)),
            "consensus": per_tf,
        }
    
    def combine_vision_analyses(self, vision_analyses: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        try:
            self.ensure_client()
        except Exception as e:
            return {"error": str(e)}
        
        tf_summaries = []
        for tf, va in vision_analyses.items():
            if isinstance(va, dict) and ("error" not in va):
                tf_summaries.append({
                    "timeframe": self.timeframes.get(tf, tf),
                    "trend": va.get("trend", "neutral"),
                    "support_levels": va.get("support_levels", []),
                    "resistance_levels": va.get("resistance_levels", []),
                    "potential_entry": va.get("potential_entry", {}),
                    "analysis_notes": va.get("analysis_notes", ""),
                    "invalidation": va.get("invalidation", ""),
                })
        
        prompt = f"""
Ты профессиональный трейдер, специализирующийся на Smart Money Concepts (SMC) и анализе на нескольких таймфреймах.

Данные компьютерного зрения по {symbol} на всех таймфреймах:
{json.dumps(tf_summaries, ensure_ascii=False, indent=2)}

Сделай итог:
1) Общий тренд по иерархии ТФ: Weekly -> 4H -> 1H -> 15m -> 5m
2) Ключевые уровни (вход/SL/TP1/TP2/TP3) согласованные по ТФ
3) План: контекст (старшие ТФ) + триггер входа (младшие ТФ)
4) Риски и условия отмены

Ответ строго JSON:
{{
  "overall_trend": "bullish|bearish|neutral",
  "timeframe_analysis": [{{"timeframe": "Weekly", "trend": "bullish|bearish|neutral", "key_levels": {{"support": 0.0, "resistance": 0.0}}, "notes": ""}}],
  "entry_recommendation": {{"direction": "long|short|wait", "entry_price": 0.0, "stop_loss": 0.0, "take_profit_1": 0.0, "take_profit_2": 0.0, "take_profit_3": 0.0}},
  "smart_money_analysis": "string",
  "risk_reward_ratio": "string",
  "confidence": 0
}}
""".strip()
        
        try:
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": "Ты профессиональный трейдер, специализирующийся на Smart Money Concepts и компьютерном зрении."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=2000
            )
            analysis = json.loads(response.choices[0].message.content)
            if not isinstance(analysis, dict):
                return {"error": "bad_analysis_payload"}
            quality = self._score_setup(vision_analyses, analysis)
            analysis["setup_score"] = int(quality.get("setup_score", 0))
            analysis["trade_allowed"] = bool(quality.get("trade_allowed", False))
            analysis["alignment_score"] = int(quality.get("alignment_score", 0))
            analysis["guardrails"] = quality.get("guardrails", [])
            analysis["consensus"] = quality.get("consensus", {})
            return analysis
        except Exception as e:
            return {"error": str(e)}
