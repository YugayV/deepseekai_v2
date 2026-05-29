
"""
Trading Assistant Module
- Multi-timeframe computer vision analysis (Weekly, 4h, 1h, 15m, 5m)
- Smart Money Concepts (SMC) + SNR
- DeepSeek (text) + Vision model via OpenRouter
"""

import os
import json
import base64
from datetime import datetime
from typing import Any, Dict, Optional
from dotenv import load_dotenv

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
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}
