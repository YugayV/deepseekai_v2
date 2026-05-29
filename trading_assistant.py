
"""
Trading Assistant Module
- Multi-timeframe analysis (Weekly, 4h, 1h, 15m, 5m)
- Smart Money Concepts (SMC)
- Support and Resistance (SNR)
- Computer Vision integration
- DeepSeek API integration
- Image annotation capabilities
"""

import os
import json
import base64
import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageDraw, ImageFont

try:
    import openai
except ImportError:
    openai = None

load_dotenv()

class TradingAssistant:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "")
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "")
        self.text_model = os.getenv("OPENROUTER_TEXT_MODEL", "deepseek/deepseek-chat")
        self.vision_model = os.getenv("OPENROUTER_VISION_MODEL", "openai/gpt-4o-mini")
        
        self.timeframes = {
            "1wk": "Weekly",
            "4h": "4-Hour",
            "1h": "1-Hour",
            "15m": "15-Minute",
            "5m": "5-Minute"
        }
        
        self.periods = {
            "1wk": "2y",
            "4h": "60d",
            "1h": "30d",
            "15m": "7d",
            "5m": "2d"
        }
        
        self.client = None
        if self.api_key and openai:
            headers = {}
            if self.site_url:
                headers["HTTP-Referer"] = self.site_url
            if self.app_name:
                headers["X-Title"] = self.app_name
            
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers=headers if headers else None
            )
    
    def fetch_multi_timeframe_data(self, symbol: str) -&gt; Dict[str, pd.DataFrame]:
        """Fetch data for all required timeframes"""
        data = {}
        for tf in self.timeframes:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=self.periods[tf], interval=tf)
                if df is not None and not df.empty:
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    df.columns = ['open', 'high', 'low', 'close', 'volume']
                    data[tf] = df
            except Exception as e:
                print(f"Error fetching {tf} data for {symbol}: {e}")
        return data
    
    def calculate_indicators(self, df: pd.DataFrame) -&gt; pd.DataFrame:
        """Calculate technical indicators including SNR and SMC concepts"""
        df = df.copy()
        
        # EMA
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta &gt; 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta &lt; 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Pivot points (support/resistance)
        df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['r1'] = 2 * df['pivot'] - df['low'].shift(1)
        df['s1'] = 2 * df['pivot'] - df['high'].shift(1)
        df['r2'] = df['pivot'] + (df['high'].shift(1) - df['low'].shift(1))
        df['s2'] = df['pivot'] - (df['high'].shift(1) - df['low'].shift(1))
        
        # Swing highs and lows (for SNR)
        df['swing_high'] = df['high'].rolling(window=3, center=True).apply(
            lambda x: x.iloc[1] if x.iloc[1] == x.max() else np.nan
        )
        df['swing_low'] = df['low'].rolling(window=3, center=True).apply(
            lambda x: x.iloc[1] if x.iloc[1] == x.min() else np.nan
        )
        
        # FVG (Fair Value Gap) - Smart Money Concept
        df['bullish_fvg'] = np.where(
            (df['low'] &gt; df['high'].shift(2)),
            df['low'] - df['high'].shift(2),
            np.nan
        )
        df['bearish_fvg'] = np.where(
            (df['high'] &lt; df['low'].shift(2)),
            df['low'].shift(2) - df['high'],
            np.nan
        )
        
        return df
    
    def generate_chart(self, df: pd.DataFrame, symbol: str, timeframe: str, title: str = "") -&gt; Image.Image:
        """Generate annotated chart image"""
        df = df.tail(100).copy()
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=symbol
            ),
            row=1, col=1
        )
        
        # EMAs
        if 'ema_21' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['ema_21'], name='EMA 21', line=dict(color='blue', width=1)), row=1, col=1)
        if 'ema_50' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['ema_50'], name='EMA 50', line=dict(color='red', width=1)), row=1, col=1)
        
        # Support/Resistance
        if 'swing_high' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['swing_high'], mode='markers', name='Swing High', marker=dict(color='red', size=8, symbol='triangle-down')), row=1, col=1)
        if 'swing_low' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['swing_low'], mode='markers', name='Swing Low', marker=dict(color='green', size=8, symbol='triangle-up')), row=1, col=1)
        
        # RSI
        if 'rsi' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple', width=1)), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(
            title=f"{symbol} - {title or self.timeframes.get(timeframe, timeframe)}",
            xaxis_rangeslider_visible=False,
            height=800,
            template="plotly_dark"
        )
        
        img_bytes = fig.to_image(format="png", width=1200, height=800)
        return Image.open(io.BytesIO(img_bytes))
    
    def analyze_with_deepseek(self, data: Dict[str, pd.DataFrame], symbol: str) -&gt; Dict:
        """Get comprehensive analysis from DeepSeek"""
        if not self.client:
            return {"error": "OpenAI client not initialized"}
        
        # Prepare summary for each timeframe
        tf_summaries = []
        for tf, df in data.items():
            if df is None or df.empty:
                continue
            df = self.calculate_indicators(df)
            last = df.iloc[-1]
            tf_summaries.append({
                "timeframe": self.timeframes[tf],
                "current_price": float(last['close']),
                "trend": "bullish" if last.get('ema_21', 0) &gt; last.get('ema_50', 0) else "bearish",
                "rsi": float(last.get('rsi', 50)),
                "atr": float(last.get('atr', 0)),
                "support": float(last.get('s1', last['low'])),
                "resistance": float(last.get('r1', last['high']))
            })
        
        prompt = f"""
        Ты профессиональный трейдер, специализирующийся на Smart Money Concepts (SMC) и анализе на нескольких таймфреймах.
        
        Проанализируй следующие данные для {symbol}:
        
        {json.dumps(tf_summaries, ensure_ascii=False, indent=2)}
        
        Требования к анализу:
        1. Определи общий тренд на всех таймфреймах (от неделю до 5 минут)
        2. Найди ключевые уровни поддержки и сопротивления
        3. Определи zones of interest (имбалансы, ликвидность)
        4. Дай рекомендацию по входу в позицию (лонг/шорт/ожидание)
        5. Укажи конкретные уровни входа, тейк-профит и стоп-лосс
        6. Объясни логику на основе Smart Money Concepts
        
        Ответь в JSON формате:
        {{
            "overall_trend": "bullish|bearish|neutral",
            "timeframe_analysis": [
                {{
                    "timeframe": "Weekly",
                    "trend": "bullish|bearish|neutral",
                    "key_levels": {{"support": 0, "resistance": 0}},
                    "notes": "..."
                }}
            ],
            "entry_recommendation": {{
                "direction": "long|short|wait",
                "entry_price": 0,
                "stop_loss": 0,
                "take_profit_1": 0,
                "take_profit_2": 0,
                "take_profit_3": 0
            }},
            "smart_money_analysis": "текст с анализом SMC",
            "risk_reward_ratio": "1:2",
            "confidence": 0-100
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": "Ты профессиональный трейдер, специализирующийся на Smart Money Concepts."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=2000
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}
    
    def encode_image_to_base64(self, img: Image.Image) -&gt; str:
        """Encode PIL Image to base64 string"""
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def analyze_chart_with_vision(self, img: Image.Image, symbol: str, timeframe: str) -&gt; Dict:
        """Analyze chart using computer vision via vision model"""
        if not self.client:
            return {"error": "OpenAI client not initialized"}
        
        img_b64 = self.encode_image_to_base64(img)
        
        prompt = f"""
        Analyze this {timeframe} chart for {symbol} using Smart Money Concepts (SMC):
        
        1. Identify trend direction (bullish/bearish/neutral)
        2. Find key support and resistance levels
        3. Identify liquidity zones and sweeps
        4. Find fair value gaps (FVG) and order blocks
        5. Look for potential entry opportunities
        
        Respond in JSON:
        {{
            "trend": "bullish|bearish|neutral",
            "support_levels": [0.0],
            "resistance_levels": [0.0],
            "liquidity_sweeps": [{{"price": 0.0, "type": "bullish|bearish", "timestamp": ""}}],
            "fair_value_gaps": [{{"price_low": 0.0, "price_high": 0.0, "type": "bullish|bearish"}}],
            "potential_entry": {{
                "direction": "long|short|none",
                "entry_price": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "confidence": 0-100
            }},
            "analysis_notes": "string"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=1500
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}
    
    def full_analysis(self, symbol: str) -&gt; Dict:
        """Complete multi-timeframe analysis with mandatory computer vision"""
        data = self.fetch_multi_timeframe_data(symbol)
        if not data:
            return {"error": "No data available"}
        
        # Calculate indicators for all timeframes
        analyzed_data = {}
        charts = {}
        vision_analyses = {}
        for tf, df in data.items():
            analyzed_df = self.calculate_indicators(df)
            analyzed_data[tf] = analyzed_df
            chart_img = self.generate_chart(analyzed_df, symbol, tf)
            charts[tf] = chart_img
            # MANDATORY: Run vision analysis on each chart
            vision_analyses[tf] = self.analyze_chart_with_vision(chart_img, symbol, self.timeframes[tf])
        
        # Combine all vision analyses into a final recommendation
        final_recommendation = self.combine_vision_analyses(vision_analyses, symbol)
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "data": analyzed_data,
            "charts": charts,
            "vision_analyses": vision_analyses,
            "final_recommendation": final_recommendation
        }
    
    def combine_vision_analyses(self, vision_analyses: Dict, symbol: str) -&gt; Dict:
        """Combine vision analyses from all timeframes into final recommendation"""
        if not self.client:
            return {"error": "OpenAI client not initialized"}
        
        # Prepare summary of vision analyses
        tf_summaries = []
        for tf, va in vision_analyses.items():
            if "error" not in va:
                tf_summaries.append({
                    "timeframe": self.timeframes.get(tf, tf),
                    "trend": va.get("trend", "neutral"),
                    "support_levels": va.get("support_levels", []),
                    "resistance_levels": va.get("resistance_levels", []),
                    "potential_entry": va.get("potential_entry", {})
                })
        
        prompt = f"""
        Ты профессиональный трейдер, специализирующийся на Smart Money Concepts (SMC) и анализе на нескольких таймфреймах.
        
        Проанализируй данные компьютерного зрения для {symbol} на всех таймфреймах:
        
        {json.dumps(tf_summaries, ensure_ascii=False, indent=2)}
        
        Требования к анализу:
        1. Определи общий тренд на всех таймфреймах (от неделю до 5 минут)
        2. Найди ключевые уровни поддержки и сопротивления
        3. Определи zones of interest (имбалансы, ликвидность) на основе данных компьютерного зрения
        4. Дай финальную рекомендацию по входу в позицию (лонг/шорт/ожидание)
        5. Укажи конкретные уровни входа, тейк-профит (3 уровня) и стоп-лосс
        6. Объясни логику на основе Smart Money Concepts и компьютерного зрения
        
        Ответь в JSON формате:
        {{
            "overall_trend": "bullish|bearish|neutral",
            "timeframe_analysis": [
                {{
                    "timeframe": "Weekly",
                    "trend": "bullish|bearish|neutral",
                    "key_levels": {{"support": 0, "resistance": 0}},
                    "notes": "..."
                }}
            ],
            "entry_recommendation": {{
                "direction": "long|short|wait",
                "entry_price": 0,
                "stop_loss": 0,
                "take_profit_1": 0,
                "take_profit_2": 0,
                "take_profit_3": 0
            }},
            "smart_money_analysis": "текст с анализом SMC и компьютерным зрением",
            "risk_reward_ratio": "1:2",
            "confidence": 0-100
        }}
        """
        
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
