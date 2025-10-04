"""
Simplified AI-powered trading signals endpoint
Uses live market data to generate real trading signals
"""
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class SimpleAITrader:
    """Simple AI trading logic using live market data"""

    async def get_binance_data(self, symbol: str, interval: str = "1h", limit: int = 50):
        """Fetch live OHLCV data from Binance"""
        url = "https://api.binance.com/api/v3/klines"
        from typing import Dict
        params: Dict[str, str] = {
            "symbol": symbol,
            "interval": interval,
            "limit": str(limit)
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Convert to DataFrame
                        df = pd.DataFrame(data, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_volume', 'trades', 'buy_base', 'buy_quote', 'ignore'
                        ])

                        # Convert numeric columns
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col])

                        return df
                    else:
                        logger.warning(f"Failed to fetch {symbol}: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    def analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Simple technical analysis"""
        if len(df) < 20:
            return {"action": "HOLD", "score": 0.0, "reason": "Insufficient data"}

        # Calculate indicators
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi'] = self.calculate_rsi(df['close'])

        current_price = df['close'].iloc[-1]
        sma_5 = df['sma_5'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        rsi = df['rsi'].iloc[-1]

        # Simple trading logic
        score = 0.0
        signals = []

        # Check if we have valid indicators (handle NaN values)
        if pd.isna(sma_5) or pd.isna(sma_20) or pd.isna(rsi):
            return {"action": "HOLD", "score": 0.0, "reason": "Insufficient indicator data"}

        # Trend following
        if sma_5 > sma_20:
            score += 0.3
            signals.append("Uptrend")
        else:
            score -= 0.3
            signals.append("Downtrend")

        # RSI signals - make more aggressive to generate signals
        if rsi < 40:  # Less strict oversold
            score += 0.4
            signals.append("Oversold")
        elif rsi > 60:  # Less strict overbought
            score -= 0.4
            signals.append("Overbought")

        # Price momentum
        price_change = (current_price - df['close'].iloc[-5]) / df['close'].iloc[-5]
        if price_change > 0.01:  # Lower threshold
            score += 0.3
            signals.append("Strong momentum up")
        elif price_change < -0.01:  # Lower threshold
            score -= 0.3
            signals.append("Strong momentum down")

        # Determine action - make more sensitive
        if score > 0.2:  # Lower threshold for BUY
            action = "BUY"
        elif score < -0.2:  # Lower threshold for SELL
            action = "SELL"
        else:
            action = "HOLD"

        return {
            "action": action,
            "score": abs(score),
            "confidence": min(abs(score), 1.0),
            "reason": " | ".join(signals),
            "rsi": rsi,
            "price_change": price_change
        }

    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        prices_numeric = pd.to_numeric(prices, errors='coerce')
        delta = prices_numeric.diff()
        delta_numeric = pd.to_numeric(delta, errors='coerce')
        gain = (delta_numeric.where(delta_numeric > 0, 0)).rolling(window=window).mean()
        loss = (-delta_numeric.where(delta_numeric < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    async def generate_signals(self, symbols: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Generate trading signals for multiple symbols"""
        signals = []
        current_time = datetime.now()

        for i, symbol in enumerate(symbols[:limit]):
            try:
                # Get market data
                df = await self.get_binance_data(symbol)

                if df is None or len(df) < 20:
                    continue

                # Analyze
                analysis = self.analyze_trend(df)

                # Only include actionable signals - make more permissive
                if analysis["action"] != "HOLD" and analysis["score"] > 0.1:
                    signal = {
                        "id": f"ai_{symbol}_{int(current_time.timestamp())}",
                        "timestamp": (current_time - timedelta(minutes=i*2)).isoformat(),
                        "symbol": symbol,
                        "side": analysis["action"].lower(),
                        "score": round(analysis["score"], 3),
                        "confidence": round(analysis["confidence"], 3),
                        "details": {
                            "source": "Live AI Analysis",
                            "note": f"RSI: {analysis['rsi']:.1f}, {analysis['reason']}"
                        }
                    }
                    signals.append(signal)

                # Rate limiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue

        return signals


# Global instance
ai_trader = SimpleAITrader()


async def get_live_ai_signals(limit: int = 20, profile: str = "mixed") -> List[Dict[str, Any]]:
    """Generate live AI trading signals"""

    # Symbol selection based on profile
    if profile == "left":  # Conservative
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LTCUSDT"]
    elif profile == "right":  # Aggressive
        symbols = ["SOLUSDT", "AVAXUSDT", "MATICUSDT", "LINKUSDT", "UNIUSDT", "AAVEUSDT", "SUSHIUSDT"]
    else:  # Mixed
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT"]

    try:
        return await ai_trader.generate_signals(symbols, limit)
    except Exception as e:
        logger.error(f"Error generating live signals: {e}")

        return []
