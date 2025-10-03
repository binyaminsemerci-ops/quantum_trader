#!/usr/bin/env python3
"""
Enhanced Live Data Feed & Continuous Learning Engine
Integrates Twitter sentiment, news feeds, live market data for real-time strategy evolution

Quantum Trader AI kontinuerlig læring system som:
- Henter live Twitter sentiment i real-time
- Analyserer nyhetsfeeds og markedsbevegelser
- Trener modell kontinuerlig med fresh data
- Bygger og evolerer egne trading strategier
- Tilpasser strategier basert på market regimes
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import threading
from dataclasses import dataclass
import sqlite3
from pathlib import Path

# Import existing components
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.twitter_client import TwitterClient
from backend.routes.external_data import (
    binance_ohlcv,
    twitter_sentiment,
    enhanced_market_data,
    fear_greed_index,
    reddit_sentiment,
    comprehensive_crypto_news,
    on_chain_metrics,
    market_indicators,
)
from ai_engine.train_and_save import train_and_save
from ai_engine.agents.xgb_agent import make_default_agent

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    """Market regime classification."""

    regime_type: str  # bull, bear, sideways, volatile
    volatility: float
    trend_strength: float
    sentiment_score: float
    timestamp: datetime


@dataclass
class LiveDataPoint:
    """Live data point for continuous learning."""

    symbol: str
    price_data: Dict[str, Any]
    sentiment_data: Dict[str, Any]
    news_data: List[Dict[str, Any]]
    volume_spike: bool
    timestamp: datetime


class ContinuousLearningEngine:
    """Main engine for continuous AI learning from live data feeds."""

    def __init__(
        self,
        symbols: List[str],
        twitter_update_interval: int = 60,  # Update Twitter every 1 min
        market_update_interval: int = 30,  # Update market data every 30 sec
        training_interval: int = 3600,  # Retrain model every hour
        sentiment_threshold: float = 0.3,  # Minimum sentiment impact
        enhanced_fetch_interval: int = 300,
    ):  # Enhanced data every 5 min

        self.symbols = symbols
        self.twitter_update_interval = twitter_update_interval
        self.market_update_interval = market_update_interval
        self.training_interval = training_interval
        self.sentiment_threshold = sentiment_threshold

        # Clients for data feeds
        self.twitter_client = TwitterClient()

        # Enhanced data source tracking
        self.enhanced_data_cache = {}
        self.last_enhanced_fetch = 0
        self.enhanced_fetch_interval = enhanced_fetch_interval

        # Live data storage
        self.live_data: Dict[str, List[LiveDataPoint]] = {
            symbol: [] for symbol in symbols
        }
        self.market_regimes: List[MarketRegime] = []

        # AI learning state
        self.last_training_time = time.time()
        self.model_performance = {"accuracy": 0.5, "sharpe": 0.0, "total_trades": 0}
        self.adaptive_features = set()

        # Threading
        self.is_running = False
        self.twitter_thread: Optional[threading.Thread] = None
        self.market_thread: Optional[threading.Thread] = None
        self.training_thread: Optional[threading.Thread] = None
        self.enhanced_thread: Optional[threading.Thread] = None

        # Database for persistence
        self.db_path = Path("continuous_learning.db")
        self._init_database()

        # Lock for thread safety
        self._lock = threading.Lock()

    def _init_database(self):
        """Initialize continuous learning database."""
        with sqlite3.connect(self.db_path) as conn:
            # Live data points table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS live_data_points (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price_data TEXT,
                    sentiment_data TEXT,
                    news_data TEXT,
                    volume_spike BOOLEAN,
                    timestamp TEXT NOT NULL
                )
            """
            )

            # Market regimes table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS market_regimes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    regime_type TEXT NOT NULL,
                    volatility REAL NOT NULL,
                    trend_strength REAL NOT NULL,
                    sentiment_score REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """
            )

            # Strategy evolution tracking
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_evolution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    features_used TEXT,
                    performance_metrics TEXT,
                    market_regime TEXT,
                    timestamp TEXT NOT NULL
                )
            """
            )

            # Real-time sentiment tracking
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sentiment_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    sentiment_score REAL NOT NULL,
                    tweet_count INTEGER,
                    news_count INTEGER,
                    price_impact REAL,
                    timestamp TEXT NOT NULL
                )
            """
            )

    async def fetch_live_twitter_sentiment(self) -> Dict[str, Dict[str, Any]]:
        """Fetch live Twitter sentiment for all symbols."""
        sentiment_data = {}

        for symbol in self.symbols:
            try:
                # Get symbol without quote currency (BTC from BTCUSDC)
                base_symbol = (
                    symbol.replace("USDC", "").replace("USDT", "").replace("USD", "")
                )

                # Fetch Twitter sentiment
                sentiment = self.twitter_client.sentiment_for_symbol(
                    base_symbol, max_results=50
                )

                # Enhanced sentiment analysis
                enhanced_sentiment = await self._analyze_enhanced_sentiment(
                    base_symbol, sentiment
                )

                sentiment_data[symbol] = enhanced_sentiment

                logger.debug(
                    f"📱 Twitter sentiment for {symbol}: {sentiment['score']:.3f} ({sentiment['label']})"
                )

            except Exception as e:
                logger.error(f"Failed to fetch Twitter sentiment for {symbol}: {e}")
                sentiment_data[symbol] = {
                    "score": 0.0,
                    "label": "neutral",
                    "source": "error",
                }

        return sentiment_data

    async def fetch_enhanced_market_data(self) -> Dict[str, Any]:
        """Fetch enhanced market data from multiple sources."""
        current_time = time.time()

        # Check if we need to fetch fresh enhanced data
        if current_time - self.last_enhanced_fetch < self.enhanced_fetch_interval:
            return self.enhanced_data_cache

        logger.info("🔄 Fetching enhanced market data from multiple sources...")

        enhanced_data = {}

        try:
            # Fetch data from all enhanced sources concurrently
            tasks = [
                enhanced_market_data(self.symbols),
                fear_greed_index(),
                reddit_sentiment(self.symbols),
                comprehensive_crypto_news(),
                on_chain_metrics(self.symbols),
                market_indicators(),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            enhanced_data = {
                "multi_source_data": (
                    results[0] if not isinstance(results[0], Exception) else {}
                ),
                "fear_greed": (
                    results[1] if not isinstance(results[1], Exception) else {}
                ),
                "reddit": results[2] if not isinstance(results[2], Exception) else {},
                "news": results[3] if not isinstance(results[3], Exception) else {},
                "on_chain": results[4] if not isinstance(results[4], Exception) else {},
                "indicators": (
                    results[5] if not isinstance(results[5], Exception) else {}
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Extract key insights for AI learning
            enhanced_data["ai_insights"] = self._extract_ai_insights(enhanced_data)

            # Cache the results
            self.enhanced_data_cache = enhanced_data
            self.last_enhanced_fetch = current_time

            # Log summary
            sources_active = sum(
                1
                for key, data in enhanced_data.items()
                if isinstance(data, dict)
                and data.get("source")
                and key != "ai_insights"
            )
            logger.info(
                f"✅ Enhanced data fetched from {sources_active} active sources"
            )

        except Exception as e:
            logger.error(f"Error fetching enhanced market data: {e}")
            enhanced_data = {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        return enhanced_data

    def _extract_ai_insights(self, enhanced_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract actionable AI insights from enhanced data."""
        insights = {
            "market_sentiment": "neutral",
            "momentum_signals": [],
            "risk_factors": [],
            "opportunity_signals": [],
            "regime_indicators": {},
        }

        try:
            # Fear & Greed Index insights
            fear_greed = enhanced_data.get("fear_greed", {}).get("current", {})
            if fear_greed:
                fg_value = fear_greed.get("value", 50)
                fg_classification = fear_greed.get("value_classification", "Neutral")

                insights["market_sentiment"] = fg_classification.lower()

                # Extreme fear/greed signals
                if fg_value <= 25:
                    insights["opportunity_signals"].append(
                        "extreme_fear_buy_opportunity"
                    )
                elif fg_value >= 75:
                    insights["risk_factors"].append("extreme_greed_sell_signal")

            # Reddit sentiment analysis
            reddit_data = enhanced_data.get("reddit", {}).get("symbols", {})
            for symbol, reddit_info in reddit_data.items():
                sentiment_score = reddit_info.get("sentiment_score", 0)
                total_posts = reddit_info.get("total_posts", 0)

                if total_posts > 5:  # Significant discussion
                    if sentiment_score > 0.3:
                        insights["momentum_signals"].append(f"{symbol}_reddit_bullish")
                    elif sentiment_score < -0.3:
                        insights["momentum_signals"].append(f"{symbol}_reddit_bearish")

            # CoinGecko market data insights
            coingecko_data = (
                enhanced_data.get("multi_source_data", {})
                .get("data", {})
                .get("coingecko", {})
            )
            market_data = coingecko_data.get("market_data", [])

            for coin in market_data:
                symbol = coin.get("symbol", "").upper()
                price_change_24h = coin.get("price_change_percentage_24h", 0)
                volume_24h = coin.get("total_volume", 0)
                market_cap_rank = coin.get("market_cap_rank", 999)

                # Volume spike detection
                if volume_24h and coin.get("market_cap", 0):
                    volume_ratio = volume_24h / coin["market_cap"]
                    if volume_ratio > 0.1:  # 10% of market cap in 24h volume
                        insights["momentum_signals"].append(f"{symbol}_volume_spike")

                # Momentum signals from top coins
                if market_cap_rank <= 20:  # Top 20 coins
                    if price_change_24h > 10:
                        insights["momentum_signals"].append(f"{symbol}_strong_bullish")
                    elif price_change_24h < -10:
                        insights["momentum_signals"].append(f"{symbol}_strong_bearish")

            # News sentiment analysis
            news_data = enhanced_data.get("news", {}).get("news", [])
            positive_news = 0
            negative_news = 0

            for news_item in news_data[:10]:  # Analyze recent news
                title = news_item.get("title", "").lower()

                # Simple keyword-based sentiment
                positive_keywords = [
                    "bullish",
                    "rally",
                    "surge",
                    "adoption",
                    "partnership",
                    "breakthrough",
                ]
                negative_keywords = [
                    "crash",
                    "dump",
                    "hack",
                    "regulation",
                    "ban",
                    "selloff",
                ]

                if any(keyword in title for keyword in positive_keywords):
                    positive_news += 1
                elif any(keyword in title for keyword in negative_keywords):
                    negative_news += 1

            if positive_news > negative_news and positive_news > 2:
                insights["momentum_signals"].append("positive_news_sentiment")
            elif negative_news > positive_news and negative_news > 2:
                insights["risk_factors"].append("negative_news_sentiment")

            # Global market indicators
            global_data = coingecko_data.get("global_data", {})
            if global_data:
                btc_dominance = global_data.get("market_cap_percentage", {}).get(
                    "btc", 0
                )
                total_market_cap = global_data.get("total_market_cap", {}).get("usd", 0)

                insights["regime_indicators"] = {
                    "btc_dominance": btc_dominance,
                    "total_market_cap": total_market_cap,
                    "market_stage": (
                        "altcoin_season" if btc_dominance < 40 else "btc_dominance"
                    ),
                }

        except Exception as e:
            logger.error(f"Error extracting AI insights: {e}")
            insights["error"] = str(e)

        return insights

    async def _analyze_enhanced_sentiment(
        self, symbol: str, base_sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhanced sentiment analysis with market context."""
        enhanced = base_sentiment.copy()

        try:
            # Add sentiment velocity (rate of change)
            recent_sentiment = self._get_recent_sentiment(symbol, minutes=30)
            if len(recent_sentiment) > 1:
                sentiment_velocity = (
                    enhanced["score"] - recent_sentiment[-2]
                ) / 0.5  # per hour
                enhanced["velocity"] = sentiment_velocity
            else:
                enhanced["velocity"] = 0.0

            # Add sentiment strength based on tweet volume
            tweet_volume = base_sentiment.get("tweet_count", 20)
            enhanced["strength"] = min(1.0, tweet_volume / 100.0)  # Normalize to [0,1]

            # Market-sentiment divergence detection
            price_trend = await self._get_recent_price_trend(symbol)
            sentiment_score = enhanced["score"]

            # Divergence: positive sentiment but falling price (or vice versa)
            if (sentiment_score > 0.2 and price_trend < -0.01) or (
                sentiment_score < -0.2 and price_trend > 0.01
            ):
                enhanced["divergence"] = True
                enhanced["divergence_strength"] = abs(sentiment_score - price_trend)
            else:
                enhanced["divergence"] = False
                enhanced["divergence_strength"] = 0.0

            return enhanced

        except Exception as e:
            logger.error(f"Enhanced sentiment analysis failed for {symbol}: {e}")
            return enhanced

    def _get_recent_sentiment(self, symbol: str, minutes: int = 30) -> List[float]:
        """Get recent sentiment scores for sentiment velocity calculation."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
                cursor = conn.execute(
                    """
                    SELECT sentiment_score FROM sentiment_tracking 
                    WHERE symbol = ? AND timestamp > ?
                    ORDER BY timestamp ASC
                """,
                    (symbol, cutoff_time.isoformat()),
                )

                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get recent sentiment for {symbol}: {e}")
            return []

    async def _get_recent_price_trend(self, symbol: str, minutes: int = 30) -> float:
        """Get recent price trend for divergence analysis."""
        try:
            # Fetch recent price data
            market_data = await binance_ohlcv(symbol=symbol, limit=10)
            candles = market_data.get("candles", [])

            if len(candles) >= 2:
                latest_price = float(candles[-1]["close"])
                earlier_price = float(candles[-2]["close"])
                return (latest_price - earlier_price) / earlier_price

            return 0.0

        except Exception as e:
            logger.error(f"Failed to get price trend for {symbol}: {e}")
            return 0.0

    async def fetch_live_market_data(self) -> Dict[str, Dict[str, Any]]:
        """Fetch live market data with enhanced analytics."""
        market_data = {}

        for symbol in self.symbols:
            try:
                # Fetch OHLCV data
                ohlcv_response = await binance_ohlcv(symbol=symbol, limit=120)
                candles = ohlcv_response.get("candles", [])

                if candles:
                    # Enhanced market analysis
                    enhanced_data = await self._analyze_enhanced_market_data(
                        symbol, candles
                    )
                    market_data[symbol] = enhanced_data

                    logger.debug(
                        f"📊 Market data for {symbol}: Price={enhanced_data['current_price']:.4f}, Vol Spike={enhanced_data['volume_spike']}"
                    )

            except Exception as e:
                logger.error(f"Failed to fetch market data for {symbol}: {e}")
                market_data[symbol] = {"error": str(e)}

        return market_data

    async def _analyze_enhanced_market_data(
        self, symbol: str, candles: List[Dict]
    ) -> Dict[str, Any]:
        """Enhanced market data analysis."""
        try:
            if len(candles) < 20:
                return {"error": "Insufficient data"}

            latest = candles[-1]
            current_price = float(latest["close"])
            current_volume = float(latest["volume"])

            # Calculate enhanced metrics
            prices = [float(c["close"]) for c in candles[-20:]]
            volumes = [float(c["volume"]) for c in candles[-20:]]

            # Volatility calculation (20-period)
            price_changes = [
                (prices[i] - prices[i - 1]) / prices[i - 1]
                for i in range(1, len(prices))
            ]
            volatility = sum(abs(change) for change in price_changes) / len(
                price_changes
            )

            # Volume spike detection
            avg_volume = sum(volumes[:-1]) / len(volumes[:-1])
            volume_spike = current_volume > avg_volume * 1.5

            # Trend strength calculation
            if len(prices) >= 10:
                short_ma = sum(prices[-5:]) / 5
                long_ma = sum(prices[-10:]) / 10
                trend_strength = (short_ma - long_ma) / long_ma
            else:
                trend_strength = 0.0

            # Support/Resistance levels
            highs = [float(c["high"]) for c in candles[-20:]]
            lows = [float(c["low"]) for c in candles[-20:]]
            resistance_level = max(highs)
            support_level = min(lows)

            # Price position within range
            price_position = (
                (current_price - support_level) / (resistance_level - support_level)
                if resistance_level > support_level
                else 0.5
            )

            return {
                "current_price": current_price,
                "volatility": volatility,
                "volume_spike": volume_spike,
                "trend_strength": trend_strength,
                "support_level": support_level,
                "resistance_level": resistance_level,
                "price_position": price_position,
                "candles": candles,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Enhanced market analysis failed for {symbol}: {e}")
            return {"error": str(e)}

    def detect_market_regime(
        self, market_data: Dict[str, Any], sentiment_data: Dict[str, Any]
    ) -> MarketRegime:
        """Detect current market regime based on multiple factors."""
        try:
            # Aggregate metrics across all symbols
            avg_volatility = sum(
                data.get("volatility", 0)
                for data in market_data.values()
                if isinstance(data, dict) and "volatility" in data
            ) / max(1, len(market_data))

            avg_trend_strength = sum(
                data.get("trend_strength", 0)
                for data in market_data.values()
                if isinstance(data, dict) and "trend_strength" in data
            ) / max(1, len(market_data))

            avg_sentiment = sum(
                data.get("score", 0)
                for data in sentiment_data.values()
                if isinstance(data, dict) and "score" in data
            ) / max(1, len(sentiment_data))

            # Classify market regime
            if avg_volatility > 0.03:  # High volatility threshold
                regime_type = "volatile"
            elif avg_trend_strength > 0.02:
                regime_type = "bull" if avg_sentiment > 0 else "bear"
            elif avg_trend_strength < -0.02:
                regime_type = "bear" if avg_sentiment < 0 else "bull"
            else:
                regime_type = "sideways"

            regime = MarketRegime(
                regime_type=regime_type,
                volatility=avg_volatility,
                trend_strength=avg_trend_strength,
                sentiment_score=avg_sentiment,
                timestamp=datetime.now(timezone.utc),
            )

            logger.info(
                f"🌊 Market Regime: {regime_type.upper()} (Vol: {avg_volatility:.3f}, Trend: {avg_trend_strength:.3f}, Sent: {avg_sentiment:.3f})"
            )

            return regime

        except Exception as e:
            logger.error(f"Market regime detection failed: {e}")
            return MarketRegime("unknown", 0.0, 0.0, 0.0, datetime.now(timezone.utc))

    def detect_enhanced_market_regime(
        self,
        market_data: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        enhanced_data: Dict[str, Any],
    ) -> MarketRegime:
        """Enhanced market regime detection using multi-source data."""
        try:
            # Traditional metrics
            avg_volatility = sum(
                data.get("volatility", 0)
                for data in market_data.values()
                if isinstance(data, dict) and "volatility" in data
            ) / max(1, len(market_data))

            avg_trend_strength = sum(
                data.get("trend_strength", 0)
                for data in market_data.values()
                if isinstance(data, dict) and "trend_strength" in data
            ) / max(1, len(market_data))

            avg_sentiment = sum(
                data.get("score", 0)
                for data in sentiment_data.values()
                if isinstance(data, dict) and "score" in data
            ) / max(1, len(sentiment_data))

            # Enhanced factors
            ai_insights = enhanced_data.get("ai_insights", {})
            fear_greed = enhanced_data.get("fear_greed", {}).get("current", {})
            regime_indicators = ai_insights.get("regime_indicators", {})

            # Fear & Greed Index influence
            fg_value = fear_greed.get("value", 50)
            fg_factor = (fg_value - 50) / 50  # Normalize to [-1, 1]

            # Combine sentiment sources
            reddit_sentiment = 0.0
            reddit_data = enhanced_data.get("reddit", {}).get("symbols", {})
            if reddit_data:
                reddit_scores = [
                    data.get("sentiment_score", 0) for data in reddit_data.values()
                ]
                reddit_sentiment = (
                    sum(reddit_scores) / len(reddit_scores) if reddit_scores else 0.0
                )

            # Weighted sentiment (Twitter 60%, Fear&Greed 30%, Reddit 10%)
            combined_sentiment = (
                0.6 * avg_sentiment + 0.3 * fg_factor + 0.1 * reddit_sentiment
            )

            # News sentiment impact
            momentum_signals = ai_insights.get("momentum_signals", [])
            risk_factors = ai_insights.get("risk_factors", [])

            momentum_score = len(momentum_signals) - len(risk_factors)
            sentiment_adjustment = momentum_score * 0.1  # Small adjustment

            adjusted_sentiment = combined_sentiment + sentiment_adjustment

            # Enhanced regime classification
            btc_dominance = regime_indicators.get("btc_dominance", 50)

            # Volatility thresholds adjusted by market conditions
            vol_threshold = 0.03
            if btc_dominance > 60:  # BTC dominance phase, lower vol threshold
                vol_threshold = 0.025
            elif btc_dominance < 40:  # Altcoin season, higher vol threshold
                vol_threshold = 0.04

            # Classify enhanced regime
            if avg_volatility > vol_threshold:
                if fg_value < 20:  # Extreme fear
                    regime_type = "panic_sell"
                elif fg_value > 80:  # Extreme greed
                    regime_type = "euphoric_buy"
                else:
                    regime_type = "volatile"
            elif adjusted_sentiment > 0.3:
                if btc_dominance < 40:
                    regime_type = "altcoin_bull"
                else:
                    regime_type = "btc_bull"
            elif adjusted_sentiment < -0.3:
                regime_type = "bear_market"
            elif abs(avg_trend_strength) < 0.01:
                regime_type = "accumulation" if fg_value < 40 else "distribution"
            else:
                regime_type = "sideways"

            enhanced_regime = MarketRegime(
                regime_type=regime_type,
                volatility=avg_volatility,
                trend_strength=avg_trend_strength,
                sentiment_score=adjusted_sentiment,
                timestamp=datetime.now(timezone.utc),
            )

            logger.info(
                f"🌊 Enhanced Market Regime: {regime_type.upper()} "
                f"(Vol: {avg_volatility:.3f}, Trend: {avg_trend_strength:.3f}, "
                f"Sent: {adjusted_sentiment:.3f}, F&G: {fg_value})"
            )

            return enhanced_regime

        except Exception as e:
            logger.error(f"Enhanced market regime detection failed: {e}")
            # Fall back to traditional detection
            return self.detect_market_regime(market_data, sentiment_data)

    async def continuous_twitter_feed(self):
        """Continuous Twitter sentiment monitoring."""
        logger.info("🐦 Starting continuous Twitter feed monitoring...")

        while self.is_running:
            try:
                sentiment_data = await self.fetch_live_twitter_sentiment()

                # Store sentiment data
                with self._lock:
                    for symbol, sentiment in sentiment_data.items():
                        self._store_sentiment_data(symbol, sentiment)

                await asyncio.sleep(self.twitter_update_interval)

            except Exception as e:
                logger.error(f"Twitter feed error: {e}")
                await asyncio.sleep(30)  # Wait before retrying

    async def continuous_market_feed(self):
        """Enhanced continuous market data monitoring with multi-source integration."""
        logger.info("📈 Starting enhanced continuous market data monitoring...")

        while self.is_running:
            try:
                # Fetch all data sources
                market_data = await self.fetch_live_market_data()
                sentiment_data = await self.fetch_live_twitter_sentiment()
                enhanced_data = await self.fetch_enhanced_market_data()

                # Detect market regime with enhanced data
                regime = self.detect_enhanced_market_regime(
                    market_data, sentiment_data, enhanced_data
                )

                # Store data
                with self._lock:
                    for symbol in self.symbols:
                        if symbol in market_data and symbol in sentiment_data:
                            self._store_enhanced_data_point(
                                symbol,
                                market_data[symbol],
                                sentiment_data[symbol],
                                enhanced_data,
                            )

                    self.market_regimes.append(regime)
                    self._store_market_regime(regime)

                    # Store enhanced insights for AI learning
                    if enhanced_data.get("ai_insights"):
                        self._store_ai_insights(enhanced_data["ai_insights"])

                await asyncio.sleep(self.market_update_interval)

            except Exception as e:
                logger.error(f"Enhanced market feed error: {e}")
                await asyncio.sleep(30)

    async def continuous_enhanced_feed(self):
        """Continuous enhanced data monitoring from multiple sources."""
        logger.info("🔄 Starting continuous enhanced data feed monitoring...")

        while self.is_running:
            try:
                # Fetch enhanced data every 5 minutes
                enhanced_data = await self.fetch_enhanced_market_data()

                # Extract and log key insights
                insights = enhanced_data.get("ai_insights", {})
                if insights:
                    logger.info(
                        f"🧠 AI Insights - Sentiment: {insights.get('market_sentiment', 'neutral')}, "
                        f"Signals: {len(insights.get('momentum_signals', []))}, "
                        f"Risks: {len(insights.get('risk_factors', []))}"
                    )

                # Store insights for AI learning
                with self._lock:
                    if insights:
                        self._store_ai_insights(insights)

                await asyncio.sleep(self.enhanced_fetch_interval)

            except Exception as e:
                logger.error(f"Enhanced feed error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def continuous_learning_loop(self):
        """Continuous AI model retraining loop."""
        logger.info("🤖 Starting continuous learning loop...")

        while self.is_running:
            try:
                current_time = time.time()

                if current_time - self.last_training_time >= self.training_interval:
                    logger.info("🔄 Starting model retraining with fresh data...")

                    # Retrain model with latest data
                    await self._retrain_model_with_live_data()

                    self.last_training_time = current_time

                    logger.info("✅ Model retraining completed!")

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Continuous learning error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def _retrain_model_with_live_data(self):
        """Retrain AI model with accumulated live data including enhanced multi-source features."""
        try:
            # Get comprehensive data for training
            fresh_market_data = await self.fetch_live_market_data()
            fresh_sentiment_data = await self.fetch_live_twitter_sentiment()
            enhanced_data = await self.fetch_enhanced_market_data()

            # Analyze current market conditions for adaptive training
            current_regime = self.detect_enhanced_market_regime(
                fresh_market_data, fresh_sentiment_data, enhanced_data
            )

            # Extract AI insights for feature engineering
            ai_insights = enhanced_data.get("ai_insights", {})

            # Adaptive training parameters based on market conditions
            training_symbols = self.symbols[:5]  # Limit for performance
            training_limit = 600

            # Adjust training based on market regime
            if current_regime.regime_type in ["panic_sell", "euphoric_buy"]:
                training_limit = 400  # Use shorter history for extreme conditions
            elif current_regime.regime_type == "volatile":
                training_limit = 800  # Use more data for volatile markets

            # Enhanced feature engineering
            enhanced_features = self._extract_enhanced_features(
                enhanced_data, ai_insights
            )

            logger.info(
                f"🧠 Retraining with {len(enhanced_features)} enhanced features in {current_regime.regime_type} regime"
            )

            # Train with enhanced features including sentiment and regime data
            result = train_and_save(
                symbols=training_symbols,
                limit=training_limit,
                use_live_data=True,
                write_report=True,
                enhanced_features=enhanced_features,  # Pass enhanced features
            )

            # Update performance tracking with enhanced metrics
            new_accuracy = result.get("metrics", {}).get("directional_accuracy", 0.5)
            sharpe_ratio = result.get("metrics", {}).get("sharpe_ratio", 0.0)

            with self._lock:
                self.model_performance["accuracy"] = new_accuracy
                self.model_performance["sharpe"] = sharpe_ratio
                self.model_performance["total_trades"] += 1

                # Update adaptive features based on performance
                if new_accuracy > 0.6:  # Good performance
                    self.adaptive_features.update(enhanced_features.keys())
                elif new_accuracy < 0.4:  # Poor performance, reset features
                    self.adaptive_features.clear()

            # Store enhanced strategy evolution
            enhanced_result = result.copy()
            enhanced_result["market_regime"] = current_regime.regime_type
            enhanced_result["enhanced_features_count"] = len(enhanced_features)
            enhanced_result["ai_insights"] = ai_insights

            self._store_strategy_evolution(enhanced_result)

            logger.info(
                f"🎯 Enhanced model retrained: Accuracy={new_accuracy:.3f}, "
                f"Sharpe={sharpe_ratio:.3f}, Regime={current_regime.regime_type}"
            )

        except Exception as e:
            logger.error(f"Enhanced model retraining failed: {e}")

    def _extract_enhanced_features(
        self, enhanced_data: Dict[str, Any], ai_insights: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract quantitative features from enhanced data sources."""
        features = {}

        try:
            # Fear & Greed Index features
            fear_greed = enhanced_data.get("fear_greed", {}).get("current", {})
            if fear_greed:
                features["fear_greed_index"] = float(fear_greed.get("value", 50)) / 100
                features["fear_greed_extreme"] = (
                    1.0
                    if fear_greed.get("value", 50) < 20
                    or fear_greed.get("value", 50) > 80
                    else 0.0
                )

            # Market sentiment features
            sentiment = ai_insights.get("market_sentiment", "neutral")
            features["sentiment_bullish"] = (
                1.0 if sentiment in ["bullish", "extreme_greed"] else 0.0
            )
            features["sentiment_bearish"] = (
                1.0 if sentiment in ["bearish", "extreme_fear"] else 0.0
            )

            # Signal strength features
            momentum_signals = ai_insights.get("momentum_signals", [])
            risk_factors = ai_insights.get("risk_factors", [])
            opportunity_signals = ai_insights.get("opportunity_signals", [])

            features["momentum_strength"] = len(momentum_signals) / 10.0  # Normalize
            features["risk_level"] = len(risk_factors) / 10.0
            features["opportunity_score"] = len(opportunity_signals) / 10.0
            features["signal_divergence"] = (
                abs(len(momentum_signals) - len(risk_factors)) / 10.0
            )

            # Market regime features
            regime_indicators = ai_insights.get("regime_indicators", {})
            if regime_indicators:
                btc_dominance = regime_indicators.get("btc_dominance", 50)
                features["btc_dominance"] = btc_dominance / 100
                features["altcoin_season"] = 1.0 if btc_dominance < 40 else 0.0
                features["btc_dominance_high"] = 1.0 if btc_dominance > 60 else 0.0

            # Global market features
            coingecko_data = (
                enhanced_data.get("multi_source_data", {})
                .get("data", {})
                .get("coingecko", {})
            )
            global_data = coingecko_data.get("global_data", {})
            if global_data:
                total_mcap = global_data.get("total_market_cap", {}).get("usd", 0)
                features["market_cap_trillions"] = (
                    total_mcap / 1e12 if total_mcap else 0.0
                )

            # Reddit sentiment aggregation
            reddit_data = enhanced_data.get("reddit", {}).get("symbols", {})
            if reddit_data:
                reddit_scores = [
                    data.get("sentiment_score", 0) for data in reddit_data.values()
                ]
                avg_reddit_sentiment = (
                    sum(reddit_scores) / len(reddit_scores) if reddit_scores else 0.0
                )
                features["reddit_sentiment"] = max(
                    -1.0, min(1.0, avg_reddit_sentiment)
                )  # Clamp to [-1, 1]
                features["reddit_activity"] = len(reddit_data) / 10.0  # Normalize

            logger.debug(f"🔧 Extracted {len(features)} enhanced features")

        except Exception as e:
            logger.error(f"Error extracting enhanced features: {e}")

        return features

    def _store_live_data_point(
        self, symbol: str, market_data: Dict, sentiment_data: Dict
    ):
        """Store live data point to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO live_data_points 
                    (symbol, price_data, sentiment_data, volume_spike, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        symbol,
                        json.dumps(market_data),
                        json.dumps(sentiment_data),
                        market_data.get("volume_spike", False),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to store live data point: {e}")

    def _store_enhanced_data_point(
        self, symbol: str, market_data: Dict, sentiment_data: Dict, enhanced_data: Dict
    ):
        """Store enhanced data point with multi-source information."""
        try:
            # Store traditional data point
            self._store_live_data_point(symbol, market_data, sentiment_data)

            # Store enhanced data in news_data field (repurposing for enhanced data)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE live_data_points 
                    SET news_data = ?
                    WHERE symbol = ? AND timestamp = (
                        SELECT MAX(timestamp) FROM live_data_points WHERE symbol = ?
                    )
                """,
                    (
                        json.dumps(
                            {
                                "enhanced_data": enhanced_data,
                                "ai_insights": enhanced_data.get("ai_insights", {}),
                            }
                        ),
                        symbol,
                        symbol,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to store enhanced data point: {e}")

    def _store_ai_insights(self, insights: Dict):
        """Store AI insights for learning enhancement."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create AI insights table if it doesn't exist
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ai_insights (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        market_sentiment TEXT,
                        momentum_signals TEXT,
                        risk_factors TEXT,
                        opportunity_signals TEXT,
                        regime_indicators TEXT,
                        timestamp TEXT NOT NULL
                    )
                """
                )

                # Store the insights
                conn.execute(
                    """
                    INSERT INTO ai_insights
                    (market_sentiment, momentum_signals, risk_factors, 
                     opportunity_signals, regime_indicators, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        insights.get("market_sentiment", "neutral"),
                        json.dumps(insights.get("momentum_signals", [])),
                        json.dumps(insights.get("risk_factors", [])),
                        json.dumps(insights.get("opportunity_signals", [])),
                        json.dumps(insights.get("regime_indicators", {})),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to store AI insights: {e}")

    def _store_sentiment_data(self, symbol: str, sentiment: Dict):
        """Store sentiment data to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO sentiment_tracking
                    (symbol, sentiment_score, tweet_count, news_count, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        symbol,
                        sentiment.get("score", 0.0),
                        sentiment.get("tweet_count", 0),
                        sentiment.get("news_count", 0),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to store sentiment data: {e}")

    def _store_market_regime(self, regime: MarketRegime):
        """Store market regime to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO market_regimes
                    (regime_type, volatility, trend_strength, sentiment_score, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        regime.regime_type,
                        regime.volatility,
                        regime.trend_strength,
                        regime.sentiment_score,
                        regime.timestamp.isoformat(),
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to store market regime: {e}")

    def _store_strategy_evolution(self, training_result: Dict):
        """Store strategy evolution tracking."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO strategy_evolution
                    (model_version, features_used, performance_metrics, timestamp)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        f"v{int(time.time())}",
                        json.dumps(training_result.get("features", [])),
                        json.dumps(training_result.get("metrics", {})),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to store strategy evolution: {e}")

    def start_continuous_learning(self):
        """Start all continuous learning threads."""
        if self.is_running:
            logger.warning("Continuous learning already running")
            return

        self.is_running = True

        # Start Twitter feed thread
        self.twitter_thread = threading.Thread(
            target=asyncio.run, args=(self.continuous_twitter_feed(),)
        )
        self.twitter_thread.daemon = True
        self.twitter_thread.start()

        # Start market feed thread
        self.market_thread = threading.Thread(
            target=asyncio.run, args=(self.continuous_market_feed(),)
        )
        self.market_thread.daemon = True
        self.market_thread.start()

        # Start learning loop thread
        self.training_thread = threading.Thread(
            target=asyncio.run, args=(self.continuous_learning_loop(),)
        )
        self.training_thread.daemon = True
        self.training_thread.start()

        # Start enhanced data collection thread
        self.enhanced_thread = threading.Thread(
            target=asyncio.run, args=(self.continuous_enhanced_feed(),)
        )
        self.enhanced_thread.daemon = True
        self.enhanced_thread.start()

        logger.info("🚀 Enhanced Continuous Learning Engine started!")
        logger.info(f"🐦 Twitter updates: every {self.twitter_update_interval}s")
        logger.info(f"📊 Market updates: every {self.market_update_interval}s")
        logger.info(f"🔄 Enhanced data: every {self.enhanced_fetch_interval}s")
        logger.info(f"🤖 Model retraining: every {self.training_interval}s")

    def stop_continuous_learning(self):
        """Stop continuous learning."""
        self.is_running = False

        if self.twitter_thread:
            self.twitter_thread.join(timeout=5)
        if self.market_thread:
            self.market_thread.join(timeout=5)
        if self.training_thread:
            self.training_thread.join(timeout=5)
        if self.enhanced_thread:
            self.enhanced_thread.join(timeout=5)

        logger.info("🛑 Continuous Learning Engine stopped!")

    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status and metrics."""
        with self._lock:
            return {
                "is_running": self.is_running,
                "symbols_monitored": len(self.symbols),
                "last_training": self.last_training_time,
                "model_performance": self.model_performance.copy(),
                "current_regime": (
                    self.market_regimes[-1].regime_type
                    if self.market_regimes
                    else "unknown"
                ),
                "data_points_collected": len(sum(self.live_data.values(), [])),
                "adaptive_features": list(self.adaptive_features),
            }


def create_continuous_learning_service(symbols: List[str]) -> ContinuousLearningEngine:
    """Factory function to create continuous learning service."""
    return ContinuousLearningEngine(
        symbols=symbols,
        twitter_update_interval=60,  # 1 minute Twitter updates
        market_update_interval=30,  # 30 second market updates
        training_interval=3600,  # 1 hour model retraining
        sentiment_threshold=0.3,
    )


if __name__ == "__main__":
    # Test continuous learning engine
    symbols = ["BTCUSDC", "ETHUSDC", "ADAUSDC"]
    engine = create_continuous_learning_service(symbols)

    try:
        engine.start_continuous_learning()

        # Run for demonstration
        time.sleep(300)  # 5 minutes

        status = engine.get_learning_status()
        print("Learning Status:", json.dumps(status, indent=2))

    finally:
        engine.stop_continuous_learning()
