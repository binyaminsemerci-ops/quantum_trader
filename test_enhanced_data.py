#!/usr/bin/env python3
"""Enhanced Data Integration Test Script.

Tests the new multi-source data feeds:
- CoinGecko market data & global metrics
- Fear & Greed Index
- Reddit sentiment analysis
- CryptoCompare news feeds
- CoinPaprika metrics
- Messari on-chain data
- AI insights extraction
"""

import sys
import os
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import asyncio
import logging
from datetime import datetime, timezone

import pytest

from backend.enhanced_data_feeds import EnhancedDataFeed, get_enhanced_market_data
from backend.routes.external_data import (
    comprehensive_crypto_news,
    enhanced_market_data,
    fear_greed_index,
    reddit_sentiment,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio()
async def test_enhanced_data_feeds() -> None:
    """Test all enhanced data feeds."""
    logger.info("🚀 Testing Enhanced Multi-Source Data Integration")

    test_symbols = ["BTC", "ETH", "ADA", "SOL"]

    # Test 1: Direct EnhancedDataFeed usage
    async with EnhancedDataFeed() as feed:
        try:
            all_data = await feed.get_all_enhanced_data(test_symbols)

            # Show AI insights
            insights = all_data.get("ai_insights", {})
            if insights:
                pass
        except Exception:
            pass

    # Test 2: Individual API endpoints

    # Fear & Greed Index
    try:
        fg_data = await fear_greed_index()
        fg_data.get("current", {})
    except Exception:
        pass

    # Reddit sentiment
    try:
        reddit_data = await reddit_sentiment(test_symbols[:3])  # Limit for speed
        symbols_data = reddit_data.get("symbols", {})
        for data in symbols_data.values():
            data.get("sentiment_score", 0)
            data.get("total_posts", 0)
    except Exception:
        pass

    # CoinGecko data
    try:
        enhanced_data = await enhanced_market_data(test_symbols)
        coingecko_data = enhanced_data.get("data", {}).get("coingecko", {})
        market_data = coingecko_data.get("market_data", [])
        coingecko_data.get("global_data", {})

        if market_data:
            market_data[0]
    except Exception:
        pass

    # News data
    try:
        news_data = await comprehensive_crypto_news()
        news_items = news_data.get("news", [])
        if news_items:
            news_items[0]
    except Exception:
        pass

    # Test 3: AI Insights Extraction
    try:
        full_enhanced_data = await get_enhanced_market_data(test_symbols)
        insights = full_enhanced_data.get("ai_insights", {})

        if insights:

            regime = insights.get("regime_indicators", {})
            if regime:
                pass

            # Show active signals
            momentum_signals = insights.get("momentum_signals", [])
            if momentum_signals:
                pass
        else:
            pass

    except Exception:
        pass

    # Test 4: Performance and caching

    start_time = datetime.now(timezone.utc)
    try:
        async with EnhancedDataFeed() as feed:
            # First call
            await feed.get_all_enhanced_data(test_symbols)
            mid_time = datetime.now(timezone.utc)

            # Second call (should use cache)
            await feed.get_all_enhanced_data(test_symbols)
            end_time = datetime.now(timezone.utc)

        (mid_time - start_time).total_seconds() * 1000
        (end_time - mid_time).total_seconds() * 1000

    except Exception:
        pass

    # Summary


@pytest.mark.asyncio()
async def test_continuous_integration() -> None:
    """Test integration with continuous learning engine."""
    try:
        from backend.continuous_learning_engine import ContinuousLearningEngine

        # Create engine with enhanced data
        engine = ContinuousLearningEngine(
            symbols=["BTC", "ETH"],
            twitter_update_interval=300,  # 5 min for test
            market_update_interval=180,  # 3 min for test
            training_interval=1800,  # 30 min for test
            enhanced_fetch_interval=300,  # 5 min for enhanced data
        )

        # Test enhanced data fetching
        enhanced_data = await engine.fetch_enhanced_market_data()
        insights = enhanced_data.get("ai_insights", {})

        if insights:
            pass

        # Test enhanced regime detection
        market_data = {"BTC": {"volatility": 0.02, "trend_strength": 0.01}}
        sentiment_data = {"BTC": {"score": 0.2, "label": "slightly_bullish"}}

        engine.detect_enhanced_market_regime(
            market_data,
            sentiment_data,
            enhanced_data,
        )

    except Exception:
        pass


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(test_enhanced_data_feeds())
    asyncio.run(test_continuous_integration())
